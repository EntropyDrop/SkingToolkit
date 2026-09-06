"""Differentiable cell-balanced UV supervision using rendered ground truth.

UV indices are training labels only. No mirror operation is applied at inference.
"""
import torch
import torch.nn.functional as F
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology

VIEWS = ("front_left", "back_left", "front_right", "back_right")


@torch.no_grad()
def render_structured_batch(batch, renderer):
    images,truth,faces,indices,valids = [],[],[],[],[]
    for view in VIEWS:
        image,t = build_dense_parser_batch(batch["uv"],renderer,view)
        valid = (t["foreground"][:,0]>.5) & (t["part"] == 0)
        xy = (t["uv"]*63).round().long().clamp(0,63)
        index = xy[:,1]*64+xy[:,0]
        label = batch["labels"].flatten(1).gather(1,index.flatten(1)).reshape_as(index)
        images.append(image);truth.append(label.masked_fill(~valid,0))
        faces.append((t["face"]+1).masked_fill(~valid,0))
        indices.append(index);valids.append(valid)
    pack = lambda xs: torch.stack(xs,1).flatten(0,1)
    return tuple(pack(xs) for xs in (images,truth,faces,indices,valids))


def pool_uv(probabilities, indices, valid, views=4):
    n,c,h,w = probabilities.shape
    if n % views:
        raise ValueError("Incomplete paired views")
    b = n//views
    p = probabilities.reshape(b,views,c,h*w).permute(0,2,1,3).reshape(b,c,-1)
    index = indices.reshape(b,-1)
    weight = valid.reshape(b,-1).float()
    counts = p.new_zeros(b,4096).scatter_add(1,index,weight)
    sums = p.new_zeros(b,c,4096).scatter_add(2,index[:,None].expand(-1,c,-1),p*weight[:,None])
    return sums/counts[:,None].clamp_min(1),counts


def structured_uv_loss(logits, indices, valid, labels, mode, symmetric):
    pooled,counts = pool_uv(logits.float().softmax(1),indices,valid)
    target = labels.flatten(1)
    visible = (counts>=1) & (mode[:,None] == 0) & (target>0)
    weights = torch.where(torch.isin(target,target.new_tensor([2,3,4,5])),3.,1.)
    ce = -pooled.gather(1,target[:,None].clamp(0,13)).squeeze(1).clamp_min(1e-6).log()
    cell = (ce*weights*visible).sum()/(weights*visible).sum().clamp_min(1)
    topology = build_simple_uv_topology()
    mirror = topology.mirrored_texel.to(logits.device).reshape(-1)
    beard = (target == 3) | (target == 5)
    # Only explicitly symmetric GT beard pairs visible in the training views.
    pairs = visible & visible[:,mirror] & beard & (target == target[:,mirror]) & symmetric[:,None]
    error = (pooled-pooled[:,:,mirror]).square().sum(1)
    symmetry = (error*pairs).sum()/pairs.sum().clamp_min(1)
    # Preserve coexistence: compare beard probability across matching layers,
    # not a single winning layer for the entire connected component.
    outer_ids = torch.nonzero((topology.valid & (topology.part==0) & (topology.layer==1)).reshape(-1)).flatten().to(logits.device)
    inner_ids = outer_ids-32
    mixed = visible[:,outer_ids] & visible[:,inner_ids] & (target[:,outer_ids]==5) & (target[:,inner_ids]==3)
    delta = pooled[:,5,outer_ids]-pooled[:,3,inner_ids]
    alignment = (delta.square()*mixed).sum()/mixed.sum().clamp_min(1)
    return cell + .3*symmetry + .3*alignment, {
        "uv_cell":cell.detach(),"uv_mirror":symmetry.detach(),"uv_layers":alignment.detach(),
        "visible_cells":visible.sum().detach(),"mirror_cells":pairs.sum().detach(),"mixed_cells":mixed.sum().detach(),
    }
