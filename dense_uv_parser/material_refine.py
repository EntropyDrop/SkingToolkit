"""Fit visible head materials while keeping ownership and hidden layers isolated."""
import torch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
from SkingToolkit.dense_uv_parser.accessories import head_bounds


def refine_head_material(uv, images, sources, renderer, views, steps=48, ownership=None, inner_exclusion=None, inner_texel_support=None, headwear_layer=None, headwear_uv=None, headwear_family=None, protect_inner_footprints=False):
    original=uv.detach().float()
    topology=build_simple_uv_topology()
    head=(topology.valid&(topology.part==0)).to(uv.device)[None,None]
    inner=head&(topology.layer.to(uv.device)[None,None]==0)
    outer=head&~inner
    trainable_inner=inner if inner_texel_support is None else inner&inner_texel_support[:,None]
    valid=torch.zeros_like(sources,dtype=torch.bool)
    y0,y1,x0,x1=head_bounds(*sources.shape[-2:])
    valid[:,y0:y1,x0:x1]=sources[:,y0:y1,x0:x1]
    if not valid.any():return original
    target=images[:,:3].detach().float()
    def render(skin):
        return torch.stack([renderer.forward_view(skin,v) for v in views],1).flatten(0,1)[:,:3]
    excluded = torch.zeros_like(valid)
    if ownership is not None:
        probability,identity=ownership.float().softmax(1).max(1)
        excluded |= ((identity==3)|(identity==4))&(probability>=.5)
    if inner_exclusion is not None:
        excluded |= inner_exclusion.bool()
    if protect_inner_footprints and (excluded & valid).any():
        # A UV cell projects onto many image pixels. Masking only the classified
        # accessory pixels lets unclassified pixels of that same accessory
        # repaint its underlying hair. Back-project the exclusion with FINAL
        # alpha and preserve the prior colour of any affected inner cell.
        # Newly exposed cells without conflicting evidence remain trainable.
        with torch.enable_grad():
            probe_rgb=torch.zeros_like(original[:,:3],requires_grad=True)
            probe=torch.cat([probe_rgb,original[:,3:4]],1)
            blocked_mass=(render(probe)*(excluded&valid)[:,None]).sum()
            footprint=torch.autograd.grad(blocked_mass,probe_rgb)[0].abs().sum(1,keepdim=True)
        trainable_inner = trainable_inner & (footprint <= 1e-6)
    # At fixed alpha the renderer is linear in RGB. Unit-colour probes measure
    # actual contributions, including its secondary surfaces and bilinear edges.
    with torch.no_grad():
        zero=torch.cat([torch.zeros_like(original[:,:3]),original[:,3:4]],1)
        base=render(zero)
        weights=[]
        for surface in (inner,outer):
            probe=torch.cat([surface.expand_as(original[:,:3]).float(),original[:,3:4]],1)
            contribution=(render(probe)-base).mean(1).clamp(0,1)
            weights.append((valid&(contribution>=.98))[:,None].float())
        weights[0]=weights[0]*(~excluded[:,None])
        if headwear_layer is not None:
            for layer in (0,1):
                weights[layer] *= ((headwear_layer < 0) | (headwear_layer == layer))[:,None]
        if headwear_uv is not None and headwear_family is not None:
            # Only compare like materials. The renderer must see the SAME
            # semantic component as the source pixel before its RGB may fit it.
            compatible = torch.zeros_like(valid)
            for family in range(5):
                surface = (headwear_uv[:,None] == family) & head
                probe = torch.cat([surface.expand_as(original[:,:3]).float(), original[:,3:4]],1)
                contribution = (render(probe)-base).mean(1)
                compatible |= (headwear_family == family) & (contribution >= .98)
            for layer in (0,1):
                weights[layer] *= compatible[:,None]
        denominators=[(w.sum()*3).clamp_min(1) for w in weights]
    if not any(w.any() for w in weights):return original
    def losses(skin):
        error=(render(skin)-target).square()
        return [(error*w).sum()/den for w,den in zip(weights,denominators)]
    with torch.enable_grad():
        rgb=original[:,:3].clone().requires_grad_(True)
        optimizer=torch.optim.Adam([rgb],lr=.06)
        with torch.no_grad():best_error=float(sum(losses(original)));best=original.clone()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            skin=torch.cat([torch.where(head,rgb,original[:,:3]),original[:,3:4]],1)
            loss_inner,loss_outer=losses(skin)
            if not torch.isfinite(loss_inner+loss_outer):raise RuntimeError('Non-finite head material objective')
            gi=torch.autograd.grad(loss_inner,rgb,retain_graph=True)[0]
            go=torch.autograd.grad(loss_outer,rgb)[0]
            rgb.grad=torch.where(trainable_inner,gi,torch.where(outer,go,0.))
            optimizer.step()
            with torch.no_grad():
                rgb.clamp_(0,1)
                candidate=torch.cat([torch.where(head,rgb,original[:,:3]),original[:,3:4]],1)
                error=float(sum(losses(candidate)))
                if error<best_error:best_error=error;best=candidate.detach().clone()
    return best


def refine_head_material_legacy(uv, images, sources, renderer, views, steps=48):
    # A promoted outer texel can occlude formerly inner pixels. Its material must
    # explain its entire visible footprint, not only the pixels that promoted it.
    original=uv.detach().float()
    topology=build_simple_uv_topology()
    head=(topology.valid&(topology.part==0)).to(uv.device)[None,None]
    valid=torch.zeros_like(sources,dtype=torch.bool)
    y0,y1,x0,x1=head_bounds(*sources.shape[-2:])
    valid[:,y0:y1,x0:x1]=sources[:,y0:y1,x0:x1]
    weight=valid[:,None].float()
    if not weight.any():return original
    target=images[:,:3].detach().float()
    def render(skin):
        return torch.stack([renderer.forward_view(skin,v) for v in views],1).flatten(0,1)[:,:3]
    def objective(skin):
        return ((render(skin)-target).square()*weight).sum()/(weight.sum()*3)
    with torch.enable_grad():
        rgb=original[:,:3].clone().requires_grad_(True)
        optimizer=torch.optim.Adam([rgb],lr=.06)
        with torch.no_grad():best_error=float(objective(original));best=original.clone()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            skin=torch.cat([torch.where(head,rgb,original[:,:3]),original[:,3:4]],1)
            loss=objective(skin)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite head material objective')
            loss.backward();optimizer.step()
            with torch.no_grad():
                rgb.clamp_(0,1)
                candidate=torch.cat([torch.where(head,rgb,original[:,:3]),original[:,3:4]],1)
                error=float(objective(candidate))
                if error<best_error:best_error=error;best=candidate.detach().clone()
    return best
