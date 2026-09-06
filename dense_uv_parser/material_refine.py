"""Fit visible head colours to both source views with geometry held fixed."""
import torch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
from SkingToolkit.dense_uv_parser.accessories import head_bounds


def refine_head_material(uv, images, sources, renderer, views, steps=48):
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
