"""Use observed hair as the source for hidden hair beside an accepted accessory."""
import torch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


@torch.no_grad()
def repair_hidden_head_material(uv,conditioning,details,views):
    routing=details['routing'];rejected=routing.get('headphone_colour_rejected')
    logits=details['outputs'].get('head_color_ownership_logits',details['outputs'].get('head_ownership_logits'))
    if rejected is None or logits is None or not rejected.any():return uv
    topology=build_simple_uv_topology();device=uv.device
    head=(topology.valid&(topology.part==0)&(topology.layer==0)).to(device).flatten()
    face=topology.face.to(device).flatten();position=topology.world_position.to(device).reshape(-1,3)
    groups=uv.shape[0];count=uv.new_zeros(groups,4096);hair=uv.new_zeros(groups,4096)
    probability=logits.float().softmax(1)[:,2]
    for vi in range(len(views)):
        sl=slice(vi,routing['layer'].shape[0],len(views))
        valid=routing['color_foreground'][sl]&(routing['layer'][sl]==0)&(routing['part'][sl]==0)
        index=routing['flat_uv'][sl].flatten(1)
        count.scatter_add_(1,index,valid.float().flatten(1))
        hair.scatter_add_(1,index,(probability[sl]*valid).flatten(1))
    hair=hair/count.clamp_min(1)
    result=uv.clone().flatten(2);known=(conditioning[:,4]>.5).flatten(1)
    for batch in range(groups):
        if not rejected[batch*len(views):(batch+1)*len(views)].any():continue
        reliable=known[batch]&head&(count[batch]>=8)&(hair[batch]>=.8)
        for side in (1,2,3,4):
            observed=known[batch]&head&(face==side)&(count[batch]>=8)
            sources=reliable&(face==side)
            if int(sources.sum())<2:
                # With a largely occluded ear-side face, use the observed hair on
                # adjacent head faces, provided no visible face evidence contradicts it.
                if int(observed.sum())>0 and float(sources.sum()/observed.sum())<.6:continue
                sources=reliable
            if int(sources.sum())<2:continue
            if observed.any() and float((reliable&observed).sum()/observed.sum())<.6:continue
            targets=(~known[batch]&head&(face==side)).nonzero().flatten()
            if not len(targets):continue
            source_indices=sources.nonzero().flatten()
            nearest=torch.cdist(position[targets],position[source_indices]).argmin(1)
            # Never recursively sample a repaired texel or change an observed one.
            result[batch,:3,targets]=uv.flatten(2)[batch,:3,source_indices[nearest]]
    return result.reshape_as(uv)
