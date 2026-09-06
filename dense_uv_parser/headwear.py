"""Semantic headwear components with one layer decision per player/component.

The crown class denotes a royal crown, distinct from the body of a fabric hat.
Colours are never used to create identities or decide their layer.
"""
import torch
from torch import nn
from SkingToolkit.dense_uv_parser.accessories import HeadAccessoryHead, connected_object_support

HEADWEAR_CLASSES = ('none', 'inner_hat_body', 'inner_hat_band',
                    'outer_hat_body', 'outer_hat_band', 'brim', 'royal_crown')


class HeadwearHead(HeadAccessoryHead):
    def __init__(self, semantic_dim=768, predict_presence=False):
        super().__init__(semantic_dim, False)
        self.classifier = nn.Conv2d(24, len(HEADWEAR_CLASSES), 1)
        dim = 7*8*8+semantic_dim
        self.presence = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,192),nn.SiLU(),nn.Dropout(.1),nn.Linear(192,2)) if predict_presence else None

    def forward(self,crop,semantic_features,return_presence=False):
        logits=super().forward(crop,semantic_features)
        if not return_presence:return logits
        spatial=torch.nn.functional.adaptive_avg_pool2d(logits.float().softmax(1),(8,8)).flatten(1)
        semantic=semantic_features.float().mean((2,3))
        return logits,self.presence(torch.cat([spatial,semantic],1))


def component_decisions(logits, foreground, views_per_group, seed_threshold=.80):
    """Pool both views BEFORE routing. A band cannot vote twice for two layers.

    A single player has one hat body, one decorative band, and one brim/crown.
    Separate players are never pooled. Background and low confidence isolated
    regions cannot establish a component. Holes stay outside its learned mask.
    """
    if logits.shape[0] % views_per_group:
        raise ValueError('Incomplete headwear view group')
    p = logits.float().softmax(1)
    family_p = torch.stack([p[:,0], p[:,1]+p[:,3], p[:,2]+p[:,4], p[:,5], p[:,6]], 1)
    confidence, family = family_p.max(1)
    supported = torch.zeros_like(foreground)
    layer = torch.full_like(family, -1)
    for component in (1,2,3,4):
        candidates = (family == component) & foreground
        grown, _ = connected_object_support(confidence, candidates.long(), foreground,
                                             seed_threshold, .5)
        for start in range(0, logits.shape[0], views_per_group):
            sl = slice(start, start+views_per_group)
            mask = grown[sl]
            if not mask.any():
                continue
            # A few confident speckles are insufficient to claim an object.
            if (confidence[sl][mask] >= seed_threshold).sum() < 16:
                continue
            selected_layer = 1
            if component in (1,2):
                inner_id, outer_id = (1,3) if component == 1 else (2,4)
                inner_evidence = p[sl,inner_id][mask].sum()
                outer_evidence = p[sl,outer_id][mask].sum()
                selected_layer = int(outer_evidence > inner_evidence)
            supported[sl] |= mask
            layer[sl][mask] = selected_layer
    return supported, layer, family, confidence


def apply_headwear_routing(routing, outputs, foreground, renderer, views):
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    logits = outputs.get('headwear_logits')
    if logits is None:
        return
    support, layer, family, confidence = component_decisions(logits, foreground, len(views))
    presence=outputs.get('headwear_presence_logits')
    if presence is not None:
        # Object classification uses both semantic shape and pretrained image
        # features. Require agreement of the two views before changing layers.
        probability=presence.sigmoid().reshape(-1,len(views),2).amin(1).repeat_interleave(len(views),0)
        hat=probability[:,0]>=outputs.get('headwear_presence_threshold',.95)
        crown=probability[:,1]>=outputs.get('headwear_presence_threshold',.95)
        support &= torch.where(family==4,crown[:,None,None],hat[:,None,None])
        layer=layer.masked_fill(~support,-1)
    semantic_support = support.clone()
    # Resolve residual old-route pixels at the output texel resolution. A thin
    # uncertain fringe cannot retain an outer cell whose footprint is mostly an
    # explicitly inner component. This uses semantic votes, never colour area.
    groups = foreground.shape[0] // len(views)
    votes = confidence.new_zeros(groups,4096)
    counts = torch.zeros_like(votes)
    for vi,view in enumerate(views):
        sl=slice(vi,foreground.shape[0],len(views))
        static=build_static_surface_routing(renderer,view,foreground.device)
        valid=foreground[sl]&static['masks'][1]&(static['part'][1]==0)
        index=static['flat_uv'][1].flatten()[None].expand(groups,-1)
        votes.scatter_add_(1,index,(valid&support[sl]&(layer[sl]==0)).float().flatten(1))
        counts.scatter_add_(1,index,valid.float().flatten(1))
    inner_cells=(votes/counts.clamp_min(1)>=.6)&(counts>=8)
    veto=torch.zeros_like(support)
    for vi,view in enumerate(views):
        sl=slice(vi,foreground.shape[0],len(views))
        static=build_static_surface_routing(renderer,view,foreground.device)
        cell=inner_cells.gather(1,static['flat_uv'][1].flatten()[None].expand(groups,-1)).reshape_as(support[sl])
        accepted=cell&foreground[sl]&static['masks'][0]&(static['part'][0]==0)&~(support[sl]&(layer[sl]==1))
        accepted &= ~(((family[sl]==3)|(family[sl]==4)) & (confidence[sl]>=.5))
        veto[sl]=accepted;support[sl]|=accepted;layer[sl][accepted]=0
    accepted_all = torch.zeros_like(support)
    for vi, view in enumerate(views):
        sl = slice(vi, foreground.shape[0], len(views))
        static = build_static_surface_routing(renderer, view, foreground.device)
        for selected in (0,1):
            accepted = support[sl] & (layer[sl] == selected) & static['masks'][selected] & (static['part'][selected] == 0)
            accepted_all[sl] |= accepted
            for name, value in [('layer',selected),('foreground',True),('surface',selected),
                                ('secondary',False),('secondary_routed',False),
                                ('semantic_fallback',False),('consensus_outer_gate_rejected',False)]:
                if name in routing:
                    routing[name][sl] = torch.where(accepted, torch.full_like(routing[name][sl],value), routing[name][sl])
            for name in ('flat_uv','part','face','texel_center_score'):
                if name in routing:
                    routing[name][sl] = torch.where(accepted, static[name][selected], routing[name][sl])
            routing['confidence'][sl] = torch.where(accepted, confidence[sl], routing['confidence'][sl])
            margin = (2*confidence[sl]-1).clamp_min(0)
            for name, value in [('confidence_margin',margin),('confidence_margin_ratio',margin/confidence[sl].clamp_min(1e-6))]:
                if name in routing:
                    routing[name][sl] = torch.where(accepted,value,routing[name][sl])
    routing['accessory_supported'] = (routing['accessory_supported'] | (accepted_all & (layer==1))) & ~(accepted_all & (layer==0))
    routing['headwear_supported'] = accepted_all
    routing['headwear_layer'] = layer
    routing['headwear_family'] = torch.where(accepted_all & semantic_support,family,0)
    routing['headwear_probability'] = confidence
    routing['headwear_inner_cell_veto'] = veto


def headwear_uv_families(routing, groups, views_per_group):
    """Assign only observed headwear texels; none stays an explicit identity."""
    valid=routing['color_foreground'] & (routing['part']==0)
    family=routing['headwear_family']
    confidence=routing['headwear_probability']
    votes=confidence.new_zeros(groups,5,4096)
    for vi in range(views_per_group):
        sl=slice(vi,valid.shape[0],views_per_group)
        index=routing['flat_uv'][sl].flatten(1)
        for cls in range(5):
            evidence=valid[sl] & (family[sl]==cls) & (confidence[sl]>=.7)
            votes[:,cls].scatter_add_(1,index,evidence.float().flatten(1))
    count,identity=votes.max(1)
    return identity.masked_fill(count<4,0).reshape(groups,64,64)


def isolate_headwear_material(uv,conditioning,details,views):
    """Unknown base texels never use a decorative band as an inpainting seed."""
    from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
    routing=details['routing']
    if 'headwear_supported' not in routing or not routing['headwear_supported'].any():return uv,None
    family=headwear_uv_families(routing,uv.shape[0],len(views))
    topology=build_simple_uv_topology();device=uv.device
    head=(topology.valid&(topology.part==0)&(topology.layer==0)).to(device).flatten()
    faces=topology.face.to(device).flatten()
    positions=topology.world_position.to(device).reshape(-1,3)
    known=(conditioning[:,4]>.5).flatten(1)
    original=uv.flatten(2);result=original.clone()
    for group in range(uv.shape[0]):
        band=(family[group].flatten()==2)&head
        if not band.any():continue
        for face in range(6):
            target=(head&(faces==face)&~known[group]).nonzero().flatten()
            source=(head&(faces==face)&known[group]&~band).nonzero().flatten()
            if not target.numel() or not source.numel():continue
            nearest=torch.cdist(positions[target],positions[source]).argmin(1)
            result[group,:3,target]=original[group,:3,source[nearest]]
    return result.reshape_as(uv),family


def reconcile_crown_top(conditioning,details,renderer,views):
    """Keep exposed top hair when views disagree about crown occupancy.

    Edited views are not always a consistent 3D observation. An isolated crown
    observation must not cover a clearly observed hair opening in another view.
    No RGB comparison, fixed crown outline, or inner material transfer is used.
    """
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    r=details['routing'];outputs=details['outputs']
    if len(views)<2 or 'headwear_presence_logits' not in outputs or 'headwear_supported' not in r or 'head_ownership_logits' not in outputs:return conditioning
    royal=r['headwear_supported']&(r['headwear_family']==4)
    if not royal.any():return conditioning
    groups=conditioning.shape[0];device=conditioning.device
    hair=outputs['head_ownership_logits'].float().softmax(1)[:,2]
    no_headwear=outputs['headwear_logits'].float().softmax(1)[:,0]
    seen=[];negative=[];positive=[]
    for vi,view in enumerate(views):
        sl=slice(vi,hair.shape[0],len(views))
        static=build_static_surface_routing(renderer,view,device)
        valid=r['observed_foreground'][sl]&static['masks'][1]&(static['part'][1]==0)&(static['face'][1]==4)
        index=static['flat_uv'][1].flatten()[None].expand(groups,-1)
        count=conditioning.new_zeros(groups,4096);hn=torch.zeros_like(count);nn=torch.zeros_like(count);rp=torch.zeros_like(count)
        count.scatter_add_(1,index,valid.float().flatten(1))
        hn.scatter_add_(1,index,(hair[sl]*valid).flatten(1));nn.scatter_add_(1,index,(no_headwear[sl]*valid).flatten(1))
        rp.scatter_add_(1,index,(royal[sl]&valid).float().flatten(1))
        observed=count>=16
        seen.append(observed);negative.append(observed&(hn/count.clamp_min(1)>.9)&(nn/count.clamp_min(1)>.9))
        positive.append(rp>=4)
    conflict=torch.stack(seen).all(0)&torch.stack(negative).any(0)&torch.stack(positive).any(0)
    conflict=conflict.reshape(groups,64,64)
    if not conflict.any():return conditioning
    result=conditioning.clone();offset=6 if conditioning.shape[1]==12 else 5
    result[:,offset:]=result[:,offset:].masked_fill(conflict[:,None],0)
    details['headwear_removed_top_uv']=conflict
    return result
