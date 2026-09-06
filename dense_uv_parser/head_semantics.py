"""Mutually exclusive head identities with paired-view semantic context.

An inner nose cannot simultaneously be a royal crown. A learned facial-hair
identity explicitly distinguishes flat and extruded beard geometry.
"""
import torch
from torch import nn
import torch.nn.functional as F
from SkingToolkit.dense_uv_parser.headwear import HeadwearHead
from SkingToolkit.dense_uv_parser.accessories import HeadAccessoryHead

CLASSES = ('abstain', 'inner_face', 'inner_hair', 'inner_beard', 'outer_hair',
           'outer_beard', 'eyewear', 'headphones', 'inner_hat_body',
           'inner_hat_band', 'outer_hat_body', 'outer_hat_band', 'brim', 'royal_crown')
PROJECTIONS = {
    'headwear': (tuple(range(8)), (8,), (9,), (10,), (11,), (12,), (13,)),
    'ownership': ((0,4,5,8,9,10,11,12,13), (1,3), (2,), (6,), (7,)),
}


def project_semantics(logits, task):
    """Marginalize ONE categorical distribution, preserving its probability."""
    return torch.stack([torch.logsumexp(logits[:, list(ids)].float(), 1)
                        for ids in PROJECTIONS[task]], 1)


class HeadSemanticsHead(HeadwearHead):
    def __init__(self, semantic_dim=768):
        super().__init__(semantic_dim, predict_presence=True)
        self.classifier = nn.Conv2d(24, len(CLASSES), 1)
        self.paired_context = True
        self.view_embedding = nn.Parameter(torch.zeros(2, 96))

    def forward(self, crop, semantic_features):
        logits = HeadAccessoryHead.forward(self, crop, semantic_features)
        headwear = project_semantics(logits, 'headwear').softmax(1)
        spatial = F.adaptive_avg_pool2d(headwear, (8,8)).flatten(1)
        semantic = semantic_features.float().mean((2,3))
        presence = self.presence(torch.cat([spatial, semantic], 1))
        return logits, presence


def apply_joint_head_routing(routing, outputs, foreground, renderer, views):
    """Consume the new model's inner-hair evidence across both visible top views.

    Face/beard evidence already reaches the established ownership router via
    categorical marginalization. This additional top veto requires confident
    inner hair in EVERY observed view, not a single-view colour/occupancy rule.
    Genuine outer hair and uncertain/occluded cells stay under existing routing.
    """
    if len(views) != 2:
        raise ValueError('Joint head routing requires front/back views')
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    p = outputs['head_semantics_logits'].float().softmax(1)
    groups = foreground.shape[0] // len(views)
    negative = []
    for vi, view in enumerate(views):
        static = build_static_surface_routing(renderer, view, foreground.device)
        sl = slice(vi, foreground.shape[0], len(views))
        valid = foreground[sl] & static['masks'][1] & (static['part'][1]==0) & (static['face'][1]==4)
        index = static['flat_uv'][1].flatten()[None].expand(groups,-1)
        votes = p.new_zeros(groups,4096); counts = torch.zeros_like(votes)
        votes.scatter_add_(1,index,(p[sl,2]*valid).flatten(1))
        counts.scatter_add_(1,index,valid.float().flatten(1))
        negative.append((votes/counts.clamp_min(1) >= .95) & (counts>=8))
    cells = torch.stack(negative).all(0)
    phone_expert=outputs.get('joint_phone_expert_accepted')
    if phone_expert is not None:
        cells &= ~phone_expert.reshape(groups,len(views)).any(1)[:,None]
    accepted_all = torch.zeros_like(foreground)
    for vi, view in enumerate(views):
        static = build_static_surface_routing(renderer, view, foreground.device)
        sl = slice(vi, foreground.shape[0], len(views))
        accepted = cells.gather(1,static['flat_uv'][1].flatten()[None].expand(groups,-1)).reshape_as(foreground[sl])
        accepted &= foreground[sl] & static['masks'][0] & (static['part'][0]==0)
        accepted_all[sl] = accepted
        for name,value in [('layer',0),('foreground',True),('surface',0),('secondary',False),('secondary_routed',False),('semantic_fallback',False),('consensus_outer_gate_rejected',False)]:
            if name in routing:routing[name][sl]=torch.where(accepted,torch.full_like(routing[name][sl],value),routing[name][sl])
        for name in ('flat_uv','part','face','texel_center_score'):
            if name in routing:routing[name][sl]=torch.where(accepted,static[name][0],routing[name][sl])
        confidence=p[sl,2];margin=(2*confidence-1).clamp_min(0)
        for name,value in [('confidence',confidence),('confidence_margin',margin),('confidence_margin_ratio',margin/confidence.clamp_min(1e-6))]:
            if name in routing:routing[name][sl]=torch.where(accepted,value,routing[name][sl])
    routing['accessory_supported'] &= ~accepted_all
    routing['ownership_inner_supported'] |= accepted_all
    routing['joint_inner_hair_veto'] = accepted_all
    # Clear stale component attribution after a semantic layer correction.
    routing['headwear_supported'] &= ~accepted_all
    routing['headwear_layer'] = routing['headwear_layer'].masked_fill(accepted_all,-1)
    routing['headwear_family'] = routing['headwear_family'].masked_fill(accepted_all,0)


@torch.no_grad()
def reconcile_joint_hair_top(conditioning,details,renderer,views):
    """Compare ambiguous hair-top occupancy through actual paired-view rendering.

    A clear back view may resolve an uncertain front footprint. The objective
    sums learned outer-identity probability and observed silhouette evidence,
    including secondary surfaces. No RGB, sparsity or symmetry prior is used.
    Hat/crown/phone groups are left to their established object-specific paths.
    """
    from SkingToolkit.dense_uv_parser.crown_geometry import prune_crown_top
    from SkingToolkit.dense_uv_parser.infer import simple_inpaint_uv
    outputs=details['outputs'];r=details['routing'];p=outputs['head_semantics_logits'].float().softmax(1)
    if len(views)!=2:return conditioning
    groups=conditioning.shape[0]
    other=outputs['headwear_presence_logits'].sigmoid().reshape(groups,len(views),2).amax((1,2))>=.5
    phone=outputs.get('headphone_presence_logit')
    if phone is not None:other |= phone.sigmoid().reshape(groups,len(views)).amax(1)>=.99
    evidence=((p[:,2]>=.95)&r['observed_foreground']).flatten(1).sum(1).reshape(groups,len(views)).amin(1)>=64
    active=evidence&~other
    if not active.any():return conditioning
    uv=torch.stack([simple_inpaint_uv(c[None].cpu())[0] for c in conditioning]).to(conditioning.device)
    outer_probability=p[:,[4,5,6,7,10,11,12,13]].sum(1)
    changed=torch.zeros_like(uv[:,3],dtype=torch.bool);records=[]
    for group in active.nonzero().flatten().tolist():
        sl=slice(group*len(views),(group+1)*len(views))
        fixed,record=prune_crown_top(uv[group:group+1],outer_probability[sl],r['observed_foreground'][sl],renderer,views)
        changed[group]=(uv[group,3]>.5)&(fixed[0,3]<=.5)
        records.append({'group':group,**record[0]})
    result=conditioning.clone();offset=6 if conditioning.shape[1]==12 else 5
    result[:,offset:]=result[:,offset:].masked_fill(changed[:,None],0)
    details['joint_hair_removed_top_uv']=changed;details['joint_hair_geometry']=records
    return result


def apply_beard_component_routing(routing,outputs,foreground,renderer,views):
    """One learned facial-hair component selects one layer across both views.

    Pool semantic beard evidence, not RGB or bilateral symmetry. Uncertain
    texel fringes follow confident beard votes only when no other confident
    surface identity contradicts the component's layer.
    """
    from SkingToolkit.dense_uv_parser.accessories import connected_object_support
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    p=outputs['head_semantics_logits'].float().softmax(1);beard=p[:,3]+p[:,5]
    identity=p.argmax(1);candidate=(identity==3)|(identity==5)
    head_foreground=foreground&(routing['part']==0)
    support,_=connected_object_support(beard,candidate.long(),head_foreground,.9,.5)
    groups=foreground.shape[0]//len(views)
    phone=outputs.get('joint_phone_expert_accepted')
    accepted_all=torch.zeros_like(foreground);outer_all=torch.zeros_like(foreground)
    selected_layers=[];inner_cells_all=torch.zeros(groups,4096,device=p.device,dtype=torch.bool)
    for group in range(groups):
        start=group*len(views);sl=slice(start,start+len(views));mask=support[sl]
        if phone is not None and phone[sl].any():selected_layers.append(-1);continue
        if ((beard[sl]>=.9)&mask).sum()<16:selected_layers.append(-1);continue
        selected=int(p[sl,5][mask].sum()>p[sl,3][mask].sum());selected_layers.append(selected)
        votes=p.new_zeros(4096);conflict=torch.zeros_like(votes)
        # Confident inner face/hair permits inner beard, but prevents fabricating
        # an outer beard plane across a real mouth opening or exposed skin.
        rivals=[4,6,7,10,11,12,13] if selected==0 else [1,2,4,6,7,8,9,10,11,12,13]
        for vi,view in enumerate(views):
            index=start+vi;s=build_static_surface_routing(renderer,view,p.device)
            valid=head_foreground[index]&s['masks'][1]&(s['part'][1]==0)
            flat=s['flat_uv'][1].flatten()
            votes.scatter_add_(0,flat,((beard[index]>=.9)&mask[vi]&valid).float().flatten())
            conflict.scatter_add_(0,flat,((p[index,rivals].amax(0)>=.8)&valid).float().flatten())
        cells=(votes>=8)&(votes/(votes+conflict).clamp_min(1)>=.9)
        if selected==0:inner_cells_all[group]=cells
        for vi,view in enumerate(views):
            index=start+vi;s=build_static_surface_routing(renderer,view,p.device)
            cell=cells[s['flat_uv'][1]]
            accepted=(mask[vi]|cell)&head_foreground[index]&s['masks'][selected]&(s['part'][selected]==0)
            # Strong contradictory semantic pixels themselves never change.
            accepted &= ~(p[index,rivals].amax(0)>=.8)
            accepted_all[index]=accepted
            if selected:outer_all[index]=accepted
            for name,value in [('layer',selected),('foreground',True),('surface',selected),('secondary',False),('secondary_routed',False),('semantic_fallback',False),('consensus_outer_gate_rejected',False)]:
                if name in routing:routing[name][index]=torch.where(accepted,torch.full_like(routing[name][index],value),routing[name][index])
            for name in ('flat_uv','part','face','texel_center_score'):
                if name in routing:routing[name][index]=torch.where(accepted,s[name][selected],routing[name][index])
            confidence=torch.where(cell,beard[index].clamp_min(.9),beard[index]);margin=(2*confidence-1).clamp_min(0)
            for name,value in [('confidence',confidence),('confidence_margin',margin),('confidence_margin_ratio',margin/confidence.clamp_min(1e-6))]:
                if name in routing:routing[name][index]=torch.where(accepted,value,routing[name][index])
    inner=accepted_all&~outer_all
    routing['accessory_supported']=(routing['accessory_supported']|outer_all)&~inner
    routing['ownership_inner_supported'] |= inner
    routing['beard_component_supported']=accepted_all
    routing['beard_component_layers']=torch.tensor(selected_layers,device=p.device)
    routing['beard_inner_uv_veto']=inner_cells_all.reshape(groups,64,64)
    routing['headwear_supported'] &= ~accepted_all
    routing['headwear_layer']=routing['headwear_layer'].masked_fill(accepted_all,-1)
    routing['headwear_family']=routing['headwear_family'].masked_fill(accepted_all,0)
