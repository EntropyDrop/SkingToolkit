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
    def __init__(self, semantic_dim=768, predict_surface=False):
        super().__init__(semantic_dim, predict_presence=True)
        self.classifier = nn.Conv2d(24, len(CLASSES), 1)
        self.paired_context = True
        self.view_embedding = nn.Parameter(torch.zeros(2, 96))
        from SkingToolkit.dense_uv_parser.accessories import block
        self.surface_classifier=nn.Sequential(block(32,32),nn.Conv2d(32,7,1)) if predict_surface else None

    def forward(self, crop, semantic_features, return_surface=False):
        features=HeadAccessoryHead.forward(self,crop,semantic_features,return_features=True)
        logits=self.classifier(features)
        headwear = project_semantics(logits, 'headwear').softmax(1)
        spatial = F.adaptive_avg_pool2d(headwear, (8,8)).flatten(1)
        semantic = semantic_features.float().mean((2,3))
        presence = self.presence(torch.cat([spatial, semantic], 1))
        if return_surface:
            if self.surface_classifier is None:raise ValueError('Head surface decoder is unavailable')
            return logits,presence,self.surface_classifier(self.surface_features(features,crop))
        return logits, presence

    @staticmethod
    def surface_features(features,crop):
        n,_,h,w=features.shape
        y,x=torch.meshgrid(torch.linspace(-1,1,h,device=features.device),torch.linspace(-1,1,w,device=features.device),indexing='ij')
        coordinates=torch.stack([x,y])[None].expand(n,-1,-1,-1)
        view=F.one_hot(torch.arange(n,device=features.device)%2,2)[:,:,None,None].expand(-1,-1,h,w)
        return torch.cat([features,crop.to(features.dtype),coordinates.to(features.dtype),view.to(features.dtype)],1)


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
def reconcile_joint_head_geometry(conditioning,details,renderer,views):
    """Check all head faces through actual paired-view rendering.

    A clear back view may resolve an uncertain front footprint. The objective
    sums learned outer-identity probability and observed silhouette evidence,
    including secondary surfaces. No RGB, sparsity or symmetry prior is used.
    Hat/crown/phone groups are left to their established object-specific paths.
    """
    from SkingToolkit.dense_uv_parser.crown_geometry import prune_crown_top
    from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
    from SkingToolkit.dense_uv_parser.infer import simple_inpaint_uv
    outputs=details['outputs'];r=details['routing'];p=outputs['head_semantics_logits'].float().softmax(1)
    if len(views)!=2:return conditioning
    groups=conditioning.shape[0]
    other=outputs['headwear_presence_logits'].sigmoid().reshape(groups,len(views),2).amax((1,2))>=.5
    phone=outputs.get('headphone_presence_logit')
    if phone is not None:other |= phone.sigmoid().reshape(groups,len(views)).amax(1)>=.99
    evidence=((p[:,1:].amax(1)>=.95)&r['observed_foreground']).flatten(1).sum(1).reshape(groups,len(views)).amin(1)>=64
    active=evidence&~other
    if not active.any():return conditioning
    uv=torch.stack([simple_inpaint_uv(c[None].cpu())[0] for c in conditioning]).to(conditioning.device)
    outer_probability=p[:,[4,5,6,7,10,11,12,13]].sum(1)
    topology=build_simple_uv_topology()
    candidate=(topology.valid&(topology.part==0)&(topology.layer==1)).to(uv.device)
    changed=torch.zeros_like(uv[:,3],dtype=torch.bool);records=[]
    for group in active.nonzero().flatten().tolist():
        sl=slice(group*len(views),(group+1)*len(views))
        fixed,record=prune_crown_top(uv[group:group+1],outer_probability[sl],r['observed_foreground'][sl],renderer,views,removable_uv=candidate,surface_probability=(outputs['head_surface_logits'][sl].float().softmax(1) if 'head_surface_logits' in outputs else None))
        changed[group]=(uv[group,3]>.5)&(fixed[0,3]<=.5)
        records.append({'group':group,**record[0]})
    result=conditioning.clone();offset=6 if conditioning.shape[1]==12 else 5
    result[:,offset:]=result[:,offset:].masked_fill(changed[:,None],0)
    details['joint_head_removed_uv']=changed;details['joint_head_geometry']=records
    return result


def apply_beard_component_routing(routing,outputs,foreground,renderer,views):
    """Route confident facial-hair pixels locally, allowing both beard layers.

    Never pool a whole face/person into one winning layer. Each projected cell
    needs local semantic evidence; a mouth opening or the opposite beard layer
    is a contradiction, not a hole to fill from a component majority.
    """
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    p=outputs['head_semantics_logits'].float().softmax(1)
    head_foreground=foreground&(routing['part']==0)
    groups=foreground.shape[0]//len(views)
    phone=outputs.get('joint_phone_expert_accepted')
    accepted_all=torch.zeros_like(foreground);outer_all=torch.zeros_like(foreground)
    for group in range(groups):
        start=group*len(views);sl=slice(start,start+len(views))
        if phone is not None and phone[sl].any():continue
        # Separate supports may coexist within the same player and UV face.
        for layer,identity in ((0,3),(1,5)):
            votes=p.new_zeros(4096);counts=torch.zeros_like(votes)
            for vi,view in enumerate(views):
                n=start+vi;s=build_static_surface_routing(renderer,view,p.device)
                valid=head_foreground[n]&s['masks'][layer]&(s['part'][layer]==0)
                flat=s['flat_uv'][layer].flatten()
                votes.scatter_add_(0,flat,(p[n,identity]*valid).flatten())
                counts.scatter_add_(0,flat,valid.float().flatten())
            cells=(counts>=8)&(votes/counts.clamp_min(1)>=.9)
            for vi,view in enumerate(views):
                n=start+vi;s=build_static_surface_routing(renderer,view,p.device)
                cell=cells[s['flat_uv'][layer]]
                accepted=((p[n,identity]>=.9)|(cell&(p[n,identity]>=.5)))
                accepted &= head_foreground[n]&s['masks'][layer]&(s['part'][layer]==0)
                accepted_all[n] |= accepted
                if layer:outer_all[n] |= accepted
                for name,value in [('layer',layer),('foreground',True),('surface',layer),('secondary',False),('secondary_routed',False),('semantic_fallback',False),('consensus_outer_gate_rejected',False)]:
                    if name in routing:routing[name][n]=torch.where(accepted,torch.full_like(routing[name][n],value),routing[name][n])
                for name in ('flat_uv','part','face','texel_center_score'):
                    if name in routing:routing[name][n]=torch.where(accepted,s[name][layer],routing[name][n])
                confidence=p[n,identity];margin=(2*confidence-1).clamp_min(0)
                for name,value in [('confidence',confidence),('confidence_margin',margin),('confidence_margin_ratio',margin/confidence.clamp_min(1e-6))]:
                    if name in routing:routing[name][n]=torch.where(accepted,value,routing[name][n])
    inner=accepted_all&~outer_all
    routing['accessory_supported']=(routing['accessory_supported']|outer_all)&~inner
    routing['ownership_inner_supported'] |= inner
    routing['beard_component_supported']=accepted_all
    routing['beard_inner_supported']=inner
    routing['beard_outer_supported']=outer_all
    routing['headwear_supported'] &= ~accepted_all
    routing['headwear_layer']=routing['headwear_layer'].masked_fill(accepted_all,-1)
    routing['headwear_family']=routing['headwear_family'].masked_fill(accepted_all,0)


def apply_head_surface_routing(routing,outputs,foreground,renderer,views):
    """Write hair/beard evidence to its learned cube face, including rear hits.

    A front beard seen beside the head from behind belongs to the front plane,
    not the first side plane intersected by the camera ray. Keep existing body
    ownership and abstain when no matching physical surface is available.
    """
    if 'head_surface_logits' not in outputs:return
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    sp=outputs['head_surface_logits'].float().softmax(1);confidence,face=sp.max(1)
    jp=outputs['head_semantics_logits'].float().softmax(1);identity=jp.argmax(1)
    family=torch.isin(identity,identity.new_tensor([2,3,4,5]))
    outer=(identity==4)|(identity==5)
    valid=foreground&(routing['part']==0)&family&(jp.amax(1)>=.8)&(confidence>=.8)&(face>0)
    phone=outputs.get('joint_phone_expert_accepted')
    if phone is not None:valid &= ~phone[:,None,None]
    groups=foreground.shape[0]//len(views)
    headwear=outputs.get('headwear_presence_logits')
    if headwear is not None:
        protected=headwear.sigmoid().reshape(groups,len(views),2).amax((1,2))>=.5
        valid &= ~protected.repeat_interleave(len(views))[:,None,None]
    veto=sp.new_zeros(groups,4096);total=torch.zeros_like(veto)
    accepted_all=torch.zeros_like(foreground)
    for vi,view in enumerate(views):
        s=build_static_surface_routing(renderer,view,foreground.device)
        for group in range(groups):
            n=group*len(views)+vi
            match=s['masks']&(s['part']==0)&(s['layer']==outer[n].long()[None])&(s['face']==face[n][None]-1)
            exists=match.any(0);selected=match.long().argmax(0)
            accepted=valid[n]&exists;accepted_all[n]=accepted
            choose=lambda value:value.gather(0,selected[None])[0]
            for name in ('flat_uv','part','face','texel_center_score','layer'):
                if name in routing:routing[name][n]=torch.where(accepted,choose(s[name]),routing[name][n])
            for name,value in [('surface',selected),('secondary',selected>=2),('secondary_routed',selected>=2),('semantic_fallback',torch.zeros_like(accepted)),('consensus_outer_gate_rejected',torch.zeros_like(accepted))]:
                if name in routing:routing[name][n]=torch.where(accepted,value,routing[name][n])
            routing['confidence'][n]=torch.where(accepted,torch.minimum(confidence[n],jp[n].amax(0)),routing['confidence'][n])
            # Exclude a wrong primary UV footprint only with consistent learned
            # face evidence. Other valid primary material counts against veto.
            primary=foreground[n]&s['masks'][1]&(s['part'][1]==0)
            wrong=accepted&primary&outer[n]&(s['face'][1]!=face[n]-1)
            index=s['flat_uv'][1].flatten()
            total[group].scatter_add_(0,index,primary.float().flatten())
            veto[group].scatter_add_(0,index,wrong.float().flatten())
    routing['accessory_supported']=(routing['accessory_supported']|(accepted_all&outer))&~(accepted_all&~outer)
    routing['ownership_inner_supported'] |= accepted_all&~outer
    routing['head_surface_supported']=accepted_all
    routing['head_surface_uv_veto']=((veto>=8)&(veto/total.clamp_min(1)>=.8)).reshape(groups,64,64)


@torch.no_grad()
def reconcile_beard_alignment(conditioning,details,views):
    """Align a mixed beard in face-local UV coordinates using observed identity.

    Visible skin/hair on the corresponding base texel contradicts beard there.
    Hidden base texels are not negative evidence; they may inherit the material
    of a confirmed overlapping beard after final geometry/material fitting.
    """
    r=details['routing'];p=details['outputs']['head_semantics_logits'].float().softmax(1)
    groups=conditioning.shape[0];count=p.new_zeros(groups,4096);beard=torch.zeros_like(count);other=torch.zeros_like(count)
    for vi in range(len(views)):
        sl=slice(vi,p.shape[0],len(views));valid=r['color_foreground'][sl]&(r['part'][sl]==0)
        index=r['flat_uv'][sl].flatten(1)
        count.scatter_add_(1,index,valid.float().flatten(1))
        beard.scatter_add_(1,index,((p[sl,3]+p[sl,5])*valid).flatten(1))
        other.scatter_add_(1,index,((p[sl,1]+p[sl,2])*valid).flatten(1))
    b=(beard/count.clamp_min(1)).reshape(groups,64,64);o=(other/count.clamp_min(1)).reshape_as(b);c=count.reshape_as(b)
    inner=(b[:,:16,:32]>=.8)&(c[:,:16,:32]>=8)
    outer=(b[:,:16,32:]>=.5)&(c[:,:16,32:]>=8)
    mixed=inner.flatten(1).any(1)&outer.flatten(1).any(1)
    phone=details['outputs'].get('joint_phone_expert_accepted')
    if phone is not None:mixed &= ~phone.reshape(groups,len(views)).any(1)
    headwear=details['outputs'].get('headwear_presence_logits')
    if headwear is not None:mixed &= headwear.sigmoid().reshape(groups,len(views),2).amax((1,2))<.5
    mismatch=outer&(c[:,:16,:32]>=8)&(o[:,:16,:32]>.5)&(o[:,:16,:32]>b[:,:16,:32]+.1)&mixed[:,None,None]
    changed=torch.zeros_like(b,dtype=torch.bool);changed[:,:16,32:]=mismatch
    result=conditioning.clone();offset=6 if result.shape[1]==12 else 5
    result[:,offset:]=result[:,offset:].masked_fill(changed[:,None],0)
    hidden=(conditioning[:,4,:16,:32]<=.5)&outer&~mismatch&mixed[:,None,None]
    fill=torch.zeros_like(changed);fill[:,:16,:32]=hidden
    details['beard_alignment_removed_uv']=changed;details['beard_alignment_hidden_inner_uv']=fill
    return result


@torch.no_grad()
def complete_aligned_beard_material(uv,details):
    mask=details.get('beard_alignment_hidden_inner_uv')
    if mask is None or not mask.any():return uv
    result=uv.clone();accepted=mask[:,:16,:32]&(uv[:,3,:16,32:]>.5)
    result[:,:3,:16,:32]=torch.where(accepted[:,None],uv[:,:3,:16,32:],uv[:,:3,:16,:32])
    return result
