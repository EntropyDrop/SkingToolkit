"""Independent semantic evidence for head layers; existing v101 heads stay frozen."""
import torch
from torch import nn
from SkingToolkit.dense_uv_parser.accessories import HeadAccessoryHead, connected_object_support

OWNERSHIP_CLASSES=('abstain','inner_face','inner_hair','eyewear','headphones')


class HeadOwnershipHead(HeadAccessoryHead):
    def __init__(self,semantic_dim=768,predict_presence=False):
        super().__init__(semantic_dim,False)
        self.classifier=nn.Conv2d(24,len(OWNERSHIP_CLASSES),1)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.presence = nn.Sequential(nn.Linear(5*8*8,128),nn.SiLU(),nn.Dropout(.1),nn.Linear(128,1)) if predict_presence else None

    def predict_presence(self,logits):
        if self.presence is None:raise ValueError('No trained headphone presence gate')
        evidence=torch.nn.functional.adaptive_avg_pool2d(logits.float().softmax(1),(8,8)).flatten(1)
        return self.presence(evidence).flatten()

    def forward(self,crop,semantic_features,return_presence=False):
        logits=super().forward(crop,semantic_features)
        return (logits,self.predict_presence(logits)) if return_presence else logits


def apply_head_ownership(routing,outputs,foreground,renderer,views):
    from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing
    logits=outputs.get('head_ownership_logits')
    if logits is None:return
    p=logits.float().softmax(1);confidence,identity=p.max(1)
    old=outputs['accessory_logits'].float().softmax(1)
    # The new head abstains on hats and outer hair. Preserve those established
    # identities; a confident face label can correct geometric false positives.
    protected=(old[:,2]+old[:,3])>.5
    outer=torch.zeros_like(foreground);inner=torch.zeros_like(foreground)
    # Resolve semantic conflicts at the output texture's resolution. A few
    # boundary pixels cannot keep an outer texel whose visible footprint is
    # confidently a face. These are learned class votes, independent of RGB.
    groups=foreground.shape[0]//len(views)
    face_votes=p.new_zeros(groups,4096);counts=p.new_zeros(groups,4096)
    for vi,view in enumerate(views):
        static=build_static_surface_routing(renderer,view,foreground.device)
        valid=foreground[vi::len(views)]&static['masks'][1]&(static['part'][1]==0)&~protected[vi::len(views)]
        index=static['flat_uv'][1].flatten()[None].expand(groups,-1)
        face_votes.scatter_add_(1,index,(p[vi::len(views),1]*valid).flatten(1))
        counts.scatter_add_(1,index,valid.float().flatten(1))
    face_cells=(face_votes/counts.clamp_min(1)>=.7)&(counts>=4)
    veto=torch.zeros_like(foreground)
    for vi,view in enumerate(views):
        sl=slice(vi,foreground.shape[0],len(views));static=build_static_surface_routing(renderer,view,foreground.device)
        valid=foreground[sl]&static['masks'][1]&(static['part'][1]==0)
        support,_=connected_object_support(confidence[sl],(identity[sl]==4).long(),valid,.95,.5)
        presence=outputs.get('headphone_presence_logit')
        if presence is not None:
            probabilities=presence.sigmoid()
            if outputs.get('headphone_presence_consensus',False):
                probabilities=probabilities.reshape(-1,len(views)).amin(1).repeat_interleave(len(views))
            support&=(probabilities[sl]>=outputs.get('headphone_presence_threshold',.9))[:,None,None]
        mode=outputs.get('headphone_routing_mode','unrestricted')
        if mode=='disabled':support.zero_()
        elif mode=='existing_uv':
            occupancy=outputs.get('headphone_uv_support')
            if occupancy is None:raise ValueError('Headphone colour routing requires accepted UV support')
            accepted_uv=occupancy.gather(1,static['flat_uv'][1].flatten()[None].expand(groups,-1)).reshape_as(valid)
            support&=accepted_uv
        outer[sl]=support&~protected[sl]
        cell=face_cells.gather(1,static['flat_uv'][1].flatten()[None].expand(groups,-1)).reshape_as(valid)&valid
        veto[sl]=cell
        outer[sl]&=~cell
        inner[sl]=(cell|((identity[sl]==1)&(confidence[sl]>=.95)&(old[sl,1]<.5)))&~protected[sl]&foreground[sl]&static['masks'][0]&(static['part'][0]==0)
        for layer,accepted in ((1,outer[sl]),(0,inner[sl])):
            for name,value in [('layer',layer),('foreground',True),('surface',layer),('secondary',False),('secondary_routed',False),('semantic_fallback',False),('consensus_outer_gate_rejected',False)]:
                if name in routing:routing[name][sl]=torch.where(accepted,torch.full_like(routing[name][sl],value),routing[name][sl])
            for name in ('flat_uv','part','face','texel_center_score'):
                if name in routing:routing[name][sl]=torch.where(accepted,static[name][layer],routing[name][sl])
            routing['confidence'][sl]=torch.where(accepted,confidence[sl],routing['confidence'][sl])
            cm=(2*confidence[sl]-1).clamp_min(0)
            for name,value in [('confidence_margin',cm),('confidence_margin_ratio',cm/confidence[sl].clamp_min(1e-6))]:
                if name in routing:routing[name][sl]=torch.where(accepted,value,routing[name][sl])
    routing['accessory_supported']=(routing['accessory_supported']|outer)&~inner
    presence=outputs.get('headphone_presence_logit')
    present=torch.ones(p.shape[0],device=p.device,dtype=torch.bool)
    if presence is not None:
        probability=presence.sigmoid()
        if outputs.get('headphone_presence_consensus',False):
            probability=probability.reshape(-1,len(views)).amin(1).repeat_interleave(len(views))
        present=probability>=outputs.get('headphone_presence_threshold',.9)
    routing['headphone_colour_conflict']=(p[:,4]>=.5)&present[:,None,None]&foreground
    routing['ownership_face_cell_veto']=veto
    routing['ownership_outer_supported']=outer
    routing['ownership_inner_supported']=inner
    routing['head_ownership_identity']=identity
    routing['head_ownership_probability']=confidence
