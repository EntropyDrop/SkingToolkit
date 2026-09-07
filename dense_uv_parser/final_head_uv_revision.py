"""Local image evidence, explicit edit decisions and learned material relations."""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from SkingToolkit.dense_uv_parser.accessories import head_bounds


def head_projection(ids, mappings_dir):
    """Project candidate UV cells, never target UV labels, onto the two cameras.

    Both possible layers get a footprint, including cells currently absent.
    Visibility and semantic contradictions remain input features for the model.
    """
    node=torch.full((4096,),-1,dtype=torch.long);node[ids.cpu()]=torch.arange(len(ids))
    matrices=[]
    for view in ('front_left','back_left'):
        data=torch.load(Path(mappings_dir)/f'{view}_mapping.pt',map_location='cpu',weights_only=False)
        h,w=data['inner_mask'].shape;y0,y1,x0,x1=head_bounds(h,w)
        yy,xx=torch.meshgrid(torch.arange(y0,y1),torch.arange(x0,x1),indexing='ij')
        cell_y=((yy-y0+.5)*56/(y1-y0)).long().clamp(0,55)
        cell_x=((xx-x0+.5)*56/(x1-x0)).long().clamp(0,55)
        bins=(cell_y*56+cell_x).flatten();matrix=torch.zeros(len(ids)*56*56)
        for layer in ('inner','outer'):
            xy=data[f'{layer}_uv_map'][y0:y1,x0:x1].round().long().clamp(0,63)
            active=data[f'{layer}_mask'][y0:y1,x0:x1].bool().flatten()
            nodes=node[(xy[:,:,1]*64+xy[:,:,0]).flatten()];active &= nodes>=0
            indices=nodes[active]*3136+bins[active]
            matrix.scatter_add_(0,indices,torch.ones(len(indices)))
        matrix=matrix.reshape(len(ids),3136)
        matrix=matrix/matrix.sum(1,keepdim=True).clamp_min(1)
        matrices.append(matrix)
    return torch.stack(matrices)


def init_revision(model,mappings_dir):
    if not mappings_dir:raise ValueError('Revision 2 requires renderer mappings')
    width=model.query.out_features
    model.register_buffer('projection',head_projection(model.ids,mappings_dir))
    model.local_query=nn.Linear(102 if model.robust_edits else 52,width)
    model.edit=nn.Linear(width,1)
    nn.init.zeros_(model.edit.weight);nn.init.constant_(model.edit.bias,-4.)
    model.mirror_link=nn.Linear(width,1);model.layer_link=nn.Linear(width,1)
    for module in (model.mirror_link,model.layer_link):
        nn.init.zeros_(module.weight);nn.init.constant_(module.bias,-4.)
    seen=set();orbits=[]
    for i in range(len(model.ids)):
        if i in seen:continue
        orbit=[i,int(model.mirror[i]),int(model.other[i]),int(model.mirror[model.other[i]])]
        assert len(set(orbit))==4
        seen.update(orbit);orbits.append(orbit)
    orbits=torch.tensor(orbits,dtype=torch.long)
    orbit_id=torch.empty(len(model.ids),dtype=torch.long);slot=torch.empty_like(orbit_id)
    for j,orbit in enumerate(orbits):orbit_id[orbit]=j;slot[orbit]=torch.arange(4)
    for name,value in [('material_orbits',orbits),('material_orbit_id',orbit_id),('material_slot',slot)]:model.register_buffer(name,value,persistent=False)


def local_queries(model,evidence):
    b=evidence.shape[0]//2
    # Matrix rows are normalized candidate footprints, so missing views stay zero.
    local=torch.einsum('vkp,bvcp->bkvc',model.projection,evidence.reshape(b,2,25,3136)).flatten(2)
    if model.robust_edits:
        neighborhood=F.avg_pool2d(evidence,5,stride=1,padding=2)
        context=torch.einsum('vkp,bvcp->bkvc',model.projection,neighborhood.reshape(b,2,25,3136)).flatten(2)
        local=torch.cat([local,context],2)
    visible=(model.projection.sum(2)>0).T[None].expand(b,-1,-1).to(local.dtype)
    return model.local_query(torch.cat([local,visible],2))


def warp_head_evidence(evidence,theta):
    """Move RGB, foreground and semantic evidence together within each camera."""
    offset=torch.zeros(1,25,1,1,device=evidence.device,dtype=evidence.dtype);offset[:,:3]=.5
    grid=F.affine_grid(theta,evidence.shape,align_corners=False)
    return F.grid_sample(evidence-offset,grid,align_corners=False,padding_mode='zeros')+offset


def augment_local_evidence(evidence):
    n=len(evidence)
    theta=torch.zeros(n,2,3,device=evidence.device)
    scale=torch.empty(n,2,device=evidence.device).uniform_(.96,1.04)
    theta[:,0,0]=scale[:,0];theta[:,1,1]=scale[:,1]
    theta[:,:,2]=torch.empty(n,2,device=evidence.device).uniform_(-.07,.07)
    evidence=warp_head_evidence(evidence,theta)
    # Sometimes semantics are uninformative; keep RGB/foreground as alternate evidence.
    keep=(torch.rand(n,1,1,1,device=evidence.device)>.15).to(evidence.dtype)
    evidence[:,4:]*=keep
    return evidence


def initialize_revision(model,checkpoint):
    state=dict(checkpoint['final_head_uv_state'])
    if model.robust_edits and state['local_query.weight'].shape[1]==52:
        old=state['local_query.weight'];expanded=torch.zeros_like(model.local_query.weight,device='cpu')
        expanded[:,:50]=old[:,:50];expanded[:,-2:]=old[:,-2:]
        state['local_query.weight']=expanded
    if model.topology_context:
        for k,v in model.state_dict().items():
            if k.startswith("topology_adapter.") and k not in state:state[k]=v.detach().cpu()
    model.load_state_dict(state,strict=True)


def tie_material(model,rgb,alpha,mirror_logits,layer_logits):
    """Share RGB only in connected groups selected by learned pair relations.

    Graph closure uses canonical four-cell orbits. Every member sums colours in
    the same order, giving exact pair equality, including mixed inner/outer groups.
    """
    ids=model.material_orbits;b=rgb.shape[0];count=len(ids)
    mirror=(mirror_logits[:,ids].sigmoid()>=.5)
    layer=(layer_logits[:,ids].sigmoid()>=.5)
    active=(alpha[:,ids]>.5)
    graph=torch.eye(4,device=rgb.device,dtype=torch.bool)[None,None].expand(b,count,-1,-1).clone()
    for left,right,accepted in ((0,1,mirror[:,:,0]),(2,3,mirror[:,:,2]),(0,2,layer[:,:,0]),(1,3,layer[:,:,1])):
        connection=accepted&active[:,:,left]&active[:,:,right]
        graph[:,:,left,right]=connection;graph[:,:,right,left]=connection
    for _ in range(2):graph=graph|(graph[:,:,:,:,None]&graph[:,:,None,:,:]).any(3)
    weight=graph.to(rgb.dtype);grouped=torch.matmul(weight,rgb[:,ids])/weight.sum(3,keepdim=True)
    return grouped[:,model.material_orbit_id,model.material_slot]


def decode_revision(model,features,uv):
    raw_edit_logits=model.edit(features).squeeze(2)
    semantic_logits=model.semantic(features)
    alpha_logits=torch.where(uv[:,:,3]>.5,-raw_edit_logits,raw_edit_logits)
    if model.semantic_geometry:
        # A single categorical distribution decides absence and outer material.
        # UV class zero denotes absent outer support (source-image class zero
        # still means abstention). The edit prior shifts all occupied classes
        # together; occupancy and semantic supervision both train this score.
        occupied=(4,5,6,7,10,11,12,13);inner=(1,2,3,8,9)
        joint=semantic_logits.clone()
        joint[:,:,0]=torch.logsumexp(semantic_logits[:,:,(0,*inner)],2)
        joint[:,:,inner]=-1e4
        joint[:,:,occupied]=semantic_logits[:,:,occupied]+alpha_logits[:,:,None]
        semantic_logits=torch.where(model.outer[None,:,None],joint,semantic_logits)
        joint_alpha=torch.logsumexp(semantic_logits[:,:,1:],2)-semantic_logits[:,:,0]
        alpha_logits=torch.where(model.outer[None],joint_alpha,alpha_logits)
    edit_logits=torch.where(uv[:,:,3]>.5,-alpha_logits,alpha_logits)
    edit_p=edit_logits.sigmoid();change=(edit_p>=model.edit_threshold).float()+(edit_p-edit_p.detach())
    alpha=torch.where(model.outer[None],uv[:,:,3]+(1-2*uv[:,:,3])*change,uv[:,:,3])
    gate_logits=model.color_gate(features).squeeze(2);gate_p=gate_logits.sigmoid()
    gate=(gate_p>=.5).float()+(gate_p-gate_p.detach())
    delta=model.color_delta(features).tanh();raw=(uv[:,:,:3]+delta).clamp(0,1)
    proposed=(uv[:,:,:3]+gate[:,:,None]*delta).clamp(0,1)
    mirror=model.mirror_link(features).squeeze(2);mirror=(mirror+mirror[:,model.mirror])/2
    layer=model.layer_link(features).squeeze(2);layer=(layer+layer[:,model.other])/2
    rgb=tie_material(model,proposed,alpha,mirror,layer)
    return {'alpha':alpha,'alpha_logits':alpha_logits,'alpha_probability':alpha_logits.sigmoid(),'edit_logits':edit_logits,'proposed_rgb':rgb,'untied_rgb':proposed,'raw_rgb':raw,'color_gate_logits':gate_logits,'mirror_link_logits':mirror,'layer_link_logits':layer,'semantic_logits':semantic_logits}


def masked_mean(value,mask):
    return (value*mask).sum()/mask.sum().clamp_min(1)


def balanced_bce(logits,target,positive_weight=1.,mask=None):
    loss=F.binary_cross_entropy_with_logits(logits,target.float(),reduction='none')
    valid=torch.ones_like(target,dtype=torch.bool) if mask is None else mask
    return masked_mean(loss,~target&valid)+positive_weight*masked_mean(loss,target&valid)


def relation_targets(model,truth,classes,symmetric):
    visible=truth[:,:,3]>.5;rgb=truth[:,:,:3]
    same_mirror=(rgb-rgb[:,model.mirror]).abs().amax(2)<.5/255
    eligible=torch.isin(classes,classes.new_tensor([1,3,5]))&(classes==classes[:,model.mirror])
    mirror=symmetric[:,None]&eligible&visible&visible[:,model.mirror]&same_mirror
    mixed=(((classes==3)&(classes[:,model.other]==5))|((classes==5)&(classes[:,model.other]==3)))
    layer=mixed&visible&visible[:,model.other]&((rgb-rgb[:,model.other]).abs().amax(2)<.5/255)
    return mirror,layer


def revision_loss(model,prediction,base,target,labels,symmetric):
    truth=target.flatten(2)[:,:,model.ids].transpose(1,2)
    initial=base.flatten(2)[:,:,model.ids].transpose(1,2)
    alpha=truth[:,:,3];visible=alpha>.5;outer=model.outer[None].expand_as(visible)
    need_edit=(initial[:,:,3]>.5)!=visible
    error=F.binary_cross_entropy_with_logits(prediction['edit_logits'],need_edit.float(),reduction='none')
    preserve=masked_mean(error,outer&~need_edit);correct=masked_mean(error,outer&need_edit)
    occupancy=(5 if model.robust_edits else 2)*preserve+correct
    if model.edit_risk_weight:
        occupancy=cell_edit_risk(error,outer,need_edit,model.edit_risk_weight)
    changed=((truth[:,:,:3]-initial[:,:,:3]).abs().amax(2)>1/255)&visible
    gate=balanced_bce(prediction['color_gate_logits'],changed)
    color=masked_mean((prediction['proposed_rgb']-truth[:,:,:3]).abs().mean(2),visible)
    raw=masked_mean((prediction['raw_rgb']-truth[:,:,:3]).abs().mean(2),changed)
    keep_color=masked_mean((prediction['proposed_rgb']-initial[:,:,:3]).abs().mean(2),visible&~changed)
    classes=labels.flatten(1)[:,model.ids];known=classes>=0
    semantic=F.cross_entropy(prediction['semantic_logits'][known],classes[known]) if known.any() else color*0
    mirror,layer=relation_targets(model,truth,classes,symmetric)
    # Unknown native semantic labels do not mean that a beard relation is absent.
    mirror_known=layer_known=None
    if model.mask_unknown_relations:
        # Different visible target colours are a valid negative even without
        # semantic labels. Equal colours alone do not establish a beard link.
        target_rgb=truth[:,:,:3]
        mirror_conflict=visible&visible[:,model.mirror]&((target_rgb-target_rgb[:,model.mirror]).abs().amax(2)>=.5/255)
        layer_conflict=visible&visible[:,model.other]&((target_rgb-target_rgb[:,model.other]).abs().amax(2)>=.5/255)
        mirror_known=(known&known[:,model.mirror])|mirror_conflict
        layer_known=(known&known[:,model.other])|layer_conflict
    relation=balanced_bce(prediction['mirror_link_logits'],mirror,mask=mirror_known)+balanced_bce(prediction['layer_link_logits'],layer,mask=layer_known)
    # Shared RGB is trained against target colors; pair losses remain useful while links learn.
    rgb=prediction['proposed_rgb'];symmetry=masked_mean((rgb-rgb[:,model.mirror]).abs().mean(2),mirror)
    alignment=masked_mean((rgb-rgb[:,model.other]).abs().mean(2),layer)
    left,right=model.edges;p=prediction['alpha_probability']
    seam=F.smooth_l1_loss(p[:,left]-p[:,right],alpha[:,left]-alpha[:,right])
    total=4*occupancy+8*color+2*raw+2*keep_color+.4*gate+.4*semantic+relation+symmetry+alignment+.3*seam
    boundary=total*0
    if model.boundary_loss_weight:
        from SkingToolkit.dense_uv_parser.head_topology_context import supervised_boundary_loss
        boundary=supervised_boundary_loss(p,alpha,model.edges);total=total+model.boundary_loss_weight*boundary
    return total,{k:v.detach() for k,v in dict(occupancy=occupancy,boundary=boundary,preserve=preserve,correct=correct,color=color,color_gate=gate,relations=relation,symmetry=symmetry,alignment=alignment,empty_outer_cells=(~visible&outer).sum()).items()}


def paired_occupancy_loss(model,prediction,target):
    """Learn changes caused by eyewear while keeping the same person's head.

    The first three rows are a training-only bare/glasses/phones triplet.
    No pair or target information is read during inference.
    """
    truth=target.flatten(2)[:,3,model.ids][:3,model.outer]
    p=prediction['alpha_probability'][:3,model.outer]
    delta=truth[1:]-truth[:1];error=(p[1:]-p[:1]-delta).abs()
    return masked_mean(error,delta!=0)+masked_mean(error,delta==0)


def cell_edit_risk(error,outer,need_edit,preserve_weight):
    """Expected per-texel cost, without renormalizing rare edits separately."""
    weight=torch.where(need_edit,torch.ones_like(error),torch.full_like(error,preserve_weight))
    return masked_mean(error*weight,outer)
