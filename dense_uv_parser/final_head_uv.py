"""Learn final head UV from images and automatic UV, without identity lookups."""
import os
import torch
from torch import nn
import torch.nn.functional as F
from SkingToolkit.dense_uv_parser.accessories import head_bounds, block
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


def evaluation_numerics():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def image_evidence(images, foreground, outputs):
    y0,y1,x0,x1 = head_bounds(*images.shape[-2:])
    rgb = images[:,:3]*foreground[:,None] + .5*(~foreground[:,None])
    sources = [rgb,foreground[:,None].float(),outputs['head_semantics_logits'].float().softmax(1),outputs['head_surface_logits'].float().softmax(1)]
    return torch.cat([F.interpolate(x[:,:,y0:y1,x0:x1],(56,56),mode='bilinear',align_corners=False) for x in sources],1)


class FinalHeadUVDecoder(nn.Module):
    def __init__(self, width=96, layers=2, revision=1, mappings_dir=None, robust_edits=False, semantic_geometry=False, edit_threshold=.5, edit_risk_weight=0.):
        super().__init__()
        if revision not in (1,2):raise ValueError('Unknown final head decoder revision')
        self.revision=revision
        if robust_edits and revision!=2:raise ValueError('Robust editing requires revision 2')
        self.robust_edits=robust_edits
        if semantic_geometry and revision!=2:raise ValueError("Semantic geometry requires revision 2")
        self.semantic_geometry=semantic_geometry
        if not .5<=edit_threshold<1 or edit_risk_weight<0:raise ValueError("Invalid edit calibration/risk setting")
        if revision!=2 and (edit_threshold!=.5 or edit_risk_weight):raise ValueError("Edit calibration/risk requires revision 2")
        self.edit_threshold=float(edit_threshold);self.edit_risk_weight=float(edit_risk_weight)
        t=build_simple_uv_topology()
        ids=torch.nonzero((t.valid&(t.part==0)).reshape(-1)).flatten()
        node=torch.full((4096,),-1,dtype=torch.long);node[ids]=torch.arange(len(ids))
        mirror=node[t.mirrored_texel.reshape(-1)[ids]]
        layer=t.layer.reshape(-1)[ids]
        other=node[ids+torch.where(layer==0,32,-32)]
        xyz=t.world_position.reshape(-1,3)[ids]-torch.tensor([0.,28.,0.])
        geometry=torch.cat([xyz/5,t.local_uv.reshape(-1,2)[ids],F.one_hot(t.face.reshape(-1)[ids],6),F.one_hot(layer,2)],1).float()
        edges=node[t.outer_edge_index];edges=edges[:,(edges>=0).all(0)]
        for name,value in [('ids',ids),('mirror',mirror),('other',other),('outer',layer==1),('geometry',geometry),('edges',edges)]:
            self.register_buffer(name,value,persistent=False)
        self.image_encoder=nn.Sequential(block(28,32),nn.AvgPool2d(2),block(32,width),nn.AvgPool2d(2))
        self.query=nn.Linear(25,width)
        self.uv_position=nn.Parameter(torch.randn(1,len(ids),width)*.01)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(width,4,width*3,dropout=0.,batch_first=True,norm_first=True),layers)
        self.alpha=nn.Linear(width,1);self.color_gate=nn.Linear(width,1);self.color_delta=nn.Linear(width,3);self.semantic=nn.Linear(width,14)
        for module in (self.alpha,self.color_delta):
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        nn.init.zeros_(self.color_gate.weight);nn.init.constant_(self.color_gate.bias,-2.)
        if self.revision==2:
            from SkingToolkit.dense_uv_parser.final_head_uv_revision import init_revision
            init_revision(self,mappings_dir)

    def forward(self, base_uv, evidence):
        b=base_uv.shape[0]
        if evidence.shape != (b*2,25,56,56):raise ValueError('Expected one front/back evidence pair per UV')
        uv=base_uv.flatten(2)[:,:,self.ids].transpose(1,2)
        geometry=self.geometry[None].expand(b,-1,-1)
        query=self.query(torch.cat([uv,uv[:,self.mirror],uv[:,self.other],geometry],2))+self.uv_position
        if self.revision==2:
            from SkingToolkit.dense_uv_parser.final_head_uv_revision import local_queries
            query=query+local_queries(self,evidence)
        y,x=torch.meshgrid(torch.linspace(-1,1,56,device=evidence.device),torch.linspace(-1,1,56,device=evidence.device),indexing='ij')
        position=torch.stack([x,y])[None].expand(b*2,-1,-1,-1)
        view=(torch.arange(b*2,device=evidence.device)%2).float()[:,None,None,None].expand(-1,1,56,56)
        image=self.image_encoder(torch.cat([evidence,position,view],1))
        memory=image.flatten(2).transpose(1,2).reshape(b,-1,query.shape[-1])
        features=self.decoder(query,memory)
        if self.revision==2:
            from SkingToolkit.dense_uv_parser.final_head_uv_revision import decode_revision
            prediction=decode_revision(self,features,uv)
            values=torch.cat([prediction['proposed_rgb']*prediction['alpha'][:,:,None],prediction['alpha'][:,:,None]],2)
            result=base_uv.flatten(2).clone();result[:,:,self.ids]=values.transpose(1,2)
            prediction['uv']=result.reshape_as(base_uv)
            return prediction
        alpha_logits=(uv[:,:,3]*2-1)*2+self.alpha(features).squeeze(2)
        probability=alpha_logits.sigmoid();alpha=(probability>=.5).float()-probability.detach()+probability
        alpha=torch.where(self.outer[None],alpha,uv[:,:,3])
        gate_logits=self.color_gate(features).squeeze(2);gate_p=gate_logits.sigmoid()
        gate=(gate_p>=.5).float()-gate_p.detach()+gate_p
        delta=self.color_delta(features).tanh()
        proposed=(uv[:,:,:3]+gate[:,:,None]*delta).clamp(0,1)
        values=torch.cat([proposed*alpha[:,:,None],alpha[:,:,None]],2)
        result=base_uv.flatten(2).clone();result[:,:,self.ids]=values.transpose(1,2)
        return {'uv':result.reshape_as(base_uv),'alpha_logits':alpha_logits,'alpha_probability':probability,'proposed_rgb':proposed,'raw_rgb':(uv[:,:,:3]+delta).clamp(0,1),'color_gate_logits':gate_logits,'semantic_logits':self.semantic(features)}


def final_uv_loss(model,prediction,base,target,labels,symmetric):
    if model.revision==2:
        from SkingToolkit.dense_uv_parser.final_head_uv_revision import revision_loss
        return revision_loss(model,prediction,base,target,labels,symmetric)
    truth=target.flatten(2)[:,:,model.ids].transpose(1,2)
    initial=base.flatten(2)[:,:,model.ids].transpose(1,2)
    alpha=truth[:,:,3];outer=model.outer[None].expand_as(alpha);weight=torch.where(alpha>.5,3.,1.)
    weight=weight*torch.where((initial[:,:,3]>.5)!=(alpha>.5),4.,1.)
    occupancy=(F.binary_cross_entropy_with_logits(prediction['alpha_logits'],alpha,reduction='none')*weight*outer).sum()/(weight*outer).sum()
    visible=alpha>.5;changed=(truth[:,:,:3]-initial[:,:,:3]).abs().amax(2)>1/255
    gate=F.binary_cross_entropy_with_logits(prediction['color_gate_logits'],(changed&visible).float())
    color=((prediction['proposed_rgb']-truth[:,:,:3]).abs().mean(2)*visible).sum()/visible.sum().clamp_min(1)
    raw=((prediction['raw_rgb']-truth[:,:,:3]).abs().mean(2)*changed*visible).sum()/(changed*visible).sum().clamp_min(1)
    left,right=model.edges;p=prediction['alpha_probability']
    seam=F.smooth_l1_loss(p[:,left]-p[:,right],alpha[:,left]-alpha[:,right])
    classes=labels.flatten(1)[:,model.ids];known=classes>=0
    semantic=F.cross_entropy(prediction['semantic_logits'][known],classes[known]) if known.any() else p.sum()*0
    pair=symmetric[:,None]&torch.isin(classes,classes.new_tensor([3,5]))&(classes==classes[:,model.mirror])
    symmetry=((p-p[:,model.mirror]).square()*pair).sum()/pair.sum().clamp_min(1)
    rgb=prediction['proposed_rgb'];mirror_rgb=(rgb-rgb[:,model.mirror]).abs().mean(2)
    symmetry=symmetry+(mirror_rgb*pair*visible*visible[:,model.mirror]).sum()/pair.sum().clamp_min(1)
    mixed=(classes==5)&(classes[:,model.other]==3)&visible&visible[:,model.other]
    alignment=((rgb-rgb[:,model.other]).abs().mean(2)*mixed).sum()/mixed.sum().clamp_min(1)
    total=4*occupancy+6*color+raw+.4*gate+.3*seam+.3*semantic+.5*symmetry+.5*alignment
    return total,{k:v.detach() for k,v in dict(occupancy=occupancy,color=color,color_gate=gate,seam=seam,symmetry=symmetry,alignment=alignment,empty_outer_cells=((alpha<.5)&outer).sum()).items()}


@torch.no_grad()
def apply_final_head_uv(model,result):
    decoder=getattr(model,'final_head_uv_decoder',None)
    if decoder is None:return
    details=result['details']
    evidence=image_evidence(details['rendered'],details['routing']['observed_foreground'],details['outputs'])
    before=result['uv'];prediction=decoder(before,evidence);result['uv']=prediction['uv']
    result['final_head_uv']={'alpha_probability':prediction['alpha_probability'],'color_gate_probability':prediction['color_gate_logits'].sigmoid()}
    if decoder.revision==2:
        result['final_head_uv'].update({k:prediction[k] for k in ('mirror_link_logits','layer_link_logits')})
    if not torch.equal(before[:,:,16:],result['uv'][:,:,16:]):raise RuntimeError('Final head decoder modified body UV')
