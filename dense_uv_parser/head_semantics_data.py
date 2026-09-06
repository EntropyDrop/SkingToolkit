"""Authored v102 joint semantics; real reported cases never enter this dataset."""
from functools import lru_cache
import numpy as np
import torch
from torch.utils.data import Dataset
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.ownership_data import make_ownership_skin
from SkingToolkit.dense_uv_parser.headwear_data import make_headwear_skin, make_textured_hair_negative
from SkingToolkit.dense_uv_parser.accessory_data import make_accessory_skin

FACES=((8,8),(24,8),(0,8),(16,8),(8,0),(16,0))


@lru_cache(maxsize=1)
def cap_edges():
    """Map each side's top-row columns to the adjacent top UV edge geometrically."""
    from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
    pos=build_simple_uv_topology().world_position
    top=pos[:8,40:48].reshape(64,3)
    edges=[]
    for x,y in FACES[:4]:
        indices=torch.cdist(pos[y,x+32:x+40],top).argmin(1).numpy()
        yy,xx=indices//8,indices%8
        if len(np.unique(yy))==1:direction=(1 if yy[0]==0 else -1,0)
        else:direction=(0,1 if xx[0]==0 else -1)
        edges.append((yy,xx,direction))
    return edges


def make_joint_skin(original,seed,kind=None,beard_layer=None,partial_hair=None):
    rng=np.random.default_rng(seed+1020906)
    if kind is None:kind=str(rng.choice(['bare','crown','hat','glasses','phones'],p=[.22,.32,.16,.16,.14]))
    uv=original.permute(1,2,0).numpy().copy();uv[:16]=0
    labels=np.zeros((64,64),np.int64);nose=np.zeros_like(labels,bool);beard=np.zeros_like(nose);caps=np.zeros_like(nose)
    yy,xx=np.mgrid[:8,:8]
    skin=rng.uniform(.2,.95,3)
    if rng.random()<.8:skin=np.array(rng.choice([[.96,.78,.66],[.8,.55,.4],[.48,.32,.24],[.27,.19,.16],[.7,.6,.52]]))
    hair=rng.uniform(.01,.15,3) if rng.random()<.65 else rng.uniform(.05,.9,3)
    shade=rng.uniform(.65,1.2,(4,4,1)).repeat(2,0).repeat(2,1)
    hair_material=np.clip(hair[None,None]*shade+rng.normal(0,.018,(8,8,1)),0,1)
    if rng.random()<.35:
        tiles=original[:3,8:16,8:16].permute(1,2,0).numpy()
        hair_material=np.clip(.8*hair_material+.2*tiles[rng.permutation(8)],0,1)
    hair_layer=int(rng.random()<(.15 if kind in ('crown','hat') else .35))
    # Partial shells teach local layer decisions: front fringe can protrude
    # while the top/back hair remains on the base head.
    partial_hair=(rng.random()<.45 and kind not in ('crown','hat')) if partial_hair is None else partial_hair
    if partial_hair:hair_layer=1
    fringe_depth=int(rng.integers(1,4))
    fringe_wrap=rng.random()<.3
    depth=int(rng.integers(1,5));fringe=np.maximum(1,depth+rng.integers(-1,2,(1,8)))
    for fi,(x,y) in enumerate(FACES):
        base=np.clip(skin[None,None]+rng.normal(0,.02,(8,8,1)),0,1)
        uv[y:y+8,x:x+8,:3]=base;uv[y:y+8,x:x+8,3]=1;labels[y:y+8,x:x+8]=1
        if fi in (1,4):mask=np.ones((8,8),bool)
        elif fi==5:mask=np.zeros((8,8),bool)
        elif fi==0:mask=yy<fringe
        else:mask=(yy<rng.integers(3,7))|(((xx<2) if fi==2 else (xx>5))&(yy<7))
        # Inner substrate is authored too; an outer hair shell has real support.
        uv[y:y+8,x:x+8,:3][mask]=hair_material[mask];labels[y:y+8,x:x+8][mask]=2
        if hair_layer:
            shell=mask.copy()
            if partial_hair:
                if fi==0:shell &= yy<fringe_depth
                elif fi in (2,3):shell &= (yy<fringe_depth)&((xx>=6) if fi==2 else (xx<2))&fringe_wrap
                elif fi==4:
                    shell[:]=False
                    ey,ex,_=cap_edges()[0];shell[ey,ex]=True
                else:shell[:]=False
            uv[y:y+8,x+32:x+40,:3][shell]=hair_material[shell]
            uv[y:y+8,x+32:x+40,3][shell]=1;labels[y:y+8,x+32:x+40][shell]=4
    # Distinct eyebrows/eyes/nose shading remain facial texture, in many palettes.
    eye_y=int(rng.integers(3,5))
    for x in (1,5):
        uv[8+eye_y,8+x:10+x,:3]=rng.uniform(.05,.9,3);labels[8+eye_y,8+x:10+x]=1
        if eye_y>2 and rng.random()<.5:
            uv[7+eye_y,8+x:10+x,:3]=hair;labels[7+eye_y,8+x:10+x]=1
    if rng.random()<.88:
        width=int(rng.integers(1,3));height=int(rng.integers(1,3));nx=int(rng.integers(3,5-width+1));ny=min(6,eye_y+1)
        nose[8+ny:min(16,8+ny+height),8+nx:8+nx+width]=True
        tint=skin*rng.uniform(.6,1.05)+rng.uniform(-.06,.06,3)
        if rng.random()<.5:tint+=np.array([.08,-.04,-.04])
        uv[:,:,:3][nose]=np.clip(tint,0,1);labels[nose]=1
    layer=int(rng.choice([0,1,2],p=[.45,.2,.35])) if beard_layer is None else beard_layer
    if layer not in (0,1,2):raise ValueError('beard_layer must be 0 (inner), 1 (outer), or 2 (aligned mixed)')
    if rng.random()<.72 or beard_layer is not None:
        color=np.clip(hair*rng.uniform(.65,1.5)+rng.uniform(0,.04,3),0,1)
        symmetric=rng.random()<.8
        mask=(yy>=int(rng.integers(5,7)))|((yy>=4)&((xx==0)|(xx==7)))
        if rng.random()<.6:mask |= (yy==5)&(xx>=2)&(xx<=5)
        # An open mouth preserves facial texture within the beard object.
        mask[6,3:5]=False
        if not symmetric:mask[:,int(rng.integers(0,3))]=False
        for fi in (0,2,3,5):
            x,y=FACES[fi]
            if fi==0:m=mask
            elif fi==5:m=yy>=5
            else:
                # Cube UV sides have opposite horizontal orientation. Match
                # the actual front seam, not the rear edge of the side face.
                m=(yy>=6)&((xx>=5) if fi==2 else (xx<3))
                m[:,7 if fi==2 else 0]=mask[:,0 if fi==2 else 7]
            material=np.clip(color[None,None]*rng.uniform(.8,1.15,(8,8,1)),0,1)
            if symmetric and fi==0:material=(material+material[:,::-1])/2
            # Mixed beards share face-local coordinates and material. Visible
            # inner beard remains around an outer subset; holes exist in both.
            masks=[(layer,m)] if layer!=2 else [(0,m),(1,m & ((yy>=7) if fi!=5 else (yy>=5)))]
            if layer==2 and fi==0:
                outer=m & ((yy>=7)|((yy==5)&(xx>=2)&(xx<=5)))
                masks=[(0,m),(1,outer)]
            for target_layer,selected in masks:
                tx=x+32*target_layer
                uv[y:y+8,tx:tx+8,:3][selected]=material[selected];uv[y:y+8,tx:tx+8,3][selected]=1
                labels[y:y+8,tx:tx+8][selected]=5 if target_layer else 3;beard[y:y+8,tx:tx+8][selected]=True
    # Features covered by authored foreground objects are not visible facial labels.
    if kind=='crown':
        metal=rng.uniform(.05,.98,3)
        if rng.random()<.4:metal=np.array([rng.uniform(.6,1),rng.uniform(.45,.9),rng.uniform(.05,.3)])
        highlight=np.clip(metal*rng.uniform(1,1.7)+rng.uniform(0,.2,3),0,1)
        band_y=int(rng.integers(1,4));thick=int(rng.integers(1,3))
        period=int(rng.integers(2,5));phase=int(rng.integers(period));width=int(rng.integers(1,3))
        top=np.zeros((8,8),bool);extension=int(rng.choice([0,1,2,3],p=[.2,.2,.4,.2]))
        for fi,(x,y) in enumerate(FACES[:4]):
            tooth=((np.arange(8)+phase)%period)<width
            if rng.random()<.7:tooth[[0,7]]=True
            if rng.random()<.15:tooth[int(rng.integers(8))]=False
            mask=((yy>=band_y)&(yy<band_y+thick))|((yy<band_y)&tooth[None])
            tex=np.clip(metal[None,None]*rng.uniform(.7,1.15,(8,8,1)),0,1)
            tex[:,::period]=highlight
            gems=(rng.random((8,8))<.07)&mask;tex[gems]=rng.uniform(.05,1,(gems.sum(),3))
            uv[y:y+8,x+32:x+40,:3][mask]=tex[mask];uv[y:y+8,x+32:x+40,3][mask]=1;labels[y:y+8,x+32:x+40][mask]=13
            ey,ex,(dy,dx)=cap_edges()[fi]
            for col in np.flatnonzero(tooth):
                for distance in range(extension):
                    top[ey[col]+distance*dy,ex[col]+distance*dx]=True
        uv[:8,40:48,:3][top]=np.broadcast_to(metal,(8,8,3))[top]
        uv[:8,40:48,3][top]=1;labels[:8,40:48][top]=13;caps[:8,40:48]=top
    elif kind=='hat':
        for offset in range(100):
            donor,_,components=make_accessory_skin(original,seed+offset*7919,return_components=True)
            if (components>0).any():break
        else:raise RuntimeError('No authored hat')
        selected=components.numpy()>0
        # Hat geometry takes precedence over the separately authored hair shell.
        hair_outer=(labels==4);uv[hair_outer]=0;labels[hair_outer]=0
        uv[selected]=donor.permute(1,2,0).numpy()[selected]
        labels[selected]=components.numpy()[selected]+7
    elif kind in ('glasses','phones'):
        requested=3 if kind=='glasses' else 4
        for offset in range(200):
            donor,own=make_ownership_skin(original,seed+offset*7919)
            if (own==requested).any():break
        else:raise RuntimeError('No authored object')
        selected=own.numpy()==requested
        uv[selected]=donor.permute(1,2,0).numpy()[selected];labels[selected]=6 if kind=='glasses' else 7
    nose &= labels==1;beard &= (labels==3)|(labels==5);caps &= labels==13
    return {'uv':torch.from_numpy(uv).permute(2,0,1).float(),'labels':torch.from_numpy(labels),
            'nose':torch.from_numpy(nose),'beard':torch.from_numpy(beard),'caps':torch.from_numpy(caps)}


class JointHeadDataset(Dataset):
    """60% new joint labels, 20% legacy ownership, 20% legacy headwear/replay."""
    def __init__(self,paths,count,seed,joint_only=False):
        self.paths,self.count,self.seed,self.epoch,self.joint_only=list(paths),count,seed,0,joint_only
    def __len__(self):return self.count
    def __getitem__(self,index):
        seed=self.seed+index*104729+self.epoch*self.count*104729
        original=load_skin(self.paths[(index+self.epoch*self.count)%len(self.paths)])
        mode=0 if self.joint_only else (1 if index%5==3 else 2 if index%5==4 else 0)
        if mode==0:item=make_joint_skin(original,seed)
        elif mode==1:
            uv,labels=make_ownership_skin(original,seed);item={'uv':uv,'labels':labels}
        else:
            uv,labels=(make_textured_hair_negative(original,seed) if index%10==4 else make_headwear_skin(original,seed));item={'uv':uv,'labels':labels}
        for key in ('nose','beard','caps'):item.setdefault(key,torch.zeros(64,64,dtype=torch.bool))
        item['mode']=mode
        return item


def render_joint_batch(batch,renderer,views):
    from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
    images=[];labels=[];features=[]
    uv=batch['uv'];truth=batch['labels'];masks=torch.stack([batch[k] for k in ('nose','beard','caps')],1)
    for view in views:
        image,t=build_dense_parser_batch(uv,renderer,view)
        xy=(t['uv']*63).round().long().clamp(0,63);index=xy[:,1]*64+xy[:,0]
        valid=(t['foreground'][:,0]>.5)&(t['part']==0)
        target=truth.flatten(1).gather(1,index.flatten(1)).reshape_as(index).masked_fill(~valid,0)
        feature=masks.flatten(2).gather(2,index.flatten(1)[:,None].expand(-1,3,-1)).reshape(-1,3,*index.shape[-2:])&valid[:,None]
        images.append(image);labels.append(target);features.append(feature)
    return (torch.stack(images,1).flatten(0,1),torch.stack(labels,1).flatten(0,1),
            batch['mode'].repeat_interleave(len(views)),torch.stack(features,1).flatten(0,1))
