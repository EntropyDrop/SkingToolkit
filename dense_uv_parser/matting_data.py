"""Background randomization with exact supervised alpha and premultiplied transforms."""
import io,json,math
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def warp_foreground(straight_rgba,theta,size):
    alpha=straight_rgba[3:4]
    premult=torch.cat([straight_rgba[:3]*alpha,alpha],0)[None]
    grid=F.affine_grid(theta[None],(1,4,size,size),align_corners=False)
    warped=F.grid_sample(premult,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0]
    return warped[:3],warped[3:4].clamp(0,1)


def compose(premult,alpha,background):
    return (premult+(1-alpha)*background).clamp(0,1)


def make_composite(rgba,size,seed,stress=True):
    rng=np.random.default_rng(seed)
    def uniform(shape,lo=0.,hi=1.):return torch.from_numpy(rng.uniform(lo,hi,shape).astype('float32'))
    angle=float(rng.uniform(-.045,.045));scale=float(rng.uniform(.78,1.24))
    theta=torch.tensor([[scale*math.cos(angle),-scale*math.sin(angle),rng.uniform(-.13,.13)],
                        [scale*math.sin(angle),scale*math.cos(angle),rng.uniform(-.12,.12)]],dtype=torch.float32)
    premult,alpha=warp_foreground(rgba,theta,size)
    if rng.random()<.04:premult.zero_();alpha.zero_()
    kind=int(rng.integers(0,7))
    if kind==0:bg=uniform((3,1,1)).expand(3,size,size).clone()
    elif kind==1:bg=F.interpolate(uniform((1,3,2,2)),(size,size),mode='bilinear',align_corners=False)[0]
    elif kind==2:
        yy,xx=torch.meshgrid(torch.linspace(-1,1,size),torch.linspace(-1,1,size),indexing='ij')
        phase=(torch.sin((xx*float(rng.uniform(1,8))+yy*float(rng.uniform(1,8)))*math.pi)+1)/2
        bg=uniform((3,1,1))*(1-phase)+uniform((3,1,1))*phase
    elif kind==3:bg=F.interpolate(uniform((1,3,8,8)),(size,size),mode='bicubic',align_corners=False)[0].clamp(0,1)
    elif kind==4:
        yy,xx=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
        tile=int(rng.integers(12,96));pattern=((xx//tile+yy//tile)%2).float()
        bg=uniform((3,1,1))*(1-pattern)+uniform((3,1,1))*pattern
    elif kind==5:
        opaque=rgba[3]>.9
        anchor=rgba[:3,opaque].mean(1) if opaque.any() else uniform((3,))
        bg=(anchor[:,None,None]+uniform((3,1,1),-.12,.12)).clamp(0,1).expand(3,size,size).clone()
    else:
        bg=(F.interpolate(uniform((1,3,4,4)),(size,size),mode='bilinear',align_corners=False)[0]+uniform((3,size,size),-.06,.06)).clamp(0,1)
    if stress:
        # Modify foreground illumination before compositing, keeping true alpha.
        tint=uniform((3,1,1),.85,1.15)
        premult=(premult*tint).clamp_min(0).minimum(alpha)
    image=compose(premult,alpha,bg)
    if stress and rng.random()<.35:
        buffer=io.BytesIO();Image.fromarray((image.permute(1,2,0).numpy()*255).round().astype('uint8')).save(buffer,format='JPEG',quality=int(rng.integers(65,96)))
        image=torch.from_numpy(np.array(Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')).copy()).permute(2,0,1).float()/255
    elif stress and rng.random()<.25:
        image=(image+uniform(tuple(image.shape),-.009,.009)).clamp(0,1)
    return image,alpha,kind


class MattingDataset(Dataset):
    def __init__(self,manifest,split,size=512,limit=None,epoch=0):
        content=json.loads(Path(manifest).read_text())
        if 'canonical_identity_check' not in content:raise ValueError('Dataset must pass normalized skin identity deduplication')
        self.records=[r for r in content['records'] if r['split']==split]
        if limit:self.records=self.records[:limit]
        self.size,self.epoch,self.split=size,epoch,split
        self.offset={'train':710100,'validation':1710100,'test':2710100}[split]
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        rgba=torch.from_numpy(np.array(Image.open(self.records[index]['file']).convert('RGBA')).copy()).permute(2,0,1).float()/255
        image,alpha,kind=make_composite(rgba,self.size,self.offset+index*104729+self.epoch*1000003)
        return {'image':image,'alpha':alpha,'background_kind':kind}
