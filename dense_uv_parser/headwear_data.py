"""Authored headwear instances: whole bands and open, jewelled royal crowns."""
import numpy as np
import torch
from torch.utils.data import Dataset
from SkingToolkit.dense_uv_parser.accessory_data import make_accessory_skin, render_accessories
from SkingToolkit.dense_uv_parser.ownership_data import make_ownership_skin
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin


def make_headwear_skin(original, seed):
    rng = np.random.default_rng(seed+807101)
    if rng.random() < .28:
        # Ear cups and a top band can resemble a crown in a head-only crop.
        # Keep their complete authored geometry as explicit hard negatives.
        for offset in range(100):
            negative_uv, ownership = make_ownership_skin(original, seed+offset*7919)
            if (ownership==4).any():
                return negative_uv, torch.zeros((64,64),dtype=torch.long)
        raise RuntimeError("Failed to generate headphone negative")
    uv, objects, components = make_accessory_skin(original, seed, return_components=True)
    if rng.random() < .5:
        return uv, components
    # Start with explicitly authored negative face/hair/eyewear/headphones. A
    # crown is a thin object around hair; it never labels the exposed hair top.
    uv, _ = make_ownership_skin(original, seed+915103)
    uv = uv.permute(1,2,0).numpy().copy()
    uv[:16,32:] = 0
    components = np.zeros((64,64), dtype=np.int64)
    # Some samples are difficult negatives with no headwear at all.
    negative = rng.random() < .22
    faces = [(8,8),(24,8),(0,8),(16,8),(8,0)]
    # Reauthor bare head: the negative generator may itself have had a hat.
    hair = rng.uniform(.015,.40,3) if rng.random()<.7 else rng.uniform(.1,.9,3)
    skin = rng.uniform(.25,.9,3)
    for fi,(x,y) in enumerate(faces):
        base = np.broadcast_to(hair if fi in (1,4) else skin,(8,8,3)).copy()
        if fi in (0,2,3): base[:int(rng.integers(2,5))] = hair
        uv[y:y+8,x:x+8,:3] = np.clip(base+rng.normal(0,.025,(8,8,1)),0,1)
        uv[y:y+8,x:x+8,3] = 1
    uv[11:13,9:11,:3] = rng.uniform(0,.9,3)
    uv[11:13,13:15,:3] = rng.uniform(0,.9,3)
    if negative:
        return torch.from_numpy(uv).permute(2,0,1).float(), torch.from_numpy(components)
    yy,xx = np.mgrid[:8,:8]
    metal = np.array([rng.uniform(.65,1),rng.uniform(.42,.85),rng.uniform(.02,.3)])
    if rng.random()<.45: metal = rng.uniform(.1,.95,3)  # palette is not identity
    band_y = int(rng.integers(1,4)); thickness = int(rng.integers(1,3))
    period = int(rng.integers(2,5)); phase = int(rng.integers(period))
    for fi,(x,y) in enumerate(faces):
        x += 32
        if fi==4:
            # Disconnected teeth at the top rim; the centre remains open.
            mask=((yy==0)|(yy==7)|(xx==0)|(xx==7)) & (((xx+yy+phase)%period)==0)
            if rng.random()<.5: mask[:]=False
        else:
            ring=(yy>=band_y)&(yy<band_y+thickness)
            tooth=((xx+phase)%period==0)|(xx==0)|(xx==7)
            mask=ring|((yy<band_y)&tooth)
        color=np.clip(metal[None,None]*rng.uniform(.65,1.2,(8,8,1)),0,1)
        gems=(rng.random((8,8))<.10)&mask
        color[gems]=rng.uniform(.05,1,(gems.sum(),3))
        patch=uv[y:y+8,x:x+8]
        patch[mask,:3]=color[mask];patch[mask,3]=1
        components[y:y+8,x:x+8][mask]=6
    return torch.from_numpy(uv).permute(2,0,1).float(),torch.from_numpy(components)


class HeadwearDataset(Dataset):
    def __init__(self, paths, count, seed):
        self.paths, self.count, self.seed, self.epoch = list(paths), count, seed, 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        seed=self.seed+index*104729+self.epoch*self.count*104729
        source=load_skin(self.paths[(index+self.epoch*self.count)%len(self.paths)])
        uv,components=make_headwear_skin(source,seed)
        return {'uv':uv,'components':components}


def render_headwear(uv, components, renderer, views):
    images,_,_,labels=render_accessories(uv,torch.zeros_like(components),renderer,views,components)
    return images,labels


def make_textured_hair_negative(original,seed):
    """Author hair geometry with rich material patches, never a crown silhouette.

    Shuffled source texture tiles supply realistic shading statistics. Their
    original arrangement/labels are not reused as semantic ground truth.
    """
    rng=np.random.default_rng(seed+123077)
    uv,_=make_ownership_skin(original,seed)
    uv=uv.clone();uv[:,:16,32:]=0
    yy,xx=np.mgrid[:8,:8]
    faces=[(8,8),(24,8),(0,8),(16,8),(8,0)]
    tint=torch.from_numpy(rng.uniform(.1,.95,3)).float()[:,None,None]
    shade=torch.from_numpy(rng.uniform(.25,1,(1,8,8))).float()
    material=tint*shade
    # Coherent strands and blocks span several source pixels; Gaussian noise
    # alone did not cover the edited images' complex hair materials.
    if rng.random()<.7:
        material=material.reshape(3,4,2,4,2).mean((2,4)).repeat_interleave(2,1).repeat_interleave(2,2)
    sample=original[:3,8:16,8:16].clone()
    tiles=sample.reshape(3,4,2,4,2).permute(1,3,0,2,4).reshape(16,3,2,2)
    tiles=tiles[torch.from_numpy(rng.permutation(16))]
    shuffled=tiles.reshape(4,4,3,2,2).permute(2,0,3,1,4).reshape(3,8,8)
    if rng.random()<.5:material=.6*shuffled+.4*material
    for fi,(x,y) in enumerate(faces):
        # Visible bare skin under the deliberately authored hair boundary.
        uv[:3,y:y+8,x:x+8]=torch.from_numpy(rng.uniform(.25,.85,3)).float()[:,None,None]
        uv[3,y:y+8,x:x+8]=1
        depth=int(rng.integers(2,5)) if fi==0 else int(rng.integers(4,8))
        mask=torch.from_numpy((yy<depth)|((yy==depth)&(xx%2==0)))
        if fi==4:mask[:]=True
        layer=1 if rng.random()<.75 else 0
        patch=uv[:,y:y+8,x+32*layer:x+32*layer+8]
        patch[:3,mask]=material[:,mask];patch[3,mask]=1
    uv[:3,12,9:11]=torch.from_numpy(rng.uniform(.01,.8,3)).float()[:,None]
    uv[:3,12,13:15]=torch.from_numpy(rng.uniform(.01,.8,3)).float()[:,None]
    return uv,torch.zeros((64,64),dtype=torch.long)


class HeadwearPresenceDataset(HeadwearDataset):
    def __getitem__(self,index):
        seed=self.seed+index*104729+self.epoch*self.count*104729
        if index%2:return super().__getitem__(index)
        source=load_skin(self.paths[(index+self.epoch*self.count)%len(self.paths)])
        uv,components=make_textured_hair_negative(source,seed)
        return {'uv':uv,'components':components}
