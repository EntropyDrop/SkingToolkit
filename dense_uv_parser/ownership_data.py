"""Authored head ownership: abstain, inner face, inner hair, eyewear, headphones.

Real regression examples are never used to produce training labels.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin


def make_ownership_skin(original, seed):
    rng = np.random.default_rng(seed)
    uv = original.permute(1, 2, 0).numpy().copy()
    objects = np.zeros((64, 64), dtype=np.int64)
    ownership = np.zeros((64,64),dtype=np.int64)
    components = np.zeros((64,64),dtype=np.int64) # none, inner crown/band, outer crown/band, brim
    # All six head faces are authored together. Body pixels remain the source skin.
    uv[:16, :32] = 0
    uv[:16, 32:] = 0
    faces = [(8, 8), (24, 8), (0, 8), (16, 8), (8, 0), (16, 0)]
    skin = np.array(rng.choice([[.87,.63,.43],[.45,.28,.19],[.65,.44,.30],
                              [.95,.78,.62],[.32,.20,.13],[.74,.62,.49]]))
    if rng.random() < .12:
        skin = rng.uniform(.2, .9, 3)
    hair = rng.uniform(.03, .30, 3) if rng.random() < .7 else rng.uniform(.1, .85, 3)
    for fi, (x, y) in enumerate(faces):
        tex = np.clip(skin[None,None] + rng.normal(0,.025,(8,8,1)), 0, 1)
        if fi in (1,4): tex[:] = hair + rng.normal(0,.025,(8,8,1))
        if fi in (0,2,3): tex[:rng.integers(1,3)] = hair
        uv[y:y+8, x:x+8, :3] = np.clip(tex,0,1)
        uv[y:y+8, x:x+8, 3] = 1
        own = ownership[y:y+8,x:x+8]
        own[:] = 1
        if fi in (1,4): own[:] = 2
        if fi in (0,2,3):
            # Labels follow the authored hair mask, including its sampled depth.
            own[np.all(tex == hair,axis=2)] = 2
    # Eyes, eyebrows and mouth are explicit negative facial examples.
    for x in (9, 13):
        uv[11:12, x:x+2, :3] = rng.uniform(.01,.18,3)
    uv[14,11:13,:3] = rng.uniform(.05,.5,3)
    if rng.random() < .4:
        uv[10,9:11,:3] = hair
        uv[10,13:15,:3] = hair

    def put(face, mask, color, identity, layer=1, component=0):
        x, y = faces[face]; x += 32*layer
        patch = uv[y:y+8, x:x+8]
        patch[mask, :3] = color[mask] if np.asarray(color).ndim == 3 else color
        patch[mask, 3] = 1
        objects[y:y+8, x:x+8][mask] = identity
        components[y:y+8, x:x+8][mask] = component
        ownership[y:y+8,x:x+8][mask] = 3 if identity==1 else 0

    yy, xx = np.mgrid[:8,:8]
    negative = rng.random() < .18
    if not negative and rng.random() < .78:
        top = int(rng.integers(2,4)); height = int(rng.integers(3,5))
        frame_color = rng.uniform(.025,.23,3) if rng.random()<.70 else rng.uniform(.01,.85,3)
        lens_color = rng.uniform(.05,.95,3)
        frame = np.zeros((8,8),bool); lenses = np.zeros_like(frame)
        # Distinct connected eyewear styles; class ID is independent of palette.
        visor = rng.random()<.20
        boxes = [(0,3),(4,7)] if visor else [(0,2),(5,7)]
        hollow = rng.random()<.30
        rimless_top = rng.random()<.45 and not hollow
        for box_index,(left, right) in enumerate(boxes):
            rectangle = (xx>=left)&(xx<=right)&(yy>=top)&(yy<top+height)
            edge = rectangle & ((xx==left)|(xx==right)|(yy==top)|(yy==top+height-1))
            if rimless_top:
                edge &= (yy!=top)
            # Thin inner rim variants preserve two-column lenses on an 8px face.
            if not visor and not hollow and rng.random()<.65:
                inner_x=right if box_index==0 else left
                edge &= ~((xx==inner_x)&(yy<top+height-1))
            if rng.random() < .2:
                edge &= ~(((xx==left)|(xx==right))&(yy==top+height-1))
            frame |= edge
            lenses |= rectangle & ~edge
        bridge_y = min(7,top+int(rng.integers(0,2)))
        frame[bridge_y,2:6] = True
        lenses[frame] = False
        if hollow: lenses[:] = False  # open-frame holes stay empty
        color = np.broadcast_to(frame_color,(8,8,3)).copy()
        color[lenses] = lens_color
        # Lens shading keeps the same object ID across several colours.
        color[lenses] *= rng.uniform(.65,1.05,(8,8,1))[lenses]
        put(0, frame|lenses, np.clip(color,0,1), 1)
        for face in (2,3):
            length = int(rng.integers(3,9))
            temple = (yy==bridge_y)&(xx<length)
            put(face,temple,frame_color,1)
    kind = int(rng.choice([0,2,3], p=[.35,.40,.25])) if not negative else 0
    if kind==2:
        # Geometry and material are independent variables. A whole hat can have
        # an inner crown/band and an extruded brim; the band is not background.
        color = rng.uniform(.02,.17,3) if rng.random()<.65 else rng.uniform(.05,.85,3)
        depth = int(rng.integers(2,6))
        stepped = rng.random()<.70
        variant = int(rng.choice(3,p=[.75,.15,.10])) # full ring, front visor, partial/asymmetric
        band_color = rng.uniform(.04,.95,3)
        if rng.random()<.3: band_color=np.array([rng.uniform(.5,.95),rng.uniform(.02,.12),rng.uniform(.02,.12)])
        if rng.random()<.2: band_color=color*rng.uniform(.7,1.2)
        has_band = rng.random()<.85
        crown_layer=0 if stepped else 1
        crown_component=1 if stepped else 3
        band_component=2 if stepped else 4
        put(4,np.ones((8,8),bool),color,0 if stepped else 2,layer=crown_layer,component=crown_component)
        for face in (0,1,2,3):
            tex=np.clip(np.broadcast_to(color,(8,8,3))+rng.normal(0,.02,(8,8,1)),0,1)
            crown=yy<depth-1
            put(face,crown,tex,0 if stepped else 2,layer=crown_layer,component=crown_component)
            if has_band:
                put(face,yy==depth-2,np.clip(band_color,0,1),0 if stepped else 2,layer=crown_layer,component=band_component)
            brim=yy==depth-1
            if variant==1 and face!=0: brim[:]=False
            if variant==2 and face in (1,3): brim &= xx<5
            put(face,brim,color,2,component=5)
    elif kind==3:
        put(4,np.ones((8,8),bool),hair,3)
        for face in (0,1,2,3):
            depth=int(rng.integers(1,4)) if face==0 else int(rng.integers(2,7))
            mask=(yy<depth)|((yy==depth)&(rng.random((8,8))>.45))
            tex=np.clip(np.broadcast_to(hair,(8,8,3))+rng.normal(0,.025,(8,8,1)),0,1)
            put(face,mask,tex,3)
    # Headphones are authored as one object across crown and both ears. Palettes
    # are independent of object identity; visible underlying hair is labelled inner.
    if not (components>0).any() and rng.random()<.55:
        band_row=int(rng.integers(1,5));width=int(rng.integers(1,3))
        color=rng.uniform(.02,.9,3); pad_color=rng.uniform(.01,.5,3)
        if rng.random()<.25:color=hair.copy()
        top=(yy>=band_row)&(yy<band_row+width)
        put(4,top,color,0);ownership[:8,40:48][top]=4
        for face in (2,3):
            # Convert the top's z coordinate to the mirrored side-face coordinate.
            column=band_row if face==2 else 8-band_row-width
            stem=(xx>=column)&(xx<column+width)&(yy<=5)
            center=column+width//2
            pad=(xx>=max(0,center-int(rng.integers(1,3))))&(xx<=min(7,center+int(rng.integers(1,3))))&(yy>=3)&(yy<=int(rng.integers(5,8)))
            tex=np.broadcast_to(color,(8,8,3)).copy()
            tex[pad]=pad_color
            accent=pad&(xx>=center)&(yy>=4)&(yy<=5);tex[accent]=color
            mask=stem|pad;put(face,mask,tex,0)
            x,y=faces[face];ownership[y:y+8,x+32:x+40][mask]=4
            # Side hair underneath an opaque earcup is not an observed earcup.
            inner=uv[y:y+8,x:x+8,:3];inner[:,:]=np.clip(hair+rng.normal(0,.025,(8,8,1)),0,1)
            ownership[y:y+8,x:x+8]=2
    return torch.from_numpy(uv).permute(2,0,1).float(),torch.from_numpy(ownership)


class OwnershipDataset(Dataset):
    def __init__(self,paths,count,seed):
        self.paths,self.count,self.seed,self.epoch=list(paths),count,seed,0
    def __len__(self):return self.count
    def __getitem__(self,index):
        seed=self.seed+index*104729+self.epoch*self.count*104729
        uv,labels=make_ownership_skin(load_skin(self.paths[(index+self.epoch*self.count)%len(self.paths)]),seed)
        return {'uv':uv,'ownership':labels}


def render_ownership(uv,labels,renderer,views):
    from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
    images,targets=[],[]
    for view in views:
        image,t=build_dense_parser_batch(uv,renderer,view)
        xy=(t['uv']*63).round().long().clamp(0,63);index=xy[:,1]*64+xy[:,0]
        target=labels.flatten(1).gather(1,index.flatten(1)).reshape_as(index)
        target=target.masked_fill((t['foreground'][:,0]<.5)|(t['part']!=0),0)
        images.append(image);targets.append(target)
    return torch.stack(images,1).flatten(0,1),torch.stack(targets,1).flatten(0,1)
