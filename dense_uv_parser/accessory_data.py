"""Procedural object-labelled skins. Labels describe authored objects, not RGB rules."""
import numpy as np
import torch
from torch.utils.data import Dataset
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin


def make_accessory_skin(original, seed):
    rng = np.random.default_rng(seed)
    uv = original.permute(1, 2, 0).numpy().copy()
    objects = np.zeros((64, 64), dtype=np.int64)
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
    # Eyes, eyebrows and mouth are explicit negative facial examples.
    for x in (9, 13):
        uv[11:12, x:x+2, :3] = rng.uniform(.01,.18,3)
    uv[14,11:13,:3] = rng.uniform(.05,.5,3)
    if rng.random() < .4:
        uv[10,9:11,:3] = hair
        uv[10,13:15,:3] = hair

    def put(face, mask, color, identity, layer=1):
        x, y = faces[face]; x += 32*layer
        patch = uv[y:y+8, x:x+8]
        patch[mask, :3] = color[mask] if np.asarray(color).ndim == 3 else color
        patch[mask, 3] = 1
        objects[y:y+8, x:x+8][mask] = identity

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
    if kind:
        color = rng.uniform(.02,.18,3) if (kind==2 and rng.random()<.65) else (rng.uniform(.03,.85,3) if kind==2 else hair)
        brim_only = kind==2 and rng.random()<.60
        put(4, np.ones((8,8),bool), color, 0 if brim_only else kind, layer=0 if brim_only else 1)
        # A hat is authored once in 3D: its band and brim share the same height
        # and material around the four side faces, including cube seams.
        crown_depth = int(rng.integers(3,6)) if kind==2 else None
        band_color = rng.uniform(.03,.95,3)
        has_band = rng.random()<.75
        for face in (0,1,2,3):
            depth = crown_depth if kind==2 else (int(rng.integers(1,4)) if face==0 else int(rng.integers(2,7)))
            mask = yy < depth
            if kind==3:
                mask |= (yy == depth) & (rng.random((8,8)) > .45)
            tex = np.broadcast_to(color,(8,8,3)).copy()
            tex += rng.normal(0,.025,(8,8,1))
            if kind==2 and has_band: tex[depth-2] = band_color
            if kind==2: tex[depth-1] = color # continuous brim below band
            if brim_only:
                # Taller crown at base radius, complete brim at outer radius.
                # This supplies the visual step seen in top hats; the authored
                # brim is the outer object, independently of its RGB material.
                put(face, yy<depth-1, np.clip(tex,0,1), 0, layer=0)
                mask = yy==depth-1
            put(face, mask, np.clip(tex,0,1), kind)
    return torch.from_numpy(uv).permute(2,0,1).float(), torch.from_numpy(objects)


class AccessoryDataset(Dataset):
    def __init__(self, paths, count, seed):
        self.paths, self.count, self.seed, self.epoch = list(paths), count, seed, 0

    def __len__(self): return self.count

    def __getitem__(self, index):
        seed = self.seed + index*104729 + self.epoch*self.count*104729
        source = load_skin(self.paths[(index + self.epoch*self.count) % len(self.paths)])
        uv, objects = make_accessory_skin(source, seed)
        return {'uv':uv,'objects':objects,'seed':seed}


def render_accessories(uv, objects, renderer, views):
    from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
    images, labels, foregrounds = [], [], []
    for view in views:
        rendered, target = build_dense_parser_batch(uv, renderer, view)
        xy = (target['uv']*63).round().long().clamp(0,63)
        flat = xy[:,1]*64+xy[:,0]
        identity = objects.flatten(1).gather(1,flat.flatten(1)).reshape_as(flat)
        identity = identity.masked_fill((target['layer'] != 1) | (target['foreground'][:,0] < .5), 0)
        images.append(rendered);labels.append(identity);foregrounds.append(target['foreground'][:,0].bool())
    b = uv.shape[0]
    return (torch.stack(images,1).flatten(0,1), torch.stack(labels,1).flatten(0,1),
            torch.stack(foregrounds,1).flatten(0,1))
