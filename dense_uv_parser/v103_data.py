"""V103 synthetic UV supervision; user review examples are never training data."""
import numpy as np
import torch
from torch.utils.data import Dataset
from SkingToolkit.dense_uv_parser.head_semantics_data import make_joint_skin, JointHeadDataset, cap_edges
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology


def make_structured_skin(original, seed, symmetric=None, profile=None):
    rng = np.random.default_rng(seed + 1030906)
    symmetric = bool(rng.random() < .75) if symmetric is None else symmetric
    profile = str(rng.choice(["flat", "fringe", "capped", "shell"], p=[.25,.35,.25,.15])) if profile is None else profile
    if profile not in ("flat", "fringe", "capped", "shell"):
        raise ValueError("Unknown hair profile")
    mixed = bool(rng.random() < .65)
    item = make_joint_skin(original, seed, kind="bare", beard_layer=2 if mixed else int(rng.integers(2)))
    uv = item["uv"].permute(1,2,0).numpy().copy()
    labels = item["labels"].numpy().copy()
    topo = build_simple_uv_topology()
    mirror = topo.mirrored_texel.numpy().reshape(-1)
    head = (topo.valid & (topo.part == 0)).numpy()
    outer = head & (topo.layer.numpy() == 1)

    # Author all six faces together. V102 randomized the two side materials
    # independently even when its front face was marked symmetric.
    if symmetric:
        flat = labels.reshape(-1)
        pixels = uv.reshape(-1,4)
        for cls in (3,5):
            selected = (flat == cls) | (flat[mirror] == cls)
            for i in np.flatnonzero(selected):
                j = int(mirror[i])
                if i > j:
                    continue
                sources = [k for k in (i,j) if flat[k] == cls]
                color = pixels[sources,:3].mean(0)
                pixels[[i,j],:3] = color
                pixels[[i,j],3] = 1
                flat[[i,j]] = cls
        if mixed:
            # Color agreement uses the same geometric UV pair, across layers.
            for i in np.flatnonzero(flat == 5):
                j = i - 32
                pixels[j] = pixels[i]
                flat[j] = 3
    else:
        # Explicit asymmetric examples prevent universal mirroring at inference.
        selected = (labels == 5)
        selected[:,44:48] = False
        uv[selected] = 0
        labels[selected] = 0

    # Re-author outer hair independently of beard. A fringe need not imply a
    # full top edge or a shell over the rear of the head.
    old_hair = labels == 4
    uv[old_hair] = 0
    labels[old_hair] = 0
    if profile == "shell":
        for y,x in zip(*np.where((labels == 2) & head)):
            if labels[y,x+32] == 0:
                uv[y,x+32] = uv[y,x]
                labels[y,x+32] = 4
    elif profile in ("fringe", "capped"):
        height = int(rng.integers(1,4))
        columns = np.ones(8, bool)
        if rng.random() < .5:
            columns[[0,7]] = False
        for y in range(8,8+height):
            for col in np.flatnonzero(columns):
                x = 8+col
                if labels[y,x] == 2 and labels[y,x+32] == 0:
                    uv[y,x+32] = uv[y,x]
                    labels[y,x+32] = 4
        if rng.random() < .5:
            for col in range(2):
                for x in (7-col,16+col):
                    for y in range(8,8+height):
                        if labels[y,x] == 2 and labels[y,x+32] == 0:
                            uv[y,x+32] = uv[y,x]
                            labels[y,x+32] = 4
        if profile == "capped":
            ey,ex,(dy,dx) = cap_edges()[0]
            for col in np.flatnonzero(columns):
                for depth in range(int(rng.integers(1,3))):
                    y,x = int(ey[col]+dy*depth),int(ex[col]+dx*depth)
                    uv[y,40+x] = uv[y,8+x]
                    labels[y,40+x] = 4
    item["uv"] = torch.from_numpy(uv).permute(2,0,1).float()
    item["labels"] = torch.from_numpy(labels)
    item["beard"] = (item["labels"] == 3) | (item["labels"] == 5)
    item["symmetric"] = symmetric
    item["profile"] = profile
    return item


class StructuredHeadDataset(Dataset):
    """Half targeted structures, half v102 joint/ownership/headwear replay."""
    def __init__(self, paths, count, seed):
        self.paths,self.count,self.seed,self.epoch = list(paths),count,seed,0
        self.replay = JointHeadDataset(paths,count,seed+19000000)

    def __len__(self):
        return self.count

    def __getitem__(self,index):
        if index % 2 == 0:
            seed = self.seed + 104729*(index+self.epoch*self.count)
            original = load_skin(self.paths[(index+self.epoch*self.count)%len(self.paths)])
            item = make_structured_skin(original,seed)
            item["mode"] = 0
        else:
            self.replay.epoch = self.epoch
            item = self.replay[index]
            item["symmetric"] = False
            item["profile"] = "replay"
        return item
