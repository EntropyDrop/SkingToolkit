import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def alice_to_steve(alice):
    """
    Expands 3px-wide Alex arms to 4px-wide Steve arms in-place on PIL Image.
    """
    # Create a copy so we do not mutate original directly
    img = alice.copy()
    for arm_loc, decor_offset in [((40, 16), (0, 16)), ((32, 48), (16, 0))]:
        for (size, loc, offset_x) in [
            # Top and bottom
            ((5, 4), (arm_loc[0] + 4 + 1, arm_loc[1]), 1),
            ((2, 4), (arm_loc[0] + 4 + 4 + 1, arm_loc[1]), 1),
            # Front and back
            ((9, 12), (arm_loc[0] + 4 + 1, arm_loc[1] + 4), 1),
            ((2, 12), (arm_loc[0] + 4 + 4 + 4 + 1, arm_loc[1] + 4), 1),
        ]:
            for x in range(loc[0] + size[0] - 1, loc[0] - 1, -1):
                for y in range(loc[1], loc[1] + size[1]):
                    img.putpixel((x + offset_x, y), img.getpixel((x, y)))
                    img.putpixel(
                        (x + offset_x + decor_offset[0], y + decor_offset[1]),
                        img.getpixel((x + decor_offset[0], y + decor_offset[1]))
                    )
    return img

def resolve_voxel_consistency(img):
    """
    Solves voxel edge transparent adjacent-face inconsistency.
    """
    is_slim = img.getpixel((47, 52))[3] == 0
    img = img.copy()
    
    parts = [
        # head
        [
            [
                [(8, 8, 8), (8, 8)],     # front
                [(8, 8, 8), (24, 8)],    # back
                [(8, 8, 8), (16, 8)],    # left
                [(8, 8, 8), (0, 8)],     # right
                [(8, 8, 8), (8, 0)],     # top
                [(8, 8, 8), (16, 0)],    # bottom
            ], (32, 0)
        ],
        # body
        [
            [
                [(8, 12, 4), (20, 20)],
                [(8, 12, 4), (20 + 12, 20)],
                [(4, 12, 8), (28, 20)],
                [(4, 12, 8), (16, 20)],
                [(8, 4, 12), (20, 16)],
                [(8, 4, 12), (20 + 8, 16)],
            ], (0, 16)
        ],
        # left arm
        [
            [
                [((3 if is_slim else 4), 12, 4), (32 + 4, 52)],
                [((3 if is_slim else 4), 12, 4), (32 + 12 - (1 if is_slim else 0), 52)],
                [(4, 12, 4), (32 + 8 - (1 if is_slim else 0), 52)],
                [(4, 12, 4), (32, 52)],
                [((3 if is_slim else 4), 4, 12), (32 + 4, 48)],
                [((3 if is_slim else 4), 4, 12), (32 + 8 - (1 if is_slim else 0), 48)],
            ], (16, 0)
        ],
        # right arm
        [
            [
                [((3 if is_slim else 4), 12, 4), (40 + 4, 20)],
                [((3 if is_slim else 4), 12, 4), (40 + 12 - (1 if is_slim else 0), 20)],
                [(4, 12, 4), (40 + 8 - (1 if is_slim else 0), 20)],
                [(4, 12, 4), (40, 20)],
                [((3 if is_slim else 4), 4, 12), (40 + 4, 16)],
                [((3 if is_slim else 4), 4, 12), (40 + 8 - (1 if is_slim else 0), 16)],
            ], (0, 16)
        ],
        # left leg
        [
            [
                [(4, 12, 4), (16 + 4, 52)],
                [(4, 12, 4), (16 + 12, 52)],
                [(4, 12, 4), (16 + 8, 52)],
                [(4, 12, 4), (16, 52)],
                [(4, 4, 12), (16 + 4, 48)],
                [(4, 4, 12), (16 + 8, 48)],
            ], (-16, 0)
        ],
        # right leg
        [
            [
                [(4, 12, 4), (0 + 4, 20)],
                [(4, 12, 4), (0 + 12, 20)],
                [(4, 12, 4), (0 + 8, 20)],
                [(4, 12, 4), (0, 20)],
                [(4, 4, 12), (0 + 4, 16)],
                [(4, 4, 12), (0 + 8, 16)],
            ], (0, 16)
        ],
    ]
    
    for part in parts:
        decor_offset = part[1]
        (x, y, z) = part[0][4][0]
        colors = np.zeros((x, y, z, 4))
        priorities = np.full((x, y, z), 99)
        inverse = {}
        
        for idx, (size, offset) in enumerate(part[0]):
            for dx in range(size[0]):
                for dy in range(size[1]):
                    img_x = offset[0] + dx + decor_offset[0]
                    img_y = offset[1] + dy + decor_offset[1]
                    c = img.getpixel((img_x, img_y))
                    
                    new_x = None
                    new_y = None
                    new_z = None
                    
                    if idx == 4:    # top
                        new_x, new_y, new_z = (dx, y - 1 - dy, z - 1)
                    elif idx == 5:  # bottom
                        new_x, new_y, new_z = (dx, y - 1 - dy, 0)
                    elif idx == 0:  # front
                        new_x, new_y, new_z = (dx, 0, z - 1 - dy)
                    elif idx == 1:  # back
                        new_x, new_y, new_z = (x - 1 - dx, y - 1, z - 1 - dy)
                    elif idx == 2:  # left
                        new_x, new_y, new_z = (x - 1, dx, z - 1 - dy)
                    elif idx == 3:  # right
                        new_x, new_y, new_z = (0, y - 1 - dx, z - 1 - dy)
                        
                    if (new_x, new_y, new_z) not in inverse:
                        inverse[(new_x, new_y, new_z)] = []
                    inverse[(new_x, new_y, new_z)].append((img_x, img_y))
                    
                    if c[3] == 0:
                        continue
                        
                    prio = 99
                    if idx == 0: prio = 0    # front
                    elif idx == 1: prio = 1  # back
                    elif idx == 4: prio = 2  # top
                    elif idx == 5: prio = 3  # bottom
                    elif idx == 2: prio = 4  # left
                    elif idx == 3: prio = 5  # right
                    
                    if priorities[new_x, new_y, new_z] > prio:
                        colors[new_x, new_y, new_z] = c
                        priorities[new_x, new_y, new_z] = prio
                        
        for dx in range(x):
            for dy in range(y):
                for dz in range(z):
                    if (dx, dy, dz) in inverse:
                        if priorities[dx, dy, dz] == 99:
                            continue
                        for i in inverse[(dx, dy, dz)]:
                            existing_c = img.getpixel(i)
                            if existing_c[3] == 0:
                                img.putpixel(i, tuple(colors[dx, dy, dz].astype(int)))
    return img

