import argparse
import uuid
import concurrent.futures
from PIL import Image
import os
import numpy as np
import asyncio
import random
from pathlib import Path

IMAGE_WIDTH  = 512
IMAGE_HEIGHT = 512

SCRIPT_DIR = Path(__file__).resolve().parent
SKIN_MASK = SCRIPT_DIR / "skin-mask.png"
SKIN_DECOR_MASK = SCRIPT_DIR / "skin-decor-mask.png"

bg = (128,128,128)
dot = (255, 255, 255)
alpha_threshold = 128

def create_mask():
    if os.path.exists(SKIN_MASK) and os.path.exists(SKIN_DECOR_MASK):
        return

    # 64*64

    mask = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    decor_mask = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
    for (size, offset, decor_offset) in [
        # head
        [(8,8),(0,8),(32,0)],
        [(8,8),(8,8),(32,0)],
        [(8,8),(8,0),(32,0)],
        [(8,8),(16,8),(32,0)],
        [(8,8),(16,0),(32,0)],
        [(8,8),(24,8),(32,0)],

        #left arm
        [(4,12),(32,52),(16,0)],
        [(4,12),(32+4,52),(16,0)],
        [(4,12),(32+8,52),(16,0)],
        [(4,12),(32+12,52),(16,0)],
        [(4,4),(32+4,48),(16,0)],
        [(4,4),(32+8,48),(16,0)],

        #right arm
        [(4,12),(40,20),(0,16)],
        [(4,12),(40+4,20),(0,16)],
        [(4,12),(40+8,20),(0,16)],
        [(4,12),(40+12,20),(0,16)],
        [(4,4),(40+4,16),(0,16)],
        [(4,4),(40+8,16),(0,16)],

        #body
        [(8,4),(20,16),(0,16)],
        [(8,4),(20+8,16),(0,16)],
        [(8,12),(20,20),(0,16)],
        [(8,12),(20+12,20),(0,16)],
        [(4,12),(16,20),(0,16)],
        [(4,12),(28,20),(0,16)],

        #left leg
        [(4,12),(16,52),(-16,0)],
        [(4,12),(16+4,52),(-16,0)],
        [(4,12),(16+8,52),(-16,0)],
        [(4,12),(16+12,52),(-16,0)],
        [(4,4),(16+4,48),(-16,0)],
        [(4,4),(16+8,48),(-16,0)],

        #right leg
        [(4,12),(0,20),(0,16)],
        [(4,12),(0+4,20),(0,16)],
        [(4,12),(0+8,20),(0,16)],
        [(4,12),(0+12,20),(0,16)],
        [(4,4),(0+4,16),(0,16)],
        [(4,4),(0+8,16),(0,16)],
    ]:
        mask.paste(Image.new('RGBA', size, (bg[0], bg[1], bg[2], 255)), offset)
        decor_mask.paste(Image.new('RGBA', size, (bg[0], bg[1], bg[2], 255)), (offset[0]+decor_offset[0],offset[1]+decor_offset[1]))
    
    mask.save(SKIN_MASK)
    decor_mask.save(SKIN_DECOR_MASK)

create_mask()

def apply_mask(skin_image, skin_mask):
    skin_image = Image.composite(skin_image, skin_mask, skin_mask)
    return skin_image

def create_training_image(skin_image):

    # Mask out any areas not directly mapping to the head, arm, leg, or
    # torso portion of the character.
    skin_mask = Image.open(SKIN_MASK)
    skin_decor_mask = Image.open(SKIN_DECOR_MASK)
    
    skin_mask_np = np.array(skin_mask)
    decor_mask_np = np.array(skin_decor_mask)
    active_mask = (skin_mask_np[..., 3] > 0) | (decor_mask_np[..., 3] > 0)
    
    skin_image = skin_image.resize((64, 64), resample=Image.Resampling.NEAREST)
    skin_arr = np.array(skin_image)

    tr_arr = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 4), [*bg, 255], dtype=np.uint8)
    scaling_ratio = IMAGE_WIDTH // 64
    dot_size = max(1, scaling_ratio // 2)
    dot_start = (scaling_ratio - dot_size) // 2
    dot_end = dot_start + dot_size

    y_indices, x_indices = np.where(active_mask)
    for x, y in zip(x_indices, y_indices):
        x0 = x * scaling_ratio
        y0 = y * scaling_ratio
        block = tr_arr[y0:y0 + scaling_ratio, x0:x0 + scaling_ratio]

        if skin_arr[y, x, 3] < alpha_threshold:
            block[:, :] = [*bg, 255]
            block[dot_start:dot_end, dot_start:dot_end] = [*dot, 255]
        else:
            r, g, b = skin_arr[y, x, :3]
            block[:, :] = [r, g, b, 255]

    training_image = Image.fromarray(tr_arr)

    return training_image

def build_target_img(input_path, output_path):
    # This might fail if dirs not created yet
    # But main process creates them
    try:
        if not os.path.exists(input_path):
            print('not exists', input_path)
            return

        # Ensure subdir exists in output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if os.path.exists(output_path):
            print('exists target', output_path)
            return

        print(f"Processing {input_path}")
        skin_image = Image.open(input_path)
        skin_image = skin_image.convert('RGBA')
        #skin_image = resolve_voxel_consistency(skin_image)
        training_image = create_training_image(skin_image)
        training_image.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a single Minecraft skin to a target image")
    parser.add_argument("input_path", help="Path to input skin image")
    parser.add_argument("output_path", help="Path to output target image")
    args = parser.parse_args()
    build_target_img(args.input_path, args.output_path)
