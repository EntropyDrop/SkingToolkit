import io
import base64
import json
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import sys
import os

sys.setrecursionlimit(64 * 64)

# Load skin-mask.png and skin-decor-mask.png relative to the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKIN_MASK_PATH = os.path.join(SCRIPT_DIR, "skin-mask.png")
SKIN_DECOR_MASK_PATH = os.path.join(SCRIPT_DIR, "skin-decor-mask.png")

def extract_skin(img):
    mask = Image.open(SKIN_MASK_PATH)
    decor_mask = Image.open(SKIN_DECOR_MASK_PATH)
    img = img.convert('RGBA')

    # Detect size and crop to the 2D skin layout region
    w, h = img.size
    if w == 768 and h == 768:
        img = img.crop((0, 0, 384, 384))
    elif w == 512 and h == 1024:
        img = img.crop((0, 0, 512, 512))
    elif w == 512 and h == 512:
        # no crop needed
        pass
    else:
        # Fallback
        img = img.crop((0, 0, w // 2, h // 2))

    bg_color = img.getpixel((0, 0))

    def color_diff(a, b):
        return 0.299 * (a[0] - b[0])**2 + 0.587 * (a[1] - b[1])**2 + 0.114 * (a[2] - b[2])**2

    dot_color = (255, 255, 255)
    ratio = img.width // 64
    ignore_map = {}
    
    mask_np = np.array(mask)
    decor_mask_np = np.array(decor_mask)
    active_mask = (mask_np[..., 3] > 0) | (decor_mask_np[..., 3] > 0)
    
    t1 = 3000
    t2 = 2000
    
    c_idx = ratio // 2
    cor_idx = ratio - 2
    
    for x in range(64):
        for y in range(64):
            # If not in active mask at all, it's statically transparent.
            if not active_mask[y, x]:
                ignore_map[(x, y)] = True
                continue
                
            # If it's in the base layer mask, it's always opaque.
            if mask_np[y, x, 3] > 0:
                c = img.getpixel((x*ratio + c_idx, y*ratio + c_idx))
                # Fill block area
                for i in range(x*ratio, x*ratio + ratio):
                    for j in range(y*ratio, y*ratio + ratio):
                        if i < img.width and j < img.height:
                            img.putpixel((i, j), c)
                continue
            
            # Decor layer only: check for white dot at center
            cx = x * ratio + c_idx
            cy = y * ratio + c_idx
            
            # Center 2x2 pixels
            p1 = img.getpixel((cx - 1, cy - 1))
            p2 = img.getpixel((cx, cy - 1))
            p3 = img.getpixel((cx - 1, cy))
            p4 = img.getpixel((cx, cy))
            
            center_white = (
                color_diff(p1, dot_color) < t1 or
                color_diff(p2, dot_color) < t1 or
                color_diff(p3, dot_color) < t1 or
                color_diff(p4, dot_color) < t1
            )
            
            # Corner pixels
            corner_bg = (
                color_diff(img.getpixel((x * ratio + 1, y * ratio + 1)), bg_color) < t2 or
                color_diff(img.getpixel((x * ratio + cor_idx, y * ratio + cor_idx)), bg_color) < t2 or
                color_diff(img.getpixel((x * ratio + 1, y * ratio + cor_idx)), bg_color) < t2 or
                color_diff(img.getpixel((x * ratio + cor_idx, y * ratio + 1)), bg_color) < t2
            )
            
            if center_white and corner_bg:
                ignore_map[(x, y)] = True
            else:
                c = img.getpixel((cx, cy))
                for i in range(x * ratio, x * ratio + ratio):
                    for j in range(y * ratio, y * ratio + ratio):
                        if i < img.width and j < img.height:
                            img.putpixel((i, j), c)

    img = img.resize((64, 64), Image.BOX)
    for x in range(64):
        for y in range(64):
            if ignore_map.get((x, y), False):
                img.putpixel((x, y), (0, 0, 0, 0))

    img.save('test_tmp.png')
    return img

class ExtractSkinRequest(BaseModel):
    img: str

if __name__ == '__main__':
    # from args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default="test_1.png")
    parser.add_argument('--output', type=str, default="test_output.png")
    parser.add_argument('--server', type=str, default='False')
    args = parser.parse_args()

    if args.server.lower() == 'false':
        img = Image.open(args.img)
        res_img = extract_skin(img)
        res_img.save(args.output)
    else:
        # start server
        app = FastAPI()

        # Configure CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # listen 10010
        @app.post('/extract')
        def extract(request: ExtractSkinRequest):
            try:
                # from request json body img:base64
                img_data = base64.b64decode(request.img)
                img = Image.open(io.BytesIO(img_data))
                
                # response json {img:base64}
                res_img = extract_skin(img)
                
                # Convert PIL Image to base64 PNG
                buffered = io.BytesIO()
                res_img.save(buffered, format="PNG")
                res_img.save('tmp.png')
                img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                return {'img': img_str}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        uvicorn.run(app, host="0.0.0.0", port=10010)
