"""Display exact head UV bytes and alpha differences for manual development review."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

p = argparse.ArgumentParser()
p.add_argument('--label', required=True)
o = p.parse_args()
root = Path('dense_uv_parser').resolve()
run = root / 'runs/v104_topology_20260907'
candidate = run / ('candidate_' + o.label)
report = json.loads((candidate / 'full_review.json').read_text())
rows = list(report['cases'].items())
rows.sort(key=lambda r: r[0] not in ('99LBZPR14PEYLZ69', '254NJBPVUEM759NA', 'beard'))
out = candidate / 'uv_visual_review'
out.mkdir(exist_ok=True)
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)

def rgba(path):
    return np.array(Image.open(path).convert('RGBA'))[:16]

def atlas(a):
    y, x = np.indices(a.shape[:2])
    bg = np.where(((x + y) % 2)[..., None] == 0, 220, 238)
    bg = np.broadcast_to(bg, a[..., :3].shape).astype(np.uint8)
    a = np.where((a[..., 3] > 127)[..., None], a[..., :3], bg)
    return Image.fromarray(a).resize((a.shape[1]*8, a.shape[0]*8), Image.Resampling.NEAREST)

metrics = {}
for start in range(0, len(rows), 5):
    canvas = Image.new('RGB', (1536, 960), 'white')
    d = ImageDraw.Draw(canvas)
    for i, (name, row) in enumerate(rows[start:start+5]):
        old = rgba(Path(row['old_output'])/'pred_uv.png')
        new = rgba(Path(row['output'])/'pred_uv.png')
        a, b = old[:, 32:, 3] > 127, new[:, 32:, 3] > 127
        added, removed = b & ~a, a & ~b
        info = {'added_outer_cells': int(added.sum()), 'removed_outer_cells': int(removed.sum()),
                'inner_alpha_exact': bool(np.array_equal(old[:, :32, 3], new[:, :32, 3]))}
        metrics[name] = info
        y = i*192
        d.text((5, y+3), name+' | '+row['role'], font=font, fill='black')
        for j, (label, uv) in enumerate([('Reviewed v103: inner | outer', old), ('v104: inner | outer', new)]):
            d.text((j*520+5, y+28), label, font=font, fill='black')
            canvas.paste(atlas(uv), (j*520, y+52))
        diff = np.zeros((16, 32, 4), dtype=np.uint8)
        diff[a | b] = [80, 80, 80, 255]
        diff[added] = [20, 180, 85, 255]
        diff[removed] = [220, 40, 110, 255]
        d.text((1045, y+28), f"Outer alpha: +{added.sum()} green / -{removed.sum()} red", font=font, fill='black')
        canvas.paste(atlas(diff), (1040, y+52))
    canvas.save(out/f'page_{start//5+1:02d}.png')
(candidate/'uv_changes.json').write_text(json.dumps(metrics, indent=2)+'\n')
print(json.dumps({'cases':len(rows), 'pages':(len(rows)+4)//5, 'output':str(out)}))
