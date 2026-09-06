"""Audit frozen foreground predictions, including disconnected real character details."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter


def inspect_real(directory):
    root = Path(__file__).parent
    rows = []
    for manifest_path in sorted(Path(directory).glob('*/manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        source = Path(manifest['input'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest['input_sha256']:
            raise ValueError('Regression input changed: ' + str(source))
        probability = np.array(Image.open(manifest_path.parent/'probability.png'))/255.
        row = {'input': str(source), 'foreground_sha256': manifest.get('adaptation_sha256')}
        if source.stem in {'1H6GAQA9BENWNB89_edited','2RSZETA9PCRZHVEU_edited','TWRLRRTHQP2UV368_edited','PU418PLGERMFC487_edited','img3','img23','img27','img28'}:
            # Inspected empty background above the left sloping head edge.
            row['annotated_background_false_positive_rate'] = [
                float((v[23:28,160:175]>=.5).mean()) for v in np.split(probability,2,axis=1)[:(2 if source.stem in {'TWRLRRTHQP2UV368_edited','PU418PLGERMFC487_edited','img3'} else 1)]]
        if 'img28' == source.stem:
            # Manually inspected interior rectangles at original 1024 x 1024.
            # These are true disconnected blue character details, not background.
            if probability.shape != (1024,1024):
                raise ValueError('img28 annotation requires original dimensions')
            boxes = [(262,860,280,880), (216,897,232,912),
                     (774,860,792,880), (728,897,744,912)]
            row['disconnected_detail_recall'] = [
                float((probability[y1:y2,x1:x2]>=.5).mean()) for x1,y1,x2,y2 in boxes]
        if 'PU418' in source.stem or 'TWRLR' in source.stem:
            views = np.split(probability,2,axis=1)
            native = [np.array(Image.fromarray(v.astype('float32')).resize((256,512),Image.Resampling.NEAREST)) for v in views]
            if 'PU418' in source.stem:
                mask = np.array(Image.open(root/'runs/v101_hat_audit_20260906/hat_brim_development_mask.png').filter(ImageFilter.MinFilter(3)))>0
                row['hat_brim_recall'] = [float((v[mask]>=.5).mean()) for v in native]
            else:
                mask = np.array(Image.open(root/'regression/TWRLRRTHQP2UV368_glasses.png').filter(ImageFilter.MinFilter(5)))>0
                row['glasses_recall'] = float((native[0][mask]>=.5).mean())
        rows.append(row)
    if not rows:
        raise ValueError('No prediction manifests')
    return {'role':'real development annotations; never gradient training or held-out test', 'cases': rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probability-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    opt=p.parse_args()
    result=inspect_real(opt.probability_dir)
    opt.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
