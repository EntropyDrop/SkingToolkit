"""Fingerprint-addressed batch inference with isolated, atomic sample outputs."""
import argparse
import hashlib
import json
import io
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
from SkingToolkit.dense_uv_parser.infer import load_parser, load_view_images, save_parser_uv
from SkingToolkit.dense_uv_parser.accessory_pipeline import load_pipeline, run_pipeline
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image
from SkingToolkit.renderer import DifferentiableRenderer


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--inputs',nargs='+',type=Path)
    parser.add_argument('--pipeline',type=Path)
    parser.add_argument('--save-tensors',action='store_true')
    parser.add_argument('--foreground-probabilities',type=Path,nargs='+',help='One combined grayscale probability image per input; bypass flood only when explicitly supplied')
    opt=parser.parse_args();torch.set_num_threads(4)
    config=load_pipeline(opt.pipeline)
    cases=opt.inputs or [Path(x) for x in json.loads(Path(__file__).with_name('v101_cases.json').read_text())]
    if opt.foreground_probabilities is not None and len(opt.foreground_probabilities)!=len(cases):
        parser.error('Provide exactly one foreground probability image for each input')
    if not all(x.is_file() for x in cases):raise FileNotFoundError([str(x) for x in cases if not x.is_file()])
    # A training job may atomically replace best.pt while inference is starting.
    # Hash and load the same byte snapshot so output provenance stays accurate.
    checkpoint_bytes=opt.checkpoint.read_bytes()
    checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest()
    model,args=load_parser(io.BytesIO(checkpoint_bytes),torch.device('cuda'))
    del checkpoint_bytes
    if opt.pipeline is None and args.get('_v101_pipeline'):
        config=args['_v101_pipeline']
    renderer=DifferentiableRenderer(args['mappings_dir']).cuda()
    opt.output_dir.mkdir(parents=True,exist_ok=True)
    sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    common={'checkpoint':str(opt.checkpoint.resolve()),'checkpoint_sha256':checkpoint_sha256,'pipeline':config,'source_sha256':sources,'save_tensors':opt.save_tensors}
    for case_index,case in enumerate(cases):
        case_bytes=case.read_bytes()
        manifest={**common,'input':str(case.resolve()),'input_sha256':hashlib.sha256(case_bytes).hexdigest()}
        probability_bytes=None
        if opt.foreground_probabilities is not None:
            probability_path=opt.foreground_probabilities[case_index]
            probability_bytes=probability_path.read_bytes()
            manifest['foreground_probability']={'path':str(probability_path.resolve()),'sha256':hashlib.sha256(probability_bytes).hexdigest(),'threshold':config.get('foreground_probability_threshold',.5),'source_threshold':config.get('foreground_source_threshold',.98),'source_inset':config.get('foreground_source_inset',1)}
        fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        # Include both path/input/checkpoint hashes: same basenames cannot collide.
        target=opt.output_dir/(case.stem+'_'+fingerprint[:12])
        if (target/'manifest.json').is_file():
            old=json.loads((target/'manifest.json').read_text())
            if old.get('fingerprint')==fingerprint and old.get('complete'):
                print('cached='+str(target),flush=True);continue
        if target.exists():raise FileExistsError('Incomplete or conflicting output: '+str(target))
        scratch=Path(tempfile.mkdtemp(prefix='.inference-',dir=opt.output_dir))
        try:
            images=load_view_images(SimpleNamespace(combined=io.BytesIO(case_bytes)),config['routing']['views'],renderer).cuda()
            probability=None
            if probability_bytes is not None:
                raw=Image.open(io.BytesIO(probability_bytes))
                if raw.mode not in ('L','F','I;16'):raise ValueError('Foreground probability must be a grayscale image')
                with Image.open(io.BytesIO(case_bytes)) as source_image:
                    if raw.size!=source_image.size:raise ValueError('Foreground probability must match original combined image dimensions')
                p=torch.from_numpy(np.array(raw).astype(np.float32))
                p=p/(65535. if raw.mode=='I;16' else 255.) if raw.mode!='F' else p
                if p.shape[1]%len(config['routing']['views']):raise ValueError('Combined width must divide evenly into views')
                probability=F.interpolate(torch.stack(p.chunk(len(config['routing']['views']),dim=1))[:,None],images.shape[-2:],mode='nearest-exact')[:,0].cuda()
            result=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=probability)
            if probability is not None:
                save_image(probability[:,None].cpu(),scratch/'foreground_probability.png',nrow=len(config['routing']['views']))
                save_image(result['foreground'][:,None].float().cpu(),scratch/'foreground_mask.png',nrow=len(config['routing']['views']))
                save_image(result['foreground_color_sources'][:,None].float().cpu(),scratch/'foreground_color_sources.png',nrow=len(config['routing']['views']))
            save_parser_uv(result['conditioning'],scratch/'parser_only_uv.png')
            for name in ('parser_pred_uv_simple_inpainting.png','pred_uv.png'):
                tensor_to_rgba_image(result['uv'][0]).save(scratch/name)
            save_image(torch.cat(list(result['render'][:,:3]),dim=2).cpu(),scratch/'simple_inpaint_render.png')
            routing=result['details']['routing']
            for layer,name in [(0,'inner'),(1,'outer')]:
                mask=routing['foreground']&(routing['layer']==layer)
                cutout=torch.where(mask[:,None],images[:,:3],torch.full_like(images[:,:3],.5))
                save_image(cutout.cpu(),scratch/f'parser_debug_{name}.png',nrow=len(config['routing']['views']))
            logits=result['outputs'].get('accessory_logits')
            if logits is not None:
                palette=images.new_tensor([[.5,.5,.5],[.1,.9,.5],[.95,.6,.1],[.7,.3,.9]])
                save_image(palette[logits.argmax(1)].permute(0,3,1,2).cpu(),scratch/'accessory_mask.png',nrow=len(config['routing']['views']))
                save_image((1-logits.softmax(1)[:,:1]).cpu(),scratch/'accessory_probability.png',nrow=len(config['routing']['views']))
            # Exact tensors make pixel-stage regression metrics reproducible.
            if opt.save_tensors:
                torch.save({'images':images.cpu(),'outputs':{k:v.cpu() if torch.is_tensor(v) else v for k,v in result['outputs'].items()},
                            'routing':{k:v.cpu() if torch.is_tensor(v) else v for k,v in routing.items()}},scratch/'diagnostics.pt')
            manifest.update(fingerprint=fingerprint,complete=True)
            (scratch/'manifest.json').write_text(json.dumps(manifest,indent=2))
            os.replace(scratch,target);print('completed='+str(target),flush=True)
        except BaseException:
            shutil.rmtree(scratch)
            raise


if __name__=='__main__':main()
