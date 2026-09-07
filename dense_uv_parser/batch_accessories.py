"""Fingerprint-addressed batch inference with isolated, atomic sample outputs."""
import argparse
import hashlib
import json
import io
import os
import shutil
import tempfile
import subprocess
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
    if opt.foreground_probabilities is None and config.get('foreground_release'):
        # New checkpoints must use their paired trained foreground model even
        # when called through the lower-level batch entry point.
        root=Path(__file__).resolve().parents[1];release=config['foreground_release']
        delta=root/release['checkpoint']
        if not release['passed'] or sha(delta)!=release['checkpoint_sha256']:raise ValueError('Foreground release changed')
        with tempfile.TemporaryDirectory(prefix='foreground-auto-') as temporary:
            index=Path(temporary)/'predictions.json'
            subprocess.run(['/home/ds/miniconda3/envs/comfy/bin/python',str(root/'dense_uv_parser/foreground_provider.py'),'--checkpoint',str(delta),'--model-dir',release['model_dir'],'--output-dir',str(root/'dense_uv_parser/cache/released_foreground'),'--result-manifest',str(index),'--inputs',*[str(x.resolve()) for x in cases]],cwd=root,check=True)
            opt.foreground_probabilities=[Path(x) for x in json.loads(index.read_text())['probabilities']]
            if len(opt.foreground_probabilities)!=len(cases):raise ValueError('Foreground output count does not match inputs')
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
            sidecar=probability_path.parent/'manifest.json'
            if sidecar.is_file():
                provider=json.loads(sidecar.read_text())
                if provider.get('input_sha256')!=manifest['input_sha256']:raise ValueError('Foreground prediction belongs to different input bytes')
                manifest['foreground_provider']={k:provider.get(k) for k in ('model_sha256','adaptation_sha256','adaptation_version','provider_sha256')}
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
            if 'final_head_material_refit' in result:
                (scratch/'final_head_material_refit.json').write_text(json.dumps(result['final_head_material_refit'],indent=2))
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
            ownership=result['outputs'].get('head_ownership_logits')
            if ownership is not None:
                palette=images.new_tensor([[.5,.5,.5],[1.,.65,.4],[.4,.3,.8],[.1,.9,.5],[1.,.2,.3]])
                save_image(palette[ownership.argmax(1)].permute(0,3,1,2).cpu(),scratch/'head_ownership.png',nrow=len(config['routing']['views']))
            headwear=result['outputs'].get('headwear_logits')
            if headwear is not None:
                palette=images.new_tensor([[.5,.5,.5],[.1,.8,.8],[.1,.3,1],[.8,.2,.8],[1,.3,.2],[1,1,.1],[.95,.65,.05]])
                save_image(palette[headwear.argmax(1)].permute(0,3,1,2).cpu(),scratch/'headwear_components.png',nrow=len(config['routing']['views']))
            components=result['outputs'].get('hat_component_logits')
            if components is not None:
                palette=images.new_tensor([[.5,.5,.5],[.2,.8,.8],[.2,.4,1.],[.7,.4,.1],[.9,.2,.5],[1.,.85,.1]])
                save_image(palette[components.argmax(1)].permute(0,3,1,2).cpu(),scratch/'hat_components.png',nrow=len(config['routing']['views']))
            if 'crown_geometry' in result['details']:
                (scratch/'crown_geometry.json').write_text(json.dumps(result['details']['crown_geometry'], indent=2))
                save_image(result['details']['headwear_removed_top_uv'][:,None].float().cpu(), scratch/'crown_removed_top_uv.png')
            if 'joint_head_geometry' in result['details']:
                (scratch/'joint_head_geometry.json').write_text(json.dumps(result['details']['joint_head_geometry'],indent=2))
            # Exact tensors make pixel-stage regression metrics reproducible.
            if opt.save_tensors:
                torch.save({'images':images.cpu(),'outputs':{k:v.cpu() if torch.is_tensor(v) else v for k,v in result['outputs'].items()},
                            'routing':{k:v.cpu() if torch.is_tensor(v) else v for k,v in routing.items()},
                            'canonical_outputs':{k:v.cpu() if torch.is_tensor(v) else v for k,v in result['details']['outputs'].items()},
                            'head_alignment':{k:v.cpu() if torch.is_tensor(v) else v for k,v in result['details'].items() if k in ('joint_head_geometry','joint_head_removed_uv','beard_alignment_removed_uv','beard_alignment_hidden_inner_uv')},
                            'crown_geometry': result['details'].get('crown_geometry'),
                            'headwear_removed_top_uv': result['details'].get('headwear_removed_top_uv', torch.zeros(0)).cpu()},scratch/'diagnostics.pt')
            manifest.update(fingerprint=fingerprint,complete=True)
            (scratch/'manifest.json').write_text(json.dumps(manifest,indent=2))
            os.replace(scratch,target);print('completed='+str(target),flush=True)
        except BaseException:
            shutil.rmtree(scratch)
            raise


if __name__=='__main__':main()
