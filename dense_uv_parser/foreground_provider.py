"""Offline BiRefNet foreground confidence; kept separate from UV/parser semantics."""
import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True,help='Reviewed local BiRefNet snapshot with code and model.safetensors')
    parser.add_argument('--result-manifest',type=Path,help='Write the exact ordered probability paths for a downstream batch')
    parser.add_argument('--view-count',type=int,default=2)
    parser.add_argument('--checkpoint',type=Path,help='Trained Minecraft foreground decoder checkpoint')
    opt=parser.parse_args()
    if opt.view_count<1:parser.error('View count must be positive')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    if __package__:
        from .foreground_model import load_foreground_model
    else:
        from foreground_model import load_foreground_model
    torch.set_num_threads(4)
    model_path=opt.model_dir.resolve()
    model_sources={p.name:sha(p) for p in model_path.glob('*.py')}
    provenance={'model_dir':str(model_path),'model_sha256':sha(model_path/'model.safetensors'),
                'model_code_sha256':model_sources,'config_sha256':sha(model_path/'config.json'),
                'provider_sha256':sha(__file__),'loader_sha256':sha(Path(__file__).with_name('foreground_model.py')),'view_count':opt.view_count,'resolution':1024,
                'output':'Segmentation confidence, not a calibrated alpha matte','torch':torch.__version__}
    model,trained_metadata=load_foreground_model(model_path,opt.checkpoint)
    if opt.checkpoint:
        provenance['adaptation_checkpoint']=str(opt.checkpoint.resolve())
        provenance['adaptation_sha256']=trained_metadata['checkpoint_sha256']
        provenance['adaptation_version']=trained_metadata['version']
        provenance['output']='Supervised foreground alpha prediction with separate reliable RGB sampling'
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    opt.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for path in opt.inputs:
        input_bytes=path.read_bytes()
        manifest={**provenance,'input':str(path.resolve()),'input_sha256':hashlib.sha256(input_bytes).hexdigest()}
        fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        target=opt.output_dir/(path.stem+'_'+fingerprint[:12])
        if (target/'manifest.json').is_file():
            prior=json.loads((target/'manifest.json').read_text())
            if prior.get('fingerprint')==fingerprint and (target/'probability.png').is_file():
                results.append(str((target/'probability.png').resolve()))
                print('cached='+str(target/'probability.png'),flush=True);continue
        if target.exists():raise FileExistsError(target)
        im=np.asarray(Image.open(io.BytesIO(input_bytes)).convert('RGB')).copy()
        if im.shape[1]%opt.view_count:raise ValueError('Image width must divide evenly into views')
        rgb=torch.from_numpy(im).permute(2,0,1).float().cuda()/255
        views=torch.stack(rgb.chunk(opt.view_count,dim=2))
        x=F.interpolate(views,size=(1024,1024),mode='bilinear',align_corners=False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model((x-mean)/std)[-1]
        confidence=F.interpolate(logits.float().sigmoid(),views.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        if not torch.isfinite(confidence).all():raise RuntimeError('Non-finite foreground prediction')
        combined=torch.cat(list(confidence),dim=1).cpu().numpy()
        # Exact grayscale output is hashed again by batch_accessories.
        with tempfile.TemporaryDirectory(prefix='.foreground-',dir=opt.output_dir) as temporary:
            scratch=Path(temporary)
            Image.fromarray(np.uint8(combined*255)).save(scratch/'probability.png')
            manifest['fingerprint']=fingerprint
            (scratch/'manifest.json').write_text(json.dumps(manifest,indent=2))
            os.replace(scratch,target)
        results.append(str((target/'probability.png').resolve()))
        print('completed='+str(target/'probability.png'),flush=True)
    if opt.result_manifest:
        opt.result_manifest.parent.mkdir(parents=True,exist_ok=True)
        temporary=opt.result_manifest.with_suffix('.tmp')
        temporary.write_text(json.dumps({'probabilities':results,'provenance':provenance},indent=2))
        os.replace(temporary,opt.result_manifest)


if __name__=='__main__':main()
