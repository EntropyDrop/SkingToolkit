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
    parser.add_argument('--view-count',type=int,default=2)
    opt=parser.parse_args()
    if opt.view_count<1:parser.error('View count must be positive')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    from transformers import AutoConfig,AutoModelForImageSegmentation
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    model_path=opt.model_dir.resolve()
    model_sources={p.name:sha(p) for p in model_path.glob('*.py')}
    provenance={'model_dir':str(model_path),'model_sha256':sha(model_path/'model.safetensors'),
                'model_code_sha256':model_sources,'config_sha256':sha(model_path/'config.json'),
                'provider_sha256':sha(__file__),'view_count':opt.view_count,'resolution':1024,
                'output':'Segmentation confidence, not a calibrated alpha matte','torch':torch.__version__}
    # Construct normally: older BiRefNet code calls item() while constructing
    # its backbone, which is incompatible with newer transformers meta loading.
    config=AutoConfig.from_pretrained(str(model_path),trust_remote_code=True,local_files_only=True)
    model=AutoModelForImageSegmentation.from_config(config,trust_remote_code=True)
    model.load_state_dict(load_file(model_path/'model.safetensors'),strict=True)
    model=model.eval().cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
    std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    opt.output_dir.mkdir(parents=True,exist_ok=True)
    for path in opt.inputs:
        input_bytes=path.read_bytes()
        manifest={**provenance,'input':str(path.resolve()),'input_sha256':hashlib.sha256(input_bytes).hexdigest()}
        fingerprint=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        target=opt.output_dir/(path.stem+'_'+fingerprint[:12])
        if (target/'manifest.json').is_file():
            prior=json.loads((target/'manifest.json').read_text())
            if prior.get('fingerprint')==fingerprint and (target/'probability.png').is_file():
                print('cached='+str(target/'probability.png'),flush=True);continue
        if target.exists():raise FileExistsError(target)
        im=np.asarray(Image.open(io.BytesIO(input_bytes)).convert('RGB')).copy()
        if im.shape[1]%opt.view_count:raise ValueError('Image width must divide evenly into views')
        rgb=torch.from_numpy(im).permute(2,0,1).float().cuda()/255
        views=torch.stack(rgb.chunk(opt.view_count,dim=2))
        x=F.interpolate(views,size=(1024,1024),mode='bilinear',align_corners=False)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
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
        print('completed='+str(target/'probability.png'),flush=True)


if __name__=='__main__':main()
