"""Visually traced development regression, kept separate from training/test data."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from SkingToolkit.dense_uv_parser.infer import load_view_images
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image


@torch.no_grad()
def evaluate_real_regression(model,renderer,config,output_dir,step):
    root=Path(__file__).parent
    metadata=json.loads((root/'v101_real_regression.json').read_text())
    image_path=Path(metadata['combined'])
    if hashlib.sha256(image_path.read_bytes()).hexdigest()!=metadata['source_sha256']:
        raise ValueError('Regression input changed; review the annotation before using it')
    images=load_view_images(SimpleNamespace(combined=image_path),config['routing']['views'],renderer).cuda()
    result=run_pipeline(model,renderer,images,config,complete=True)
    baseline=run_pipeline(model,renderer,images,config,outputs={k:v for k,v in result['outputs'].items() if k!='accessory_logits'})
    annotation=torch.from_numpy(np.array(Image.open(root/metadata['mask']))>0).cuda()
    eroded=(-F.max_pool2d(-annotation[None,None].float(),5,1,2))[0,0]>.5
    expanded=F.max_pool2d(annotation[None,None].float(),5,1,2)[0,0]>.5
    uncertain=expanded&~eroded
    pred=result['outputs']['accessory_logits'][0].softmax(0)
    glasses=(pred.argmax(0)==1)&result['details']['routing']['accessory_supported'][0]
    valid=result['foreground'][0]&~uncertain
    tp=int((glasses&annotation&valid).sum());fp=int((glasses&~annotation&valid).sum());fn=int((~glasses&annotation&valid).sum())
    r=result['details']['routing'];br=baseline['details']['routing']
    final_outer=r['foreground'][0]&(r['layer'][0]==1)
    base_outer=br['foreground'][0]&(br['layer'][0]==1)
    rgb=images[0,:3];yy=torch.arange(rgb.shape[1],device=rgb.device)[:,None];xx=torch.arange(rgb.shape[2],device=rgb.device)[None,:]
    green=(rgb[1]>rgb[0]+.12)&(rgb[1]>rgb[2]+.14)&(rgb[1]>.25)&(yy<145)&(xx>90)
    metrics={'scope':metadata['role'],'glasses_mask_precision':tp/max(1,tp+fp),'glasses_mask_recall':tp/max(1,tp+fn),
             'whole_glasses_outer_recall':float(final_outer[eroded].float().mean()),
             'baseline_whole_glasses_outer_recall':float(base_outer[eroded].float().mean())}
    for name,mask in [('left',green&(xx<160)),('right',green&(xx>=160))]:
        metrics[name+'_lens_outer_recall']=float(final_outer[mask].float().mean())
        metrics['baseline_'+name+'_lens_outer_recall']=float(base_outer[mask].float().mean())
    save_image(torch.cat(list(result['render'][:,:3]),dim=2).cpu(),output_dir/f'real_{step}_render.png')
    save_image(pred[1:2].cpu(),output_dir/f'real_{step}_glasses_probability.png')
    tensor_to_rgba_image(result['uv'][0]).save(output_dir/f'real_{step}_uv.png')
    return metrics
