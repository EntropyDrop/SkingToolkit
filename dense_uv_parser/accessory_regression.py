"""Visually traced development regression, kept separate from training/test data."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
from PIL import ImageDraw,ImageFilter
import torch.nn.functional as F
from torchvision.utils import save_image
from SkingToolkit.dense_uv_parser.infer import load_view_images
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline,cached_real_foreground
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image


@torch.no_grad()
def evaluate_real_regression(model,renderer,config,output_dir,step):
    root=Path(__file__).parent
    metadata=json.loads((root/'v101_real_regression.json').read_text())
    image_path=Path(metadata['combined'])
    if hashlib.sha256(image_path.read_bytes()).hexdigest()!=metadata['source_sha256']:
        raise ValueError('Regression input changed; review the annotation before using it')
    images=load_view_images(SimpleNamespace(combined=image_path),config['routing']['views'],renderer).cuda()
    probability=cached_real_foreground(images,image_path,config)
    result=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=probability)
    baseline=run_pipeline(model,renderer,images,config,outputs={k:v for k,v in result['outputs'].items() if k not in ('accessory_logits','hat_component_logits')},foreground_probability=probability)
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
    if config.get("hat_development"):
        metrics.update(evaluate_hat_regression(model,renderer,config,output_dir,step))
    return metrics


@torch.no_grad()
def evaluate_hat_regression(model,renderer,config,output_dir,step):
    spec=config['hat_development'];source=Path(spec['input'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=spec['input_sha256']:raise ValueError('Hat input changed')
    images=load_view_images(SimpleNamespace(combined=source),config['routing']['views'],renderer).cuda()
    probability=cached_real_foreground(images,source,config)
    result=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=probability)
    mask=torch.from_numpy(np.array(Image.open(spec['brim_mask']).filter(ImageFilter.MinFilter(3)))>0).cuda()
    routing=result['details']['routing'];outer=routing['foreground']&(routing['layer']==1)
    # Explicit colour-band region, used only as a development render diagnostic.
    band=Image.new('L',(256,512));ImageDraw.Draw(band).polygon([(48,56),(107,80),(207,65),(207,79),(107,94),(48,69)],fill=255)
    band=torch.from_numpy(np.array(band.filter(ImageFilter.MaxFilter(9)))>0).cuda()
    def chroma(rgb):return (rgb[:,0]-rgb[:,1:3].amax(1)-.15).clamp_min(0)[:,band].sum(1)
    reference=chroma(images[:,:3]);observed=chroma(result['render'][:,:3])
    metrics={'hat_brim_outer_recall':[float(v[mask].float().mean()) for v in outer],
             'hat_band_chroma_retention':[float(x) for x in (observed/reference.clamp_min(1e-5))]}
    save_image(torch.cat(list(result['render'][:,:3]),2).cpu(),output_dir/f'hat_{step}_render.png')
    tensor_to_rgba_image(result['uv'][0]).save(output_dir/f'hat_{step}_uv.png')
    return metrics
