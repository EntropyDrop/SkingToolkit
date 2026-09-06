"""Train visible head-face identity while freezing the released semantic model."""
import argparse,hashlib,inspect,json,os,time,math,shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.head_semantics_data import JointHeadDataset
from SkingToolkit.dense_uv_parser.accessories import head_crop,head_bounds,restore_logits,HeadAccessoryHead
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.dense_uv_parser.train_v102 import segmentation_loss
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment
from SkingToolkit.renderer import DifferentiableRenderer


def render_batch(batch,renderer):
    images=[];labels=[]
    for view in ('front_left','back_left'):
        image,t=build_dense_parser_batch(batch['uv'].cuda(),renderer,view)
        valid=(t['foreground'][:,0]>.5)&(t['part']==0)
        images.append(image);labels.append((t['face']+1).masked_fill(~valid,0))
    return torch.stack(images,1).flatten(0,1),torch.stack(labels,1).flatten(0,1)


def features(model,images):
    crop=head_crop(images,images[:,3]>.5)
    semantic=model._runtime_semantic_features(crop)['raw_spatial']
    f=HeadAccessoryHead.forward(model.head_semantics_head,crop,semantic,return_features=True)
    return model.head_semantics_head.surface_features(f,crop)


@torch.no_grad()
def evaluate(model,renderer,dataset):
    model.eval();c=torch.zeros(7,7,device='cuda',dtype=torch.long)
    for b in DataLoader(dataset,batch_size=6,num_workers=2):
        images,truth=render_batch(b,renderer)
        pred=restore_logits(model.head_semantics_head.surface_classifier(features(model,images)),images.shape[-2:]).argmax(1)
        y0,y1,x0,x1=head_bounds(*truth.shape[-2:]);truth=truth[:,y0:y1,x0:x1];pred=pred[:,y0:y1,x0:x1]
        c+=torch.bincount((truth*7+pred).flatten(),minlength=49).reshape(7,7)
    return {'confusion':c.tolist(),'recall':(c.diag()/c.sum(1).clamp_min(1)).tolist(),'precision':(c.diag()/c.sum(0).clamp_min(1)).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=2400);o=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1022306);torch.backends.cudnn.benchmark=True
    root=Path(__file__).parent;out=o.output_dir;out.mkdir(parents=True,exist_ok=False)
    parent=torch.load(o.parent,map_location='cpu',weights_only=False);cfg={k:v for k,v in parent['model_config'].items() if k in inspect.signature(DenseUVParserNet).parameters};cfg['predict_head_surface']=True
    model=DenseUVParserNet(**cfg).cuda();missing,extra=model.load_state_dict(parent['model'],strict=False)
    if extra or any(not k.startswith('head_semantics_head.surface_classifier.') for k in missing):raise ValueError((missing,extra))
    model.requires_grad_(False);decoder=model.head_semantics_head.surface_classifier;decoder.requires_grad_(True)
    attach_semantic_runtime(model,'siglip2','google/siglip2-base-patch16-224','cuda',local_files_only=True);renderer=DifferentiableRenderer(parent['args']['mappings_dir']).cuda()
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text());train=JointHeadDataset(split['train'],32768,10223006);val=JointHeadDataset(split['validation'],128,20223006)
    opt=torch.optim.AdamW(decoder.parameters(),lr=5e-4,weight_decay=1e-4)
    manifest={'version':'v102','purpose':'visible head face identity, including secondary faces behind transparent outer planes','parent':str(o.parent),'parent_sha256':hashlib.sha256(o.parent.read_bytes()).hexdigest(),'frozen_tensors':len(parent['model']),'trainable_parameters':sum(x.numel() for x in decoder.parameters()),'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')},'validation':'128 reused train-disjoint source identities; procedural validation, not independent real holdout'}
    (out/'config.json').write_text(json.dumps(manifest,indent=2));snapshot=out/'source';snapshot.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    step=0;start=time.time();epoch=0
    while step<o.steps:
        train.epoch=epoch
        for b in DataLoader(train,batch_size=6,shuffle=True,num_workers=4):
            if step>=o.steps:break
            model.eval();decoder.train()
            with torch.no_grad():
                images,truth=render_batch(b,renderer)
                if step%2==0:images=appearance_augment(images,images[:,3:4]>.5,strength=1.4)
                with torch.autocast('cuda',dtype=torch.bfloat16):f=features(model,images)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=restore_logits(decoder(f),images.shape[-2:]);y0,y1,x0,x1=head_bounds(*truth.shape[-2:]);loss=segmentation_loss(logits[:,:,y0:y1,x0:x1],truth[:,y0:y1,x0:x1],[1,2,2,2,2,2,2])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite surface loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.);opt.step();step+=1
            for group in opt.param_groups:group['lr']=5e-4*(.1+.9*(1+math.cos(math.pi*step/o.steps))/2)
            if step%50==0:
                status={'state':'training','step':step,'steps':o.steps,'pid':os.getpid(),'elapsed_seconds':time.time()-start,'loss':float(loss)};(out/'status.json').write_text(json.dumps(status));print(status,flush=True)
            if step%600==0 or step==o.steps:
                report=evaluate(model,renderer,val);(out/f'evaluation_{step}.json').write_text(json.dumps(report,indent=2))
                for name,value in parent['model'].items():
                    if not torch.equal(value,model.state_dict()[name].cpu()):raise RuntimeError('Frozen model changed: '+name)
                torch.save({**{k:v for k,v in parent.items() if k!='model'},'model':model.state_dict(),'model_config':cfg,'surface_step':step,'surface_manifest':manifest,'surface_metrics':report},out/f'step_{step}.pt')
                print('checkpoint',step,report['recall'],flush=True)
        epoch+=1
    (out/'status.json').write_text(json.dumps({'state':'complete','step':step,'elapsed_seconds':time.time()-start,'parent_tensors_unchanged':True}))

if __name__=='__main__':main()
