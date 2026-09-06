"""Adapt an independent headwear decoder, preserving the released parser."""
import argparse, hashlib, inspect, json, math, shutil, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.headwear_data import HeadwearDataset, render_headwear
from SkingToolkit.dense_uv_parser.headwear import HEADWEAR_CLASSES
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment, write_json
from SkingToolkit.renderer import DifferentiableRenderer


def headwear_loss(logits, truth):
    probability = logits.float().softmax(1)
    ce = F.cross_entropy(logits.float(), truth, weight=logits.new_tensor([1,2,4,2,4,3,4]).float())
    onehot = F.one_hot(truth,7).permute(0,3,1,2).float()
    dice = 1-(2*(probability*onehot).sum((2,3))+1)/(probability.sum((2,3))+onehot.sum((2,3))+1)
    # One authored component has one layer across the two views. Penalize the
    # opposite layer on every member, including low confidence boundaries.
    consistency = logits.sum()*0
    for ids in ((1,3),(2,4)):
        for start in range(0,truth.shape[0],2):
            mask=(truth[start:start+2]==ids[0])|(truth[start:start+2]==ids[1])
            if not mask.any():continue
            correct=int((truth[start:start+2][mask]==ids[1]).all())
            pair=probability[start:start+2][:,ids]
            logp=(pair[:,correct]/pair.sum(1).clamp_min(1e-6)).clamp_min(1e-6).log()
            consistency += -logp[mask].mean()/max(1,truth.shape[0])
    return ce+dice[:,1:].mean()+.5*consistency


@torch.no_grad()
def evaluate(model,renderer,dataset):
    confusion=torch.zeros(7,7,device='cuda',dtype=torch.long);model.eval()
    for batch in DataLoader(dataset,batch_size=8):
        images,truth=render_headwear(batch['uv'].cuda(),batch['components'].cuda(),renderer,['front_left','back_left'])
        logits=model.predict_headwear_components(images,images[:,3]>.5)
        y0,y1,x0,x1=head_bounds(*truth.shape[-2:])
        truth=truth[:,y0:y1,x0:x1];pred=logits.argmax(1)[:,y0:y1,x0:x1]
        confusion+=torch.bincount((truth*7+pred).flatten(),minlength=49).reshape(7,7)
    tp=confusion.diag()
    return {'confusion':confusion.cpu().tolist(),'precision':(tp/confusion.sum(0).clamp_min(1)).cpu().tolist(),'recall':(tp/confusion.sum(1).clamp_min(1)).cpu().tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=1800)
    p.add_argument('--eval-every',type=int,default=600);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--resume',type=Path)
    opt=p.parse_args();out=opt.output_dir;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parent;origin=root/'runs/v101_semantic_release_gated_20260906/parser.pt'
    torch.manual_seed(910107);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.set_float32_matmul_precision('high')
    ckpt=torch.load(origin,map_location='cpu',weights_only=False)
    kwargs={k:v for k,v in ckpt['model_config'].items() if k in inspect.signature(DenseUVParserNet).parameters}
    kwargs['predict_headwear']=True;model=DenseUVParserNet(**kwargs).cuda()
    missing,extra=model.load_state_dict(ckpt['model'],strict=False)
    if extra or any(not x.startswith('headwear_head.') for x in missing):raise ValueError((missing,extra))
    warm={k.removeprefix('accessory_head.'):v for k,v in ckpt['model'].items() if k.startswith('accessory_head.') and not k.startswith(('accessory_head.components.','accessory_head.classifier.'))}
    model.headwear_head.load_state_dict(warm,strict=False)
    with torch.no_grad():
        model.headwear_head.classifier.weight[:6].copy_(ckpt['model']['accessory_head.components.weight'])
        model.headwear_head.classifier.bias[:6].copy_(ckpt['model']['accessory_head.components.bias'])
        model.headwear_head.classifier.weight[6].zero_();model.headwear_head.classifier.bias[6]=-4
    if opt.resume:model.load_state_dict(torch.load(opt.resume,map_location='cuda',weights_only=False)['model'])
    model.requires_grad_(False);model.headwear_head.requires_grad_(True)
    attach_semantic_runtime(model,'siglip2','google/siglip2-base-patch16-224','cuda',local_files_only=True)
    renderer=DifferentiableRenderer(ckpt['args']['mappings_dir']).cuda()
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
    train=HeadwearDataset(split['train'],32768,7101000);val=HeadwearDataset(split['validation'],64,69101000)
    assert not set(split['train'])&(set(split['validation'])|set(split['test']))
    manifest={'classes':HEADWEAR_CLASSES,'base_checkpoint':str(origin),'base_sha256':hashlib.sha256(origin.read_bytes()).hexdigest(),
              'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()},
              'training':'Authored whole components and visibility labels; all released parameters frozen; real examples are development only.',
              'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}}
    write_json(out/'config.json',manifest);snapshot=out/'source';snapshot.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    loader=DataLoader(train,batch_size=opt.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    optimizer=torch.optim.AdamW(model.headwear_head.parameters(),lr=opt.lr,weight_decay=1e-4)
    start=time.time();step=0;epoch=0
    try:
        while step<opt.steps:
            train.epoch=epoch
            for batch in loader:
                if step>=opt.steps:break
                model.eval();model.headwear_head.train()
                with torch.no_grad():images,truth=render_headwear(batch['uv'].cuda(),batch['components'].cuda(),renderer,['front_left','back_left']);fg=images[:,3]>.5
                if step%2==0:images=appearance_augment(images,fg[:,None],strength=1.4)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model.predict_headwear_components(images,fg);y0,y1,x0,x1=head_bounds(*truth.shape[-2:])
                    loss=headwear_loss(logits[:,:,y0:y1,x0:x1],truth[:,y0:y1,x0:x1])
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.headwear_head.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
                optimizer.step();step+=1
                for group in optimizer.param_groups:group['lr']=opt.lr*(.1+.9*(1+math.cos(math.pi*step/opt.steps))/2)
                if step%25==0 or step==1:
                    status={'state':'training','step':step,'total_steps':opt.steps,'loss':float(loss.detach()),'elapsed_seconds':time.time()-start};write_json(out/'status.json',status);print(json.dumps(status),flush=True)
                if step%opt.eval_every==0 or step==opt.steps:
                    metrics=evaluate(model,renderer,val);write_json(out/f'evaluation_{step}.json',metrics)
                    for name,value in ckpt['model'].items():
                        if not torch.equal(model.state_dict()[name].cpu(),value):raise RuntimeError('Released tensor changed: '+name)
                    payload={**{k:v for k,v in ckpt.items() if k not in ('model','optimizer')},'model':model.state_dict(),'model_config':{**ckpt['model_config'],'predict_headwear':True},'step':step,'headwear_manifest':manifest,'headwear_metrics':metrics}
                    torch.save(payload,out/f'step_{step}.pt');print('evaluation='+json.dumps(metrics),flush=True)
            epoch+=1
        write_json(out/'status.json',{'state':'complete','step':step,'released_tensors_unchanged':True,'elapsed_seconds':time.time()-start})
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','step':step,'error':repr(error)});raise


if __name__=='__main__':main()
