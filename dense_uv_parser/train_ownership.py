"""Train head ownership independently, preserving every released v101 parameter."""
import argparse,hashlib,inspect,json,os,shutil,time,math
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.ownership_data import OwnershipDataset,render_ownership
from SkingToolkit.dense_uv_parser.ownership import OWNERSHIP_CLASSES
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline,cached_real_foreground
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment,write_json
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image
from SkingToolkit.renderer import DifferentiableRenderer


@torch.no_grad()
def evaluate(model,renderer,dataset):
    confusion=torch.zeros(5,5,device='cuda',dtype=torch.long);model.eval()
    for batch in DataLoader(dataset,batch_size=8):
        images,labels=render_ownership(batch['uv'].cuda(),batch['ownership'].cuda(),renderer,['front_left','back_left'])
        logits=model.predict_ownership(images,images[:,3]>.5);y0,y1,x0,x1=head_bounds(*labels.shape[-2:])
        truth=labels[:,y0:y1,x0:x1];pred=logits.argmax(1)[:,y0:y1,x0:x1]
        confusion+=torch.bincount((truth*5+pred).flatten(),minlength=25).reshape(5,5)
    tp=confusion.diag();precision=tp/confusion.sum(0).clamp_min(1);recall=tp/confusion.sum(1).clamp_min(1)
    return {'confusion':confusion.cpu().tolist(),'precision':precision.cpu().tolist(),'recall':recall.cpu().tolist()}


@torch.no_grad()
def real_review(model,renderer,config,out,step):
    cases=[('glasses','TWRLRRTHQP2UV368_edited_1f54edecd442'),('headphones','img27_610024a091fe'),('hat','PU418PLGERMFC487_edited_edada35d51d2')]
    root=Path(__file__).parent
    stats={}
    for name,folder in cases:
        src=root/'output_history/v101_retrained_20260906'/folder
        manifest=json.loads((src/'manifest.json').read_text());d=torch.load(src/'diagnostics.pt',weights_only=False)
        images=d['images'].cuda();fg=cached_real_foreground(images,manifest['input'],config)
        outputs={k:v.cuda() if torch.is_tensor(v) else v for k,v in d['outputs'].items()}
        outputs['head_ownership_logits']=model.predict_ownership(images,fg>=.5)
        r=run_pipeline(model,renderer,images,config,complete=True,outputs=outputs,foreground_probability=fg)
        p=out/f'real_{step}'/name;p.mkdir(parents=True,exist_ok=True)
        tensor_to_rgba_image(r['uv'][0]).save(p/'uv.png');save_image(torch.cat(list(r['render'][:,:3]),2),p/'render.png')
        palette=images.new_tensor([[.5,.5,.5],[1.,.65,.4],[.4,.3,.8],[.1,.9,.5],[1.,.2,.3]])
        cls=outputs['head_ownership_logits'].argmax(1)
        save_image(torch.cat(list(palette[cls].permute(0,3,1,2)),2),p/'ownership.png')
        route=r['details']['routing']
        torch.save({'uv':r['uv'].cpu(),'conditioning':r['conditioning'].cpu(),'outputs':{k:v.cpu() if torch.is_tensor(v) else v for k,v in outputs.items()},'routing':{k:v.cpu() if torch.is_tensor(v) else v for k,v in route.items()}},p/'diagnostics.pt')
        # Development probes are not labels or gradients; colour is only a diagnostic.
        uv=r['uv'][0];stats[name]={'inner_corrections':int(route['ownership_inner_supported'].sum()),'headphone_pixels':int(route['ownership_outer_supported'].sum())}
        if name=='glasses':stats[name]['forehead_outer_alpha']=float(uv[3,10,41])
        if name=='headphones':
            green=(uv[1]-torch.maximum(uv[0],uv[2])).clamp_min(0)
            mask=torch.zeros_like(green,dtype=torch.bool);mask[:8,8:16]=True;mask[8:16,:8]=True;mask[8:16,16:32]=True
            stats[name]['inner_nonface_green_sum']=float(green[mask].sum())
        if name=='hat':stats[name]['brim_ring']=int(sum((uv[3,11,x:x+8]>.5).sum() for x in [32,40,48,56]))
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--steps',type=int,default=1200);parser.add_argument('--eval-every',type=int,default=300);parser.add_argument('--batch-size',type=int,default=8);parser.add_argument('--lr',type=float,default=1e-4);parser.add_argument('--resume',type=Path)
    opt=parser.parse_args();out=opt.output_dir;out.mkdir(exist_ok=False,parents=True)
    root=Path(__file__).parent;origin=root/'runs/v101_release_20260906/parser.pt';torch.manual_seed(10106);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.set_float32_matmul_precision('high')
    ckpt=torch.load(origin,map_location='cpu',weights_only=False)
    kwargs={k:v for k,v in ckpt['model_config'].items() if k in inspect.signature(DenseUVParserNet).parameters};kwargs['predict_head_ownership']=True
    model=DenseUVParserNet(**kwargs).cuda();missing,extra=model.load_state_dict(ckpt['model'],strict=False)
    if extra or any(not x.startswith('ownership_head.') for x in missing):raise ValueError((missing,extra))
    warm={k.removeprefix('accessory_head.'):v for k,v in ckpt['model'].items() if k.startswith('accessory_head.') and not k.startswith(('accessory_head.components.','accessory_head.classifier.'))}
    model.ownership_head.load_state_dict(warm,strict=False)
    if opt.resume:model.load_state_dict(torch.load(opt.resume,map_location='cuda',weights_only=False)['model'])
    model.requires_grad_(False);model.ownership_head.requires_grad_(True)
    attach_semantic_runtime(model,'siglip2','google/siglip2-base-patch16-224','cuda',local_files_only=True)
    renderer=DifferentiableRenderer(ckpt['args']['mappings_dir']).cuda()
    config=json.loads((root/'runs/v101_release_20260906/pipeline.json').read_text());config['head_material_refine_mode']='isolated'
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
    train=OwnershipDataset(split['train'],32768,3101000);val=OwnershipDataset(split['validation'],64,29101000);test=OwnershipDataset(split['test'],128,39101000)
    assert not set(split['train'])&(set(split['validation'])|set(split['test']))
    manifest={'classes':OWNERSHIP_CLASSES,'base_checkpoint':str(origin),'base_sha256':hashlib.sha256(origin.read_bytes()).hexdigest(),'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()},'pipeline':config,'source_splits':{k:len(v) for k,v in split.items()},'real_scope':'development probes only; not training or independent test','source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in root.glob('*.py')}}
    write_json(out/'config.json',manifest);write_json(out/'pipeline.json',config)
    snapshot=out/'source';snapshot.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    loader=DataLoader(train,batch_size=opt.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    optimizer=torch.optim.AdamW(model.ownership_head.parameters(),lr=opt.lr,weight_decay=1e-4);start=time.time();step=0;epoch=0
    try:
        while step<opt.steps:
            train.epoch=epoch
            for batch in loader:
                if step>=opt.steps:break
                model.eval();model.ownership_head.train()
                with torch.no_grad():images,target=render_ownership(batch['uv'].cuda(),batch['ownership'].cuda(),renderer,['front_left','back_left']);fg=images[:,3]>.5
                if step%2==0:images=appearance_augment(images,fg[:,None],strength=1.4)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model.predict_ownership(images,fg);y0,y1,x0,x1=head_bounds(*target.shape[-2:]);logits=logits[:,:,y0:y1,x0:x1];truth=target[:,y0:y1,x0:x1]
                    ce=F.cross_entropy(logits.float(),truth,weight=images.new_tensor([1.,2.,2.,3.,4.]))
                    probability=logits.float().softmax(1);onehot=F.one_hot(truth,5).permute(0,3,1,2).float()
                    dice=1-(2*(probability*onehot).sum((2,3))+1)/(probability.sum((2,3))+onehot.sum((2,3))+1)
                    loss=ce+dice[:,1:].mean()
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.ownership_head.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
                optimizer.step();step+=1
                lr=opt.lr*(.1+.9*(1+math.cos(math.pi*step/opt.steps))/2)
                for g in optimizer.param_groups:g['lr']=lr
                if step%25==0 or step==1:
                    status={'state':'training','step':step,'total_steps':opt.steps,'loss':float(loss.detach()),'elapsed_seconds':time.time()-start};write_json(out/'status.json',status);print(json.dumps(status),flush=True)
                if step%opt.eval_every==0 or step==opt.steps:
                    metrics=evaluate(model,renderer,val);real=real_review(model,renderer,config,out,step)
                    record={'step':step,'validation':metrics,'real':real};write_json(out/f'evaluation_{step}.json',record)
                    for name,value in ckpt['model'].items():
                        if not torch.equal(model.state_dict()[name].cpu(),value):raise RuntimeError('Released tensor changed: '+name)
                    payload={**{k:v for k,v in ckpt.items() if k not in ('model','optimizer')},'model':model.state_dict(),'model_config':{**ckpt['model_config'],'predict_head_ownership':True},'step':step,'ownership_manifest':manifest,'ownership_metrics':record}
                    torch.save(payload,out/f'step_{step}.pt');print('evaluation='+json.dumps(record),flush=True)
            epoch+=1
        write_json(out/'heldout.json',evaluate(model,renderer,test))
        write_json(out/'status.json',{'state':'complete','step':step,'released_v101_tensors_unchanged':True,'elapsed_seconds':time.time()-start})
    except BaseException as error:
        write_json(out/'status.json',{'state':'failed','step':step,'error':repr(error)});raise


if __name__=='__main__':main()
