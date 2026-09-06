"""Supervised Minecraft foreground decoder adaptation with independent identity splits."""
import argparse,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageFilter
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from SkingToolkit.dense_uv_parser.matting_data import MattingDataset
from SkingToolkit.dense_uv_parser.foreground_model import load_foreground_model,foreground_logits,sha


def write_json(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(tmp,p)


def boundary(mask,width=2):
    dil=F.max_pool2d(mask.float(),width*2+1,1,width)
    ero=-F.max_pool2d(-mask.float(),width*2+1,1,width)
    return (dil-ero)>.01


def alpha_loss(logits,target):
    p=logits.float().sigmoid();edge=boundary(target)
    weight=1+4*edge.float()
    bce=(F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none')*weight).sum()/weight.sum()
    dice=1-(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1)
    edge_error=(abs(p-target)*edge).sum()/edge.sum().clamp_min(1)
    grad=sum(abs(torch.diff(p,dim=d)-torch.diff(target,dim=d)).mean() for d in (2,3))
    # Mine the worst background pixels, including wholly empty crops.
    # Boundary-only improvement must not hide false foreground islands.
    bg_loss=F.softplus(logits.float())*(target<.01).float()
    hard_bg=bg_loss.flatten(1).topk(max(1,bg_loss[0].numel()//100),dim=1).values.mean()
    return bce+.5*dice.mean()+2*edge_error+.2*grad+.5*hard_bg


@torch.no_grad()
def evaluate(model,manifest,split,count,size,out,tag,real=True):
    model.eval();dataset=MattingDataset(manifest,split,size=size,limit=count)
    loader=DataLoader(dataset,batch_size=2,num_workers=4)
    totals={k:0 for k in ('tp','fp','fn','boundary_abs','boundary_n','abs','n','edge_tp_p','edge_tp_g','edge_p','edge_g')}
    by_background={}
    for bi,batch in enumerate(loader):
        x,a=batch['image'].cuda(),batch['alpha'].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):p=foreground_logits(model,x).float().sigmoid()
        pred=p>=.5;gt=a>=.5
        totals['tp']+=int((pred&gt).sum());totals['fp']+=int((pred&~gt).sum());totals['fn']+=int((~pred&gt).sum())
        edge=boundary(a)
        totals['boundary_abs']+=float((abs(p-a)*edge).sum());totals['boundary_n']+=int(edge.sum())
        totals['abs']+=float(abs(p-a).sum());totals['n']+=p.numel()
        ep=boundary(pred,1);eg=boundary(gt,1)
        totals['edge_tp_p']+=int((ep&(F.max_pool2d(eg.float(),5,1,2)>.5)).sum())
        totals['edge_tp_g']+=int((eg&(F.max_pool2d(ep.float(),5,1,2)>.5)).sum())
        totals['edge_p']+=int(ep.sum());totals['edge_g']+=int(eg.sum())
        for i,k in enumerate(batch['background_kind'].tolist()):
            row=by_background.setdefault(str(k),{'tp':0,'fp':0,'fn':0})
            row['tp']+=int((pred[i]&gt[i]).sum());row['fp']+=int((pred[i]&~gt[i]).sum());row['fn']+=int((~pred[i]&gt[i]).sum())
        if bi==0:
            save_image(torch.cat([x,torch.where(pred,x,.5),a.expand(-1,3,-1,-1),p.expand(-1,3,-1,-1)],dim=0).cpu(),out/f'{tag}_validation.png',nrow=2)
    precision=totals['edge_tp_p']/max(1,totals['edge_p']);recall=totals['edge_tp_g']/max(1,totals['edge_g'])
    result={**totals,'images':len(dataset),'iou':totals['tp']/max(1,totals['tp']+totals['fp']+totals['fn']),
            'boundary_f1_tolerance_2px':2*precision*recall/max(1e-8,precision+recall),
            'boundary_alpha_mae':totals['boundary_abs']/max(1,totals['boundary_n']),
            'alpha_mae':totals['abs']/max(1,totals['n']),
            'background_iou':{k:v['tp']/max(1,sum(v.values())) for k,v in by_background.items()}}
    if real:result['real_development']=evaluate_real(model,out,tag)
    return result


@torch.no_grad()
def evaluate_real(model,out,tag):
    root=Path(__file__).parent
    cases=json.loads((root/'v101_cases.json').read_text())
    targets=[x for x in cases if any(k in x for k in ('PU418PLGER','TWRLRRTH','img28'))]
    metrics={'role':'development checks, no gradient training on these images'}
    dest=out/f'real_{tag}';dest.mkdir(exist_ok=True)
    for source in targets:
        path=Path(source);rgb=np.array(Image.open(path).convert('RGB'))
        image=torch.from_numpy(rgb.copy()).permute(2,0,1).float().cuda()/255
        views=torch.stack(image.chunk(2,dim=2))
        resized=F.interpolate(views,(1024,1024),mode='bilinear',align_corners=False)
        with torch.autocast('cuda',dtype=torch.bfloat16):prob=foreground_logits(model,resized).float().sigmoid()
        native=F.interpolate(prob,(512,256),mode='nearest-exact')[:,0]
        original=F.interpolate(prob,views.shape[-2:],mode='bilinear',align_corners=False)[:,0]
        array=torch.cat(list(original),dim=1).cpu().numpy()
        Image.fromarray((array*255).astype('uint8')).save(dest/(path.stem+'_probability.png'))
        save_image(torch.where(native[:,None]>=.5,F.interpolate(views,(512,256),mode='nearest-exact'),.5).cpu(),dest/(path.stem+'_cutout.png'),nrow=2)
        if 'PU418' in path.name:
            mask_path=root/'runs/v101_hat_audit_20260906/hat_brim_development_mask.png'
            mask=torch.from_numpy(np.array(Image.open(mask_path).filter(ImageFilter.MinFilter(3)))>0).cuda()
            metrics['hat_brim_foreground_recall']=[float((v[mask]>=.5).float().mean()) for v in native]
        if 'TWRLR' in path.name:
            mask_path=root/'regression/TWRLRRTHQP2UV368_glasses.png'
            mask=torch.from_numpy(np.array(Image.open(mask_path).filter(ImageFilter.MinFilter(5)))>0).cuda()
            metrics['glasses_foreground_recall']=float((native[0][mask]>=.5).float().mean())
    return metrics


def main():
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=800);p.add_argument('--eval-every',type=int,default=200)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--train-size',type=int,default=512)
    p.add_argument('--eval-size',type=int,default=768);p.add_argument('--val-samples',type=int,default=96)
    p.add_argument('--lr',type=float,default=0.000005)
    opt=p.parse_args()
    if min(opt.steps,opt.eval_every,opt.batch_size,opt.val_samples)<=0:raise ValueError('Counts must be positive')
    opt.output_dir.mkdir(parents=True,exist_ok=False)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
    torch.manual_seed(106);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    model,_=load_foreground_model(opt.model_dir,opt.resume)
    for name,param in model.named_parameters():param.requires_grad_(not name.startswith('bb.'))
    trainable=[v for v in model.parameters() if v.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=opt.lr,weight_decay=1e-4)
    def backbone_hash():
        h=hashlib.sha256()
        for name,value in model.state_dict().items():
            if name.startswith('bb.'):h.update(name.encode());h.update(value.cpu().numpy().tobytes())
        return h.hexdigest()
    frozen_hash=backbone_hash()
    info={'version':'minecraft_foreground_v1','architecture':'BiRefNet pretrained frozen backbone; supervised decoder adaptation',
          'base_model_dir':str(opt.model_dir.resolve()),'base_sha256':sha(opt.model_dir/'model.safetensors'),
          'backbone_tensor_sha256':frozen_hash,'dataset_manifest_sha256':sha(opt.dataset),
          'dataset_manifest':str(opt.dataset.resolve()),'trainable_parameters':sum(x.numel() for x in trainable),
          'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()},
          'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
          'sources':{f.name:sha(f) for f in Path(__file__).parent.glob('*.py')},
          'supervision':'Renderer alpha from independent normalized skin identities; randomized backgrounds and premultiplied affine resampling. No real regression images used for gradients.',
          'inference_resolution':1024,'output':'Supervised alpha prediction; retain separate reliable RGB source mask'}
    write_json(opt.output_dir/'config.json',info)
    snapshot=opt.output_dir/'source';snapshot.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    step=0;best=float('inf');start=time.time()
    def save(name,metrics):
        decoder={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('bb.')}
        temp=opt.output_dir/(name+'.tmp')
        torch.save({'decoder':decoder,'optimizer':optimizer.state_dict(),'step':step,'metrics':metrics,'manifest':info},temp)
        os.replace(temp,opt.output_dir/name)
    def accepted(m,baseline):
        r=m['real_development']
        return (m['iou']>=max(.985,baseline['iou']-.002) and m['boundary_f1_tolerance_2px']>=baseline['boundary_f1_tolerance_2px']-.003
                and m['boundary_alpha_mae']<=baseline['boundary_alpha_mae']*.98
                and min(r['hat_brim_foreground_recall']+[r['glasses_foreground_recall']])>=.995)
    try:
        write_json(opt.output_dir/'status.json',{'state':'baseline_validation','pid':os.getpid(),'step':0})
        baseline=evaluate(model,opt.dataset,'validation',opt.val_samples,opt.eval_size,opt.output_dir,'baseline')
        write_json(opt.output_dir/'baseline.json',baseline);print('baseline='+json.dumps(baseline),flush=True)
        dataset=MattingDataset(opt.dataset,'train',size=opt.train_size)
        epoch=0
        while step<opt.steps:
            dataset.epoch=epoch
            loader=DataLoader(dataset,batch_size=opt.batch_size,shuffle=True,num_workers=4,pin_memory=True)
            for batch in loader:
                if step>=opt.steps:break
                model.eval() # preserves pretrained BN statistics and uses the inference decoder path; gradients remain enabled
                x,a=batch['image'].cuda(non_blocking=True),batch['alpha'].cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=alpha_loss(foreground_logits(model,x),a)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite foreground loss')
                loss.backward();gn=torch.nn.utils.clip_grad_norm_(trainable,1.)
                if not torch.isfinite(gn):raise RuntimeError('Non-finite foreground gradient')
                optimizer.step();step+=1
                rate=opt.lr*(.1+.9*(1+math.cos(math.pi*step/opt.steps))/2)
                for group in optimizer.param_groups:group['lr']=rate
                if step==1 or step%10==0:
                    state={'state':'training','pid':os.getpid(),'step':step,'total_steps':opt.steps,'loss':float(loss),'gradient_norm':float(gn),'elapsed_seconds':time.time()-start}
                    write_json(opt.output_dir/'status.json',state);print(json.dumps(state),flush=True)
                if step%opt.eval_every==0 or step==opt.steps:
                    metrics=evaluate(model,opt.dataset,'validation',opt.val_samples,opt.eval_size,opt.output_dir,str(step))
                    valid=accepted(metrics,baseline);score=metrics['boundary_alpha_mae']+5*(1-metrics['iou'])
                    save('latest.pt',metrics);promoted=valid and score<best
                    if promoted:best=score;save('best.pt',metrics)
                    record={'step':step,'accepted':valid,'promoted':promoted,'metrics':metrics}
                    write_json(opt.output_dir/'last_evaluation.json',record)
                    with (opt.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print('evaluation='+json.dumps(record),flush=True)
            epoch+=1
        if backbone_hash()!=frozen_hash:raise RuntimeError('Frozen BiRefNet backbone changed')
        selected=opt.output_dir/'best.pt'
        if selected.exists():
            model.load_state_dict(torch.load(selected,map_location='cuda',weights_only=False)['decoder'],strict=False)
            heldout=evaluate(model,opt.dataset,'test',128,opt.eval_size,opt.output_dir,'heldout',real=False)
            write_json(opt.output_dir/'heldout.json',heldout)
            del model
            torch.cuda.empty_cache()
            base_model,_=load_foreground_model(opt.model_dir)
            heldout_base=evaluate(base_model,opt.dataset,'test',128,opt.eval_size,opt.output_dir,'heldout_pretrained',real=False)
            write_json(opt.output_dir/'heldout_pretrained.json',heldout_base)
        write_json(opt.output_dir/'status.json',{'state':'complete','steps':step,'best_exists':selected.exists(),'elapsed_seconds':time.time()-start,'backbone_unchanged':True})
    except BaseException as error:
        write_json(opt.output_dir/'status.json',{'state':'failed','step':step,'error':repr(error)})
        raise


if __name__=='__main__':main()
