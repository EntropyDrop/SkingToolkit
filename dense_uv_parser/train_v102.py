"""Train only a new paired-view joint head from the frozen released v101 model."""
import argparse,hashlib,inspect,json,math,os,shutil,time,subprocess
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.head_semantics import CLASSES,PROJECTIONS,project_semantics
from SkingToolkit.dense_uv_parser.head_semantics_data import JointHeadDataset,render_joint_batch
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment,write_json
from SkingToolkit.renderer import DifferentiableRenderer


def segmentation_loss(logits,truth,weights):
    p=logits.float().softmax(1);onehot=F.one_hot(truth,logits.shape[1]).permute(0,3,1,2).float()
    ce=F.cross_entropy(logits.float(),truth,weight=logits.new_tensor(weights).float())
    dice=1-(2*(p*onehot).sum((2,3))+1)/(p.sum((2,3))+onehot.sum((2,3))+1)
    # Preserve object holes and facial boundaries; do not impose symmetry.
    boundary=sum(F.smooth_l1_loss(torch.diff(p,dim=axis),torch.diff(onehot,dim=axis)) for axis in (2,3))
    return ce+dice[:,1:].mean()+.2*boundary


def joint_loss(logits,presence,truth,modes):
    total=logits.sum()*0;denom=0
    for mode,task,weights in ((0,None,[1,2,2,4,2,4,3,4,2,4,2,4,3,4]),(1,'ownership',[1,2,2,3,4]),(2,'headwear',[1,2,4,2,4,3,4])):
        selected=modes==mode
        if not selected.any():continue
        prediction=logits[selected] if task is None else project_semantics(logits[selected],task)
        loss=segmentation_loss(prediction,truth[selected],weights)
        if mode in (0,2):
            target=truth[selected]
            hat=((target>=8)&(target<=12)) if mode==0 else ((target>=1)&(target<=5))
            crown=target==(13 if mode==0 else 6)
            gt=torch.stack([hat.flatten(1).any(1),crown.flatten(1).any(1)],1).float()
            loss+=.35*F.binary_cross_entropy_with_logits(presence[selected].float(),gt,weight=torch.where(gt>.5,1.,2.))
        count=int(selected.sum());total+=loss*count;denom+=count
    return total/max(1,denom)


def project_truth(truth,task):
    lut=torch.zeros(len(CLASSES),device=truth.device,dtype=torch.long)
    for index,ids in enumerate(PROJECTIONS[task]):lut[list(ids)]=index
    return lut[truth]


@torch.no_grad()
def evaluate(model,renderer,dataset,baseline=False):
    model.eval();confusion={name:torch.zeros(n,n,device='cuda',dtype=torch.long) for name,n in [('joint',14),('ownership',5),('headwear',7)]}
    features={k:{'correct':0,'pixels':0} for k in ('nose_inner','beard_layer','crown_cap_outer')};presence_counts={'tp':0,'fp':0,'tn':0,'fn':0}
    for batch in DataLoader(dataset,batch_size=8,num_workers=2):
        batch={k:v.cuda() for k,v in batch.items()};images,truth,modes,feature=render_joint_batch(batch,renderer,['front_left','back_left']);fg=images[:,3]>.5
        if baseline:
            own=model.predict_ownership(images,fg)
            hw,gate=model.predict_headwear_components(images,fg,return_presence=True)
            joint=None
        else:
            joint,gate=model.predict_joint_head_semantics(images,fg);own=project_semantics(joint,'ownership');hw=project_semantics(joint,'headwear')
        y0,y1,x0,x1=head_bounds(*truth.shape[-2:]);truth=truth[:,y0:y1,x0:x1]
        for name,prediction in [('joint',joint),('ownership',own),('headwear',hw)]:
            if prediction is None:continue
            pred=prediction.argmax(1)[:,y0:y1,x0:x1]
            gt=truth if name=='joint' else project_truth(truth,name)
            n=confusion[name].shape[0];confusion[name]+=torch.bincount((gt*n+pred).flatten(),minlength=n*n).reshape(n,n)
        owner=own.softmax(1).argmax(1);headwear=hw.softmax(1).argmax(1)
        for i,name in enumerate(features):
            mask=feature[:,i];gt_full=batch['labels'] # counts only actual visible authored pixels
            if i==0:correct=owner==1
            elif i==1:
                # This is a semantic inner-layer vote, not end-to-end UV accuracy.
                # Use the identical marginalized decision for both models.
                inner=(truth==3)
                full_inner=F.pad(inner,(x0,images.shape[-1]-x1,y0,images.shape[-2]-y1))
                correct=(owner==1)==full_inner
            else:correct=headwear==6
            features[name]['correct']+=int((correct&mask).sum());features[name]['pixels']+=int(mask.sum())
        gt=torch.stack([((truth>=8)&(truth<=12)).flatten(1).any(1),(truth==13).flatten(1).any(1)],1)
        # Report the exact min-over-two-views gate used by inference.
        pred=(gate.sigmoid().reshape(-1,2,2).amin(1)>=.95);gt=gt.reshape(-1,2,2).any(1)
        for key,m in [('tp',pred&gt),('fp',pred&~gt),('tn',~pred&~gt),('fn',~pred&gt)]:presence_counts[key]+=int(m.sum())
    report={'baseline':baseline,'presence':presence_counts,'features':features,'metrics':{}}
    for name,c in confusion.items():
        if baseline and name=='joint':continue
        tp=c.diag();report['metrics'][name]={'confusion':c.cpu().tolist(),'precision':(tp/c.sum(0).clamp_min(1)).cpu().tolist(),'recall':(tp/c.sum(1).clamp_min(1)).cpu().tolist()}
    for value in features.values():value['accuracy']=value['correct']/max(1,value['pixels'])
    return report


@torch.no_grad()
def evaluate_replay(model,renderer,paths,baseline=False):
    from SkingToolkit.dense_uv_parser.ownership_data import OwnershipDataset,render_ownership
    from SkingToolkit.dense_uv_parser.headwear_data import HeadwearPresenceDataset,render_headwear
    model.eval();result={}
    for task,dataset,render,count in [('ownership',OwnershipDataset(paths,32,51020906),render_ownership,5),('headwear',HeadwearPresenceDataset(paths,32,61020906),render_headwear,7)]:
        confusion=torch.zeros(count,count,device='cuda',dtype=torch.long)
        for batch in DataLoader(dataset,batch_size=8,num_workers=2):
            key='ownership' if task=='ownership' else 'components'
            images,truth=render(batch['uv'].cuda(),batch[key].cuda(),renderer,['front_left','back_left'])
            if baseline:
                logits=model.predict_ownership(images,images[:,3]>.5) if task=='ownership' else model.predict_headwear_components(images,images[:,3]>.5)
            else:
                joint,_=model.predict_joint_head_semantics(images,images[:,3]>.5);logits=project_semantics(joint,task)
            y0,y1,x0,x1=head_bounds(*truth.shape[-2:]);gt=truth[:,y0:y1,x0:x1];pred=logits.argmax(1)[:,y0:y1,x0:x1]
            confusion+=torch.bincount((gt*count+pred).flatten(),minlength=count*count).reshape(count,count)
        tp=confusion.diag();result[task]={'confusion':confusion.cpu().tolist(),'precision':(tp/confusion.sum(0).clamp_min(1)).cpu().tolist(),'recall':(tp/confusion.sum(1).clamp_min(1)).cpu().tolist()}
    return result


@torch.no_grad()
def real_review(model,renderer,pipeline,out,step):
    """Real examples are observation-only probes; no loss, labels or gradients."""
    from torchvision.utils import save_image
    from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline,cached_real_foreground
    from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image
    root=Path(__file__).parent;cases=[];stats={};model.eval()
    for c in json.loads((root/'regression/v102_development_cases.json').read_text()):
        cases.append((c['name'],Path(c['directory']),c['input']))
    for folder in sorted((root/'output_history/v101_retrained_20260906').iterdir()):
        if not (folder/'diagnostics.pt').is_file():continue
        manifest=json.loads((folder/'manifest.json').read_text());cases.append(('old_'+Path(manifest['input']).stem,folder,manifest['input']))
    for name,folder,source in cases:
        d=torch.load(folder/'diagnostics.pt',map_location='cpu',weights_only=False);images=d['images'].cuda()
        fg=d['foreground'].cuda() if name in ('crown','nose','beard') else cached_real_foreground(images,source,pipeline)
        # Old regression archives predate the phone/ownership release. Run the
        # full current model so frozen presence gates cannot go missing.
        result=run_pipeline(model,renderer,images,pipeline,complete=True,foreground_probability=fg)
        joint=result['outputs']['head_semantics_logits'];presence=result['outputs']['headwear_presence_logits']
        dest=out/f'real_{step}'/name;dest.mkdir(parents=True,exist_ok=True)
        uv=result['uv'];tensor_to_rgba_image(uv[0]).save(dest/'uv.png');save_image(torch.cat(list(result['render'][:,:3]),2),dest/'render.png')
        for layer in ('inner','outer'):
            skin=uv.clone();skin[:,:,16:]=0
            if layer=='inner':skin[:,:,:16,32:]=0
            else:skin[:,:,:16,:32]=0
            rendered=torch.cat([renderer.forward_view(skin,v) for v in pipeline['routing']['views']])
            save_image(torch.cat(list(rendered[:,:3]),2),dest/(layer+'.png'))
        row={'presence':presence.sigmoid().cpu().tolist(),'top_outer_texels':int((uv[0,3,:8,40:48]>.5).sum()),'inner_hair_veto_pixels':int(result['details']['routing'].get('joint_inner_hair_veto',torch.zeros(1)).sum()),'nose_alpha_43_13':float(uv[0,3,13,43]),'top_corner_alpha_45_6':float(uv[0,3,6,45])}
        torch.save({'uv':uv.cpu(),'conditioning':result['conditioning'].cpu(),'head_semantics_logits':joint.cpu(),'presence':presence.cpu()},dest/'diagnostics.pt')
        if name in ('crown','nose','beard'):
            row['body_exactly_equal_to_v101']=bool(torch.equal(uv.cpu()[:,:,16:],d['uv'][:,:,16:]))
        stats[name]=row
    write_json(out/f'real_{step}.json',stats)
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--steps',type=int,default=4800)
    parser.add_argument('--batch-size',type=int,default=6);parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--eval-every',type=int,default=600);parser.add_argument('--first-eval',type=int,default=200)
    parser.add_argument('--validation-count',type=int,default=96);parser.add_argument('--skip-real',action='store_true')
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--seed',type=int,default=1020906)
    opt=parser.parse_args();out=opt.output_dir;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parent;origin=root/'runs/v101_crown_geometry_release_20260906/parser.pt'
    expected='a8aa3d8cd51cc6fec28d7c525aa1b012976d7cb1206e8b00c707de76bfb205dc'
    if hashlib.sha256(origin.read_bytes()).hexdigest()!=expected:raise ValueError('v101 checkpoint hash mismatch')
    if opt.resume_checkpoint is not None:
        origin=opt.resume_checkpoint.resolve();expected=hashlib.sha256(origin.read_bytes()).hexdigest()
    torch.manual_seed(opt.seed);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;torch.set_float32_matmul_precision('high')
    ckpt=torch.load(origin,map_location='cpu',weights_only=False)
    cfg={k:v for k,v in ckpt['model_config'].items() if k in inspect.signature(DenseUVParserNet).parameters};cfg['predict_head_semantics']=True
    model=DenseUVParserNet(**cfg).cuda();missing,extra=model.load_state_dict(ckpt['model'],strict=False)
    if extra or any(not n.startswith('head_semantics_head.') for n in missing):raise RuntimeError((missing,extra))
    if not ckpt['model_config'].get('predict_head_semantics',False):
        warm={k.removeprefix('headwear_head.'):v for k,v in ckpt['model'].items() if k.startswith('headwear_head.') and not k.startswith('headwear_head.classifier.')}
        model.head_semantics_head.load_state_dict(warm,strict=False)
        with torch.no_grad():
            target=model.head_semantics_head.classifier
            for j,k in enumerate([0,None,None,None,None,None,None,None,1,2,3,4,5,6]):
                if k is not None:
                    target.weight[j].copy_(ckpt['model']['headwear_head.classifier.weight'][k]);target.bias[j].copy_(ckpt['model']['headwear_head.classifier.bias'][k])
            for j,k in [(1,1),(2,2),(3,1),(6,3),(7,4)]:
                target.weight[j].copy_(ckpt['model']['ownership_head.classifier.weight'][k]);target.bias[j].copy_(ckpt['model']['ownership_head.classifier.bias'][k])
    model.requires_grad_(False);model.head_semantics_head.requires_grad_(True)
    attach_semantic_runtime(model,'siglip2','google/siglip2-base-patch16-224','cuda',local_files_only=True)
    renderer=DifferentiableRenderer(ckpt['args']['mappings_dir']).cuda()
    pipeline=json.loads((root/'runs/v101_crown_geometry_release_20260906/pipeline.json').read_text())
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
    if set(split['train'])&(set(split['validation'])|set(split['test'])):raise ValueError('Source identity split overlap')
    real_paths={c['input'] for c in json.loads((root/'regression/v102_development_cases.json').read_text())}
    if real_paths&set(split['train']):raise ValueError('Real development input in training split')
    train=JointHeadDataset(split['train'],32768,opt.seed+20000000);val=JointHeadDataset(split['validation'],opt.validation_count,31020906,joint_only=True)
    optimizer=torch.optim.AdamW(model.head_semantics_head.parameters(),lr=opt.lr,weight_decay=1e-4)
    manifest={'version':'v102','state':'candidate_not_released','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root.parent,text=True).strip(),'parent':str(origin),'parent_sha256':expected,'classes':CLASSES,'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()},'frozen_tensors':sum(not n.startswith('head_semantics_head.') for n in ckpt['model']),'source_splits_sha256':hashlib.sha256((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_bytes()).hexdigest(),'source_split_counts':{k:len(v) for k,v in split.items()},'trainable_parameters':sum(p.numel() for p in model.head_semantics_head.parameters()),'data':'60% authored joint heads including aligned mixed beards and partial hair shells; 20% legacy ownership; 20% legacy headwear/rich hair negatives. No real development image used for gradients.','views':['front_left','back_left'],'validation':'Train-disjoint source identities, reused validation identities with new procedural seeds; not fresh real-world heldout evidence.','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}}
    write_json(out/'config.json',manifest);write_json(out/'pipeline.json',pipeline)
    snapshot=out/'source';snapshot.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    start=time.time();step=0
    def status(state,**extra):
        row={'state':state,'pid':os.getpid(),'step':step,'total_steps':opt.steps,'elapsed_seconds':time.time()-start,**extra};write_json(out/'status.json',row);print(json.dumps(row),flush=True)
    try:
        status('baseline_validation');baseline=evaluate(model,renderer,val,baseline=opt.resume_checkpoint is None);baseline['legacy_replay']=evaluate_replay(model,renderer,split['validation'],baseline=opt.resume_checkpoint is None);write_json(out/'baseline_validation.json',baseline)
        epoch=0
        while step<opt.steps:
            train.epoch=epoch
            loader=DataLoader(train,batch_size=opt.batch_size,shuffle=True,num_workers=4,pin_memory=True)
            for batch in loader:
                if step>=opt.steps:break
                model.eval();model.head_semantics_head.train()
                batch={k:v.cuda(non_blocking=True) for k,v in batch.items()}
                with torch.no_grad():images,truth,modes,_=render_joint_batch(batch,renderer,['front_left','back_left']);fg=images[:,3]>.5
                if step%2==0:images=appearance_augment(images,fg[:,None],strength=1.4)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,presence=model.predict_joint_head_semantics(images,fg);y0,y1,x0,x1=head_bounds(*truth.shape[-2:])
                    loss=joint_loss(logits[:,:,y0:y1,x0:x1],presence,truth[:,y0:y1,x0:x1],modes)
                if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.head_semantics_head.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError('Non-finite gradient')
                optimizer.step();step+=1
                for group in optimizer.param_groups:group['lr']=opt.lr*(.1+.9*(1+math.cos(math.pi*step/opt.steps))/2)
                if step==1 or step%25==0:status('training',loss=float(loss.detach()),gradient_norm=float(grad),peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                if step==opt.first_eval or step%opt.eval_every==0 or step==opt.steps:
                    status('validation');metrics=evaluate(model,renderer,val);metrics['legacy_replay']=evaluate_replay(model,renderer,split['validation']);write_json(out/f'evaluation_{step}.json',metrics)
                    for name,value in ckpt['model'].items():
                        if name.startswith('head_semantics_head.'):continue
                        if not torch.equal(value,model.state_dict()[name].cpu()):raise RuntimeError('Frozen v101 tensor changed: '+name)
                    payload={**{k:v for k,v in ckpt.items() if k not in ('model','optimizer')},'model':model.state_dict(),'model_config':{**ckpt['model_config'],'predict_head_semantics':True},'step':step,'v102_manifest':manifest,'v102_metrics':metrics,'inference_pipeline':pipeline}
                    temporary=out/'checkpoint.tmp';torch.save(payload,temporary);temporary.replace(out/f'step_{step}.pt')
                    torch.save({'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},out/'optimizer_latest.pt')
                    write_json(out/'latest.json',{'step':step,'checkpoint':str(out/f'step_{step}.pt'),'v101_tensors_unchanged':True})
                    if not opt.skip_real:status('real_regression');real_review(model,renderer,pipeline,out,step)
                    print('checkpoint='+str(out/f'step_{step}.pt'),flush=True)
            epoch+=1
        status('complete',v101_tensors_unchanged=True,release_promoted=False)
    except BaseException as e:status('failed',error=repr(e));raise


if __name__=='__main__':main()
