"""v101 object-supervised adaptation of v61 with end-to-end UV validation."""
import argparse
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.skin_dataset import SkinUVDataset, load_skin
from SkingToolkit.dense_uv_parser.accessories import accessory_loss, head_bounds
from SkingToolkit.dense_uv_parser.accessory_data import AccessoryDataset, render_accessories
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline, load_pipeline
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment, write_json
from SkingToolkit.dense_uv_parser.utils import build_static_surface_routing, build_dense_parser_batch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
from SkingToolkit.renderer import DifferentiableRenderer


def uv_structure_loss(logits, labels, renderer, views):
    """Object-labelled UV graph edges, including physical cube seams."""
    p = logits.float().softmax(1)[:,1:]
    g = F.one_hot(labels,4).permute(0,3,1,2)[:,1:].float()
    groups = logits.shape[0] // len(views)
    pred = p.new_zeros(groups,3,4096); truth = torch.zeros_like(pred)
    count = p.new_zeros(groups,1,4096)
    for vi,view in enumerate(views):
        static = build_static_surface_routing(renderer,view,logits.device)
        valid = static['masks'][1] & (static['part'][1]==0)
        index = static['flat_uv'][1][valid]
        pred.index_add_(2,index,p[vi::len(views),:,valid])
        truth.index_add_(2,index,g[vi::len(views),:,valid])
        count.index_add_(2,index,torch.ones(groups,1,index.numel(),device=p.device))
    pred, truth = pred/count.clamp_min(1), truth/count.clamp_min(1)
    topology = build_simple_uv_topology()
    edges = topology.outer_edge_index.to(p.device)
    head = topology.part.flatten().to(p.device)==0
    a,b = edges[:,head[edges[0]] & head[edges[1]]]
    visible = (count[:,:,a]>0) & (count[:,:,b]>0)
    difference = (pred[:,:,a]-pred[:,:,b])-(truth[:,:,a]-truth[:,:,b])
    edge_loss = (difference.square()*visible).sum()/visible.expand_as(difference).sum().clamp_min(1)
    support = count>0
    uv_loss = ((pred-truth).square()*support).sum()/support.expand_as(pred).sum().clamp_min(1)
    faces=topology.face.flatten().to(p.device)
    seam = faces[a] != faces[b]
    seam_valid=visible[:,:,seam]
    seam_loss=(difference[:,:,seam].square()*seam_valid).sum()/seam_valid.expand_as(difference[:,:,seam]).sum().clamp_min(1)
    return uv_loss + edge_loss + seam_loss


@torch.no_grad()
def evaluate(model, renderer, dataset, config, output_dir, step, clean_paths, max_batches=None):
    model.eval()
    views = config['routing']['views']
    totals = {k:0 for k in ('tp','fp','fn','complete_objects','objects','uv_tp','uv_fn','head_rgb_abs','head_rgb_n')}
    per_class = {str(k):{'tp':0,'fp':0,'fn':0} for k in (1,2,3)}
    visible_uv = torch.zeros(4096,dtype=torch.bool,device='cuda')
    for view in views:
        static = build_static_surface_routing(renderer,view,'cuda')
        visible_uv[static['flat_uv'][1][static['masks'][1]]] = True
    loader = DataLoader(dataset,batch_size=4,num_workers=0)
    stress_rng=torch.Generator(device='cuda').manual_seed(10199)
    for bi,batch in enumerate(loader):
        if max_batches is not None and bi>=max_batches: break
        uv,objects = batch['uv'].cuda(),batch['objects'].cuda()
        images,labels,fg = render_accessories(uv,objects,renderer,views)
        if bi%2:
            images=appearance_augment(images,fg[:,None],strength=1.25,generator=stress_rng)
        result = run_pipeline(model,renderer,images,config,complete=True)
        logits = result['outputs']['accessory_logits']
        cls = logits.argmax(1); confidence=logits.softmax(1).amax(1)
        pred = result['details']['routing']['accessory_supported']
        gt = labels>0
        for k in (1,2,3):
            pp,gg = pred&(cls==k),labels==k
            for name,value in [('tp',pp&gg),('fp',pp&~gg),('fn',~pp&gg)]: per_class[str(k)][name]+=int(value.sum())
            for n in range(labels.shape[0]):
                if gg[n].any():
                    totals['objects']+=1
                    totals['complete_objects']+=int((pp[n]&gg[n]).sum()/gg[n].sum() >= .95)
        totals['tp']+=int((pred&gt).sum());totals['fp']+=int((pred&~gt).sum());totals['fn']+=int((~pred&gt).sum())
        uv_gt=(objects>0)&visible_uv.reshape(64,64)
        known=result['conditioning'][:,9]>.5
        totals['uv_tp']+=int((known&uv_gt).sum());totals['uv_fn']+=int((~known&uv_gt).sum())
        headmask=torch.zeros_like(fg); y0,y1,x0,x1=head_bounds(*fg.shape[-2:]);headmask[:,y0:y1,x0:x1]=fg[:,y0:y1,x0:x1]
        totals['head_rgb_abs']+=float((abs(result['render'][:,:3]-images[:,:3])*headmask[:,None]).sum())
        totals['head_rgb_n']+=int(headmask.sum())*3
        if bi==0:
            save_image(images[:8,:3].cpu(),output_dir/f'validation_{step}_input.png',nrow=4)
            save_image(result['render'][:8,:3].cpu(),output_dir/f'validation_{step}_render.png',nrow=4)
            palette=images.new_tensor([[.5,.5,.5],[.1,.9,.5],[.95,.6,.1],[.7,.3,.9]])
            save_image(palette[cls[:8]].permute(0,3,1,2).cpu(),output_dir/f'validation_{step}_objects.png',nrow=4)
    # Stored UV layers are a conservative regression guard, not semantic class GT.
    from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
    inner_pixels=added_wrong=0
    for start in range(0,len(clean_paths),4):
        uv=torch.stack([load_skin(path) for path in clean_paths[start:start+4]]).cuda()
        images=torch.stack([renderer.forward_view(uv,v) for v in views],1).flatten(0,1)
        result=run_pipeline(model,renderer,images,config)
        base_outputs={k:v for k,v in result['outputs'].items() if k!='accessory_logits'}
        baseline=run_pipeline(model,renderer,images,config,outputs=base_outputs)
        topology=build_simple_uv_topology()
        head_outer=(topology.valid&(topology.layer==1)&(topology.part==0)).cuda()
        negative=(uv[:,3]<.5)&head_outer&visible_uv.reshape(64,64)
        new=(result['conditioning'][:,9]>.5)&~(baseline['conditioning'][:,9]>.5)
        inner_pixels+=int(negative.sum());added_wrong+=int((new&negative).sum())
    metrics={**totals,'classes':per_class,'validation_inputs':'Alternating canonical and appearance-stressed batches; full production pipeline',
        'object_precision':totals['tp']/max(1,totals['tp']+totals['fp']),
        'object_recall':totals['tp']/max(1,totals['tp']+totals['fn']),
        'object_iou':totals['tp']/max(1,totals['tp']+totals['fp']+totals['fn']),
        'complete_object_rate':totals['complete_objects']/max(1,totals['objects']),
        'hard_uv_object_recall':totals['uv_tp']/max(1,totals['uv_tp']+totals['uv_fn']),
        'final_head_rgb_mae':totals['head_rgb_abs']/max(1,totals['head_rgb_n']),
        'clean_added_outer_false_positive_rate':added_wrong/max(1,inner_pixels),
        'clean_negative_uv_texels':inner_pixels,'clean_added_wrong_outer':added_wrong}
    from SkingToolkit.dense_uv_parser.accessory_regression import evaluate_real_regression
    metrics['real_development'] = evaluate_real_regression(model,renderer,config,output_dir,step)
    return metrics


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--eval-every',type=int,default=1000)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--val-samples',type=int,default=64)
    parser.add_argument('--clean-samples',type=int,default=32)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--evaluate-only',action='store_true')
    parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--pipeline',type=Path)
    parser.add_argument('--replay-weight',type=float,default=.25)
    parser.add_argument('--require-approved-start',action='store_true')
    opt=parser.parse_args()
    if min(opt.steps,opt.eval_every,opt.batch_size,opt.val_samples,opt.clean_samples)<=0 or opt.lr<=0 or opt.replay_weight<0:
        parser.error('Counts and learning rate must be positive; replay weight nonnegative')
    opt.output_dir.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(101);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    torch.set_float32_matmul_precision('high')
    config=load_pipeline(opt.pipeline);views=config['routing']['views']
    origin=root/'dense_uv_parser/runs/dense_uv_parser_v61/best.pt'
    ckpt=torch.load(origin,map_location='cpu',weights_only=False)
    kwargs={k:v for k,v in ckpt['model_config'].items() if k in inspect.signature(DenseUVParserNet).parameters}
    kwargs['predict_head_accessories']=True
    model=DenseUVParserNet(**kwargs).cuda()
    missing,extra=model.load_state_dict(ckpt['model'],strict=False)
    if extra or any(not k.startswith('accessory_head.') for k in missing): raise RuntimeError((missing,extra))
    if opt.resume:
        model.load_state_dict(torch.load(opt.resume,map_location='cuda',weights_only=False)['model'],strict=True)
    for name,value in ckpt['model'].items():
        if not torch.equal(model.state_dict()[name].cpu(),value):
            raise ValueError('Warm start changed frozen v61 weights: '+name)
    model.requires_grad_(False);model.accessory_head.requires_grad_(True)
    attach_semantic_runtime(model,'siglip2','google/siglip2-base-patch16-224','cuda',local_files_only=True)
    mappings_dir=str(root.parent/'github/differentiable_minecraft_renderer/mappings_256x512')
    renderer=DifferentiableRenderer(mappings_dir).cuda()
    paths=SkinUVDataset(root/'skins',max_samples=180000).skin_paths
    split=json.loads((root/'dense_uv_parser/runs/dense_uv_parser_v100/config.json').read_text())['experiment']
    train_paths=[paths[i] for i in split['train_indices']]
    val_paths=[paths[i] for i in split['val_indices']]
    test_paths=[paths[i] for i in split['test_indices']]
    assert not (set(train_paths)&(set(val_paths)|set(test_paths)))
    assert not (set(val_paths)&set(test_paths))
    source_splits={name:[str(p) for p in items] for name,items in
                   [('train',train_paths),('validation',val_paths),('test',test_paths)]}
    split_bytes=json.dumps(source_splits,sort_keys=True).encode()
    (opt.output_dir/'source_splits.json').write_bytes(split_bytes)
    train=AccessoryDataset(train_paths,32768,101000)
    val=AccessoryDataset(val_paths,opt.val_samples,9101000)
    test=AccessoryDataset(test_paths,opt.val_samples,19101000)
    loader=DataLoader(train,batch_size=opt.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    optimizer=torch.optim.AdamW(model.accessory_head.parameters(),lr=opt.lr,weight_decay=1e-4)
    manifest={'version':'v101','base_checkpoint':str(origin),'base_sha256':hashlib.sha256(origin.read_bytes()).hexdigest(),
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'options':{k:str(v) if isinstance(v,Path) else v for k,v in vars(opt).items()},
        'source_splits_sha256':hashlib.sha256(split_bytes).hexdigest(),
        'train_source_count':len(train_paths),'val_source_count':len(val_paths),'test_source_count':len(test_paths),
        'warm_start_sha256':hashlib.sha256(opt.resume.read_bytes()).hexdigest() if opt.resume else None,
        'training':'Only accessory branch is trainable; v61 body and original route weights are frozen. Warm starts use a fresh optimizer.',
        'labels':'Authored procedural object identities. Optional stored-inner replay is layer evidence, NOT semantic truth; it can contradict painted-on hats/hair. Set replay-weight=0 for semantic experiments.',
        'real_sample_scope':'TWRLRRTHQP2UV368 is a development regression image, never an independent test.',
        'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')},
        'pipeline':config,'classes':['none','glasses','hat','outer_hair']}
    write_json(opt.output_dir/'config.json',manifest)
    snapshot=opt.output_dir/'source';snapshot.mkdir()
    for f in Path(__file__).parent.glob('*.py'): shutil.copy2(f,snapshot/f.name)
    (snapshot/'v101_pipeline.json').write_text(json.dumps(config,indent=2))
    shutil.copy2(Path(__file__).with_name('v101_real_regression.json'),snapshot/'v101_real_regression.json')
    shutil.copytree(Path(__file__).with_name('regression'),snapshot/'regression')
    start=time.time();best=-float('inf');step=0;epoch=0
    def save(name,metrics):
        args={**ckpt['args'],'siglip_local_files_only':True,'outer_uv_min_source_pixels':30,'mappings_dir':mappings_dir,'views':views}
        config_model={**ckpt['model_config'],'predict_head_accessories':True,'accessory_route_threshold':config['accessory_route_threshold']}
        payload={'model':model.state_dict(),'model_config':config_model,'args':args,'epoch':epoch,'step':step,
                 'metrics':{'val':metrics},'v101_manifest':manifest,'optimizer':optimizer.state_dict()}
        temp=opt.output_dir/(name+'.tmp');torch.save(payload,temp);os.replace(temp,opt.output_dir/name)
    def admission(metrics):
        real=metrics['real_development']
        return (metrics['object_precision']>=.90 and metrics['object_recall']>=.75
                and metrics['complete_object_rate']>=.25
                and metrics['clean_added_outer_false_positive_rate']<=.03
                and real['glasses_mask_precision']>=.85
                and real['whole_glasses_outer_recall']>=max(.90,real['baseline_whole_glasses_outer_recall']+.15))
    try:
        if opt.evaluate_only:
            metrics=evaluate(model,renderer,val,config,opt.output_dir,'calibrated',val_paths[:opt.clean_samples])
            accepted=admission(metrics)
            save('latest.pt',metrics)
            if accepted:save('best.pt',metrics)
            write_json(opt.output_dir/'last_evaluation.json',{'step':0,'metrics':metrics,'accepted':accepted})
            write_json(opt.output_dir/'status.json',{'state':'evaluated','accepted':accepted})
            print(json.dumps(metrics),flush=True)
            return
        if opt.require_approved_start:
            initial=evaluate(model,renderer,val,config,opt.output_dir,'initial',val_paths[:opt.clean_samples])
            write_json(opt.output_dir/'initial_evaluation.json',initial)
            if not admission(initial):raise RuntimeError('Warm start failed release validation')
            best=.4*initial['object_iou']+.3*initial['complete_object_rate']+.3*initial['hard_uv_object_recall']
            save('best.pt',initial)
            print('approved_start='+json.dumps(initial),flush=True)
        while step<opt.steps:
            train.epoch=epoch
            for batch in loader:
                if step>=opt.steps: break
                model.eval();model.accessory_head.train()
                images,labels,fg=render_accessories(batch['uv'].cuda(non_blocking=True),batch['objects'].cuda(non_blocking=True),renderer,views)
                if step%2==0: images=appearance_augment(images,fg[:,None],strength=1.25)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model.predict_accessories(images,fg)
                    valid=torch.zeros_like(fg);y0,y1,x0,x1=head_bounds(*fg.shape[-2:]);valid[:,y0:y1,x0:x1]=True
                    loss=accessory_loss(logits,labels,valid)+.25*uv_structure_loss(logits,labels,renderer,views)
                    if step%2==0 and opt.replay_weight>0:
                        # Only verified INNER head pixels are negative layer evidence.
                        # Unannotated outer texels are ignored, not assigned object IDs.
                        replay_uv=torch.stack([load_skin(train_paths[i]) for i in torch.randint(len(train_paths),(2,)).tolist()]).cuda()
                        replay_images=[];replay_fg=[];replay_valid=[]
                        for view in views:
                            im,t=build_dense_parser_batch(replay_uv,renderer,view)
                            replay_images.append(im);replay_fg.append(t['foreground'][:,0])
                            replay_valid.append((t['layer']==0)&(t['part']==0)&(t['foreground'][:,0]>.5))
                        ri=torch.stack(replay_images,1).flatten(0,1)
                        rf=torch.stack(replay_fg,1).flatten(0,1)
                        rv=torch.stack(replay_valid,1).flatten(0,1)
                        rl=model.predict_accessories(ri,rf)
                        if rv.any():
                            loss=loss+opt.replay_weight*(-F.log_softmax(rl.float(),1)[:,0][rv]).mean()
                if not torch.isfinite(loss): raise RuntimeError('Non-finite loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.accessory_head.parameters(),1.)
                if not torch.isfinite(grad): raise RuntimeError('Non-finite gradient')
                optimizer.step();step+=1
                lr=opt.lr*(.1+.9*(1+math.cos(math.pi*step/opt.steps))/2)
                for group in optimizer.param_groups: group['lr']=lr
                if step%25==0 or step==1:
                    status={'state':'training','pid':os.getpid(),'step':step,'total_steps':opt.steps,'loss':float(loss.detach()),
                            'gradient_norm':float(grad),'lr':lr,'elapsed_seconds':time.time()-start}
                    write_json(opt.output_dir/'status.json',status);print(json.dumps(status),flush=True)
                if step%opt.eval_every==0 or step==opt.steps:
                    metrics=evaluate(model,renderer,val,config,opt.output_dir,step,val_paths[:opt.clean_samples])
                    score=.4*metrics['object_iou']+.3*metrics['complete_object_rate']+.3*metrics['hard_uv_object_recall']
                    accepted=admission(metrics)
                    save('latest.pt',metrics)
                    promoted=accepted and score>best
                    if promoted: best=score;save('best.pt',metrics)
                    record={'step':step,'metrics':metrics,'score':score,'accepted':accepted,'promoted':promoted}
                    write_json(opt.output_dir/'last_evaluation.json',record)
                    with (opt.output_dir/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print('evaluation='+json.dumps(record),flush=True)
            epoch+=1
        selected=opt.output_dir/'best.pt'
        if selected.exists():
            model.load_state_dict(torch.load(selected,map_location='cuda',weights_only=False)['model'])
            heldout=evaluate(model,renderer,test,config,opt.output_dir,'heldout',test_paths[:opt.clean_samples])
            write_json(opt.output_dir/'heldout_test.json',heldout)
        write_json(opt.output_dir/'status.json',{'state':'complete','step':step,'best_checkpoint_exists':selected.exists(),
                                               'elapsed_seconds':time.time()-start})
    except BaseException as error:
        write_json(opt.output_dir/'status.json',{'state':'failed','step':step,'error':repr(error)})
        raise


if __name__=='__main__':main()
