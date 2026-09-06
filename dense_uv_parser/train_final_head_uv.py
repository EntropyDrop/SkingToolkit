"""Train the final-UV decoder; preserve every parent parser tensor exactly."""
import argparse,hashlib,json,math,os,random,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,final_uv_loss,evaluation_numerics
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image
from SkingToolkit.dense_uv_parser.uv_reference_repair import evaluate_annotated_review
from SkingToolkit.renderer import DifferentiableRenderer


def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def batch(rows,model,augment=False):
    result={k:torch.cat([r[k] for r in rows]).cuda() for k in ('base','target','evidence','labels','symmetric')}
    if augment:
        b=result['base'].shape[0]
        # Consistent palette changes; semantics and geometry stay unchanged.
        scale=torch.empty(b,3,1,1,device='cuda').uniform_(.85,1.15)
        shift=torch.empty(b,3,1,1,device='cuda').uniform_(-.03,.03)
        for key in ('base','target'):
            uv=result[key];uv[:,:3]=torch.where(uv[:,3:4]>.5,(uv[:,:3]*scale+shift).clamp(0,1),uv[:,:3])
        result['evidence'][:,:3]=(result['evidence'][:,:3]*scale.repeat_interleave(2,0)+shift.repeat_interleave(2,0)).clamp(0,1)
        # Add/remove and shift face-local cells on top of real parser errors.
        # The image evidence is unchanged and never receives target UV indices.
        if random.random()<.5:
            uv=result['base'].flatten(2);ids=model.ids[model.outer]
            selected=torch.rand(b,len(ids),device='cuda')<.035
            current=uv[:,3,ids]>.5
            uv[:,3,ids]=torch.where(selected,1-current.float(),current.float())
            rgb=uv[:,:3,ids]
            substrate=uv[:,:3,ids-32]
            uv[:,:3,ids]=torch.where((selected&~current)[:,None],substrate,rgb)
            uv[:,:3,ids]*=uv[:,3:4,ids]
    return result


@torch.no_grad()
def evaluate(model,rows):
    model.eval();values={key:{'tp':0,'fp':0,'fn':0,'rgb_sum':0.,'rgb_count':0} for key in ('base','decoder')}
    ids=model.ids;outer=model.outer
    for start in range(0,len(rows),4):
        b=batch(rows[start:start+4],model);pred=model(b['base'],b['evidence'])['uv']
        target=b['target'].flatten(2)[:,:,ids];truth=target[:,3]>.5
        for name,uv in [('base',b['base']),('decoder',pred)]:
            p=uv.flatten(2)[:,:,ids];alpha=p[:,3]>.5;r=values[name]
            r['tp']+=int((alpha[:,outer]&truth[:,outer]).sum());r['fp']+=int((alpha[:,outer]&~truth[:,outer]).sum());r['fn']+=int((~alpha[:,outer]&truth[:,outer]).sum())
            r['rgb_sum']+=float(((p[:,:3]-target[:,:3]).abs()*truth[:,None]).sum());r['rgb_count']+=int(truth.sum())*3
    for r in values.values():r['outer_iou']=r['tp']/max(1,r['tp']+r['fp']+r['fn']);r['visible_rgb_mae']=r['rgb_sum']/max(r['rgb_count'],1)
    return {'cases':len(rows),'metrics':values,'scope':'Train-disjoint synthetic source identities with actual parser errors; real anchor excluded.'}


@torch.no_grad()
def real_review(model,rows,renderer,out,step):
    from torchvision.utils import save_image
    model.eval();report={}
    for row in rows:
        b=batch([row],model);uv=model(b['base'],b['evidence'])['uv'];name=row['metadata']['name'];folder=out/f'real_{step}'/name;folder.mkdir(parents=True,exist_ok=True)
        image=tensor_to_rgba_image(uv[0]);image.save(folder/'uv.png')
        render=torch.cat([renderer.forward_view(uv,v) for v in ('front_left','back_left')])
        save_image(torch.cat(list(render[:,:3]),2),folder/'render.png')
        record={'role':'training_fit_not_generalization' if name=='beard' else 'development_no_gradients','body_exact':bool(torch.equal(uv[:,:,16:],b['base'][:,:,16:])),'head_alpha_changes':int(((uv[:,3,:16]>.5)!=(b['base'][:,3,:16]>.5)).sum())}
        if name=='beard':record['annotated_uv_review']=evaluate_annotated_review(np.array(image),row['metadata']['input'])
        report[name]=record
    write(out/f'real_{step}.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch-size',type=int,default=6);p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--first-eval',type=int,default=200);o=p.parse_args()
    torch.set_num_threads(4);evaluation_numerics();torch.manual_seed(1032707);random.seed(1032707)
    root=Path(__file__).resolve().parent;out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    cache=json.loads((o.cache/'manifest.json').read_text())
    if not cache.get('complete'):raise ValueError('Incomplete cache')
    if hashlib.sha256(Path(cache['parent']).read_bytes()).hexdigest()!=cache['parent_sha256']:raise ValueError('Parent changed')
    train=[load(x) for x in cache['splits']['train']];validation=[load(x) for x in cache['splits']['validation']]
    anchors=[load(x) for x in cache['real_training']];development=[load(x) for x in cache['real_development']]
    train_sources={r['metadata']['source'] for r in train};val_sources={r['metadata']['source'] for r in validation}
    if train_sources&val_sources:raise ValueError('Source leakage')
    if {r['metadata']['input_sha256'] for r in anchors}&{r['metadata']['input_sha256'] for r in development}:raise ValueError('Real training/development overlap')
    config={'width':96,'layers':2};model=FinalHeadUVDecoder(**config).cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    parent=load(cache['parent']);pipeline=cache['pipeline'];renderer=DifferentiableRenderer(parent['args']['mappings_dir']).cuda()
    manifest={'version':'v103','revision':'final_uv_decoder_20260907','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'parent':cache['parent'],'parent_sha256':cache['parent_sha256'],'cache':str(o.cache.resolve()),'cache_manifest_sha256':hashlib.sha256((o.cache/'manifest.json').read_bytes()).hexdigest(),'decoder_config':config,'trainable_parameters':sum(x.numel() for x in model.parameters()),'steps':o.steps,'real_training_identities':[r['metadata'] for r in anchors],'real_development_identities':[r['metadata'] for r in development],'validation_scope':'The beard identity is now training data. Only remaining real cases and train-disjoint synthetic identities assess transfer. No automatic release.','source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}}
    write(out/'config.json',manifest);write(out/'pipeline.json',pipeline);snapshot=out/'source';snapshot.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    step=0;start=time.time()
    def status(state,**extra):
        row={'state':state,'pid':os.getpid(),'step':step,'total_steps':o.steps,'elapsed_seconds':time.time()-start,**extra};write(out/'status.json',row);print(json.dumps(row),flush=True)
    def checkpoint():
        status('validation');metrics=evaluate(model,validation);write(out/f'evaluation_{step}.json',metrics)
        payload={**parent,'step':step,'parent_checkpoint_step':parent.get('step'),'final_head_uv_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'final_head_uv_config':config,'final_head_uv_manifest':manifest,'final_head_uv_step':step,'final_head_uv_metrics':metrics,'inference_pipeline':pipeline}
        path=out/f'step_{step}.pt';torch.save(payload,out/'checkpoint.tmp');(out/'checkpoint.tmp').replace(path)
        # In-memory and loaded decoder run exactly the same discrete output path.
        reloaded=load(path)
        for name,value in parent['model'].items():
            if not torch.equal(value,reloaded['model'][name]):raise RuntimeError('Parent tensor changed: '+name)
        restored=FinalHeadUVDecoder(**config).cuda().eval();restored.load_state_dict(reloaded['final_head_uv_state'])
        del reloaded
        anchor=batch([anchors[0]],model)
        with torch.no_grad():
            expected=model(anchor['base'],anchor['evidence'])['uv'];actual=restored(anchor['base'],anchor['evidence'])['uv']
        if not torch.equal(expected,actual):raise RuntimeError('Serialized decoder differs from in-memory evaluation')
        del restored
        status('full_inference_roundtrip')
        from SkingToolkit.dense_uv_parser.infer import load_parser
        from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
        fresh,_=load_parser(path,torch.device('cuda'));sample=load(o.cache/'anchor_input.pt')
        with torch.no_grad():result=run_pipeline(fresh,renderer,sample['images'].cuda(),pipeline,complete=True,foreground_probability=sample['foreground'].cuda())
        left=np.array(tensor_to_rgba_image(expected[0]));right=np.array(tensor_to_rgba_image(result['uv'][0]))
        if not np.array_equal(left,right):
            from SkingToolkit.dense_uv_parser.final_head_uv import image_evidence
            actual_evidence=image_evidence(result['details']['rendered'],result['details']['routing']['observed_foreground'],result['details']['outputs'])
            write(out/f'roundtrip_failure_{step}.json',{'differences':[{'xy':[int(x),int(y)],'cached':left[y,x].tolist(),'fresh':right[y,x].tolist()} for y,x in np.argwhere((left!=right).any(2))],'evidence_max_delta':float((actual_evidence-anchor['evidence']).abs().max())})
            raise RuntimeError('Full inference roundtrip differs from cached evaluation by '+str(int((left!=right).any(2).sum()))+' texels')
        del fresh,result
        status('real_development_review');real=real_review(model,anchors+development,renderer,out,step)
        torch.save({'optimizer':opt.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate()},out/'optimizer_latest.pt')
        write(out/'latest.json',{'step':step,'checkpoint':str(path),'parent_all_tensors_preserved':True,'full_inference_roundtrip_exact':True})
        status('checkpoint_saved',checkpoint=str(path),validation=metrics,anchor=real['beard'].get('annotated_uv_review'),release_promoted=False)
    try:
        status('baseline_validation');write(out/'baseline_validation.json',evaluate(model,validation))
        while step<o.steps:
            model.train();rows=random.choices(train,k=o.batch_size)
            if step%4==0:rows[-1]=random.choice(anchors)
            b=batch(rows,model,augment=step%2==1);opt.zero_grad(set_to_none=True)
            prediction=model(b['base'],b['evidence']);loss,metrics=final_uv_loss(model,prediction,b['base'],b['target'],b['labels'],b['symmetric'])
            if not torch.isfinite(loss):raise RuntimeError('Non-finite final UV loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Non-finite final UV gradient')
            opt.step();step+=1
            for group in opt.param_groups:group['lr']=3e-4*(.1+.9*(1+math.cos(math.pi*step/o.steps))/2)
            if step==1 or step%50==0:status('training',loss=float(loss.detach()),**{k:float(v) for k,v in metrics.items()})
            if step==o.first_eval or step%o.eval_every==0 or step==o.steps:checkpoint()
        status('complete',release_promoted=False,parent_all_tensors_preserved=True)
    except BaseException as e:
        status('failed',error=repr(e));raise


if __name__=='__main__':main()
