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
from SkingToolkit.dense_uv_parser.final_uv_training_state import atomic_save,save_recovery,restore_recovery,restore_rng,compare_roundtrip,upstream_rgb_warning


def load(path):return torch.load(path,map_location='cpu',weights_only=False)
def write(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def batch(rows,model,augment=False):
    result={k:torch.cat([r[k] for r in rows]).cuda() for k in ('base','target','evidence','labels','symmetric')}
    if augment and model.revision==2:
        # Correct-input examples teach the editor to retain already correct UV.
        identity=torch.rand(len(rows),device='cuda')<(.5 if model.robust_edits else .25)
        result['base'][identity]=result['target'][identity]
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
        if model.topology_context and random.random()<.5:
            from SkingToolkit.dense_uv_parser.head_topology_context import augment_connected_uv
            result["base"]=augment_connected_uv(result["base"],model)
        if model.robust_edits and random.random()<.65:
            from SkingToolkit.dense_uv_parser.final_head_uv_revision import augment_local_evidence
            result['evidence']=augment_local_evidence(result['evidence'])
    return result


@torch.no_grad()
def evaluate(model,rows):
    model.eval();values={key:{'tp':0,'fp':0,'fn':0,'rgb_sum':0.,'rgb_count':0} for key in ('base','decoder')}
    edit_audit={'incorrect_base_cells':0,'corrected_cells':0,'damaged_previously_correct_cells':0}
    ids=model.ids;outer=model.outer
    for start in range(0,len(rows),4):
        b=batch(rows[start:start+4],model);pred=model(b['base'],b['evidence'])['uv']
        target=b['target'].flatten(2)[:,:,ids];truth=target[:,3]>.5
        initial=b['base'].flatten(2)[:,3,ids][:,outer]>.5
        final=pred.flatten(2)[:,3,ids][:,outer]>.5;correct=truth[:,outer]
        edit_audit['incorrect_base_cells']+=int((initial!=correct).sum())
        edit_audit['corrected_cells']+=int(((initial!=correct)&(final==correct)).sum())
        edit_audit['damaged_previously_correct_cells']+=int(((initial==correct)&(final!=correct)).sum())
        for name,uv in [('base',b['base']),('decoder',pred)]:
            p=uv.flatten(2)[:,:,ids];alpha=p[:,3]>.5;r=values[name]
            r['tp']+=int((alpha[:,outer]&truth[:,outer]).sum());r['fp']+=int((alpha[:,outer]&~truth[:,outer]).sum());r['fn']+=int((~alpha[:,outer]&truth[:,outer]).sum())
            r['rgb_sum']+=float(((p[:,:3]-target[:,:3]).abs()*truth[:,None]).sum());r['rgb_count']+=int(truth.sum())*3
    for r in values.values():r['outer_iou']=r['tp']/max(1,r['tp']+r['fp']+r['fn']);r['visible_rgb_mae']=r['rgb_sum']/max(r['rgb_count'],1)
    return {'cases':len(rows),'metrics':values,'edit_audit':edit_audit,'scope':'Train-disjoint synthetic source identities with actual parser errors; real anchor excluded.'}


@torch.no_grad()
def real_review(model,rows,renderer,out,step):
    from torchvision.utils import save_image
    model.eval();report={}
    for row in rows:
        b=batch([row],model);uv=model(b['base'],b['evidence'])['uv'];name=row['metadata']['name'];folder=out/f'real_{step}'/name;folder.mkdir(parents=True,exist_ok=True)
        image=tensor_to_rgba_image(uv[0]);image.save(folder/'uv.png')
        render=torch.cat([renderer.forward_view(uv,v) for v in ('front_left','back_left')])
        save_image(torch.cat(list(render[:,:3]),2),folder/'render.png')
        is_training=row['metadata'].get('kind')=='explicit_partial_uv_training_annotation'
        record={'role':'training_fit_not_generalization' if is_training else 'development_no_gradients','body_exact':bool(torch.equal(uv[:,:,16:],b['base'][:,:,16:])),'head_alpha_changes':int(((uv[:,3,:16]>.5)!=(b['base'][:,3,:16]>.5)).sum())}
        if row['metadata'].get('geometry_annotation'):
            scope=torch.from_numpy(np.array(Image.open(row['metadata']['geometry_annotation']).convert('L'))>127).to(uv.device)
            errors=int((((uv[0,3]>.5)!=(b['target'][0,3]>.5))&scope).sum())
            record.update(scoped_geometry_mismatched_texels=errors,scoped_geometry_passed=errors==0)
        if name=='beard':record['annotated_uv_review']=evaluate_annotated_review(np.array(image),row['metadata']['input'])
        report[name]=record
    write(out/f'real_{step}.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',choices=['v103','v104'],default='v103');p.add_argument('--topology-context',action='store_true');p.add_argument('--boundary-loss-weight',type=float,default=0.)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=6000);p.add_argument('--batch-size',type=int,default=6)
    p.add_argument('--eval-every',type=int,default=1000);p.add_argument('--first-eval',type=int,default=200)
    p.add_argument('--resume',type=Path,help='Atomic model/optimizer/RNG recovery; output must be a new directory')
    p.add_argument('--stop-after',type=int,help='Save and stop without changing the total LR schedule')
    p.add_argument('--decoder-revision',type=int,choices=[1,2],default=1);p.add_argument('--anchor-every',type=int,default=4)
    p.add_argument('--robust-edits',action='store_true');p.add_argument('--init-checkpoint',type=Path)
    p.add_argument('--edit-risk-weight',type=float,default=0.,help='Per-cell cost of damaging a correct texel; zero retains legacy separately balanced loss')
    p.add_argument('--paired-every',type=int,default=0,help='Sample a bare/glasses/phones triplet every N steps and supervise its occupancy differences')
    p.add_argument('--semantic-geometry',action='store_true',help='Joint categorical absence/material prediction for final outer head UV')
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--native-fraction',type=float,default=0.,help='Fraction of synthetic batch slots replaying unaltered source head textures')
    o=p.parse_args()
    if min(o.steps,o.batch_size,o.eval_every,o.first_eval,o.anchor_every)<1 or (o.stop_after is not None and not 1<=o.stop_after<=o.steps):p.error('Invalid positive training limits')
    if o.boundary_loss_weight<0:p.error('boundary-loss-weight must be nonnegative')
    if (o.topology_context or o.boundary_loss_weight) and o.decoder_revision!=2:p.error('Topology training requires revision 2')
    if o.edit_risk_weight<0:p.error("edit-risk-weight must be nonnegative")
    if o.paired_every<0 or (o.paired_every and o.batch_size<4):p.error('Paired replay requires batch-size >= 4 and nonnegative interval')
    if o.learning_rate<=0 or (o.resume and o.init_checkpoint):p.error('Use positive LR and either resume or weight initialization')
    if not 0<=o.native_fraction<=1:p.error('native-fraction must be in [0,1]')
    if (o.robust_edits or o.init_checkpoint or o.semantic_geometry or o.edit_risk_weight) and o.decoder_revision!=2:p.error('These options require decoder revision 2')
    torch.set_num_threads(4);evaluation_numerics();torch.manual_seed(1032707);random.seed(1032707)
    root=Path(__file__).resolve().parent;out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    cache=json.loads((o.cache/'manifest.json').read_text())
    if not cache.get('complete'):raise ValueError('Incomplete cache')
    if hashlib.sha256(Path(cache['parent']).read_bytes()).hexdigest()!=cache['parent_sha256']:raise ValueError('Parent changed')
    train=[load(x) for x in cache['splits']['train']];validation=[load(x) for x in cache['splits']['validation']]
    native=[r for r in train if r['metadata'].get('family')=='native_texture']
    authored=[r for r in train if r['metadata'].get('family')!='native_texture']
    if o.native_fraction and (not native or not authored):raise ValueError('Native replay requires both native and authored training sources')
    pairs={}
    for r in train:
        if str(r['metadata'].get('family','')).startswith('paired_'):
            pairs.setdefault(r['metadata']['pair_id'],{})[r['metadata']['variant']]=r
    pairs=[list(p[k] for k in ('bare','glasses','phones')) for p in pairs.values() if set(p)=={'bare','glasses','phones'}]
    if o.paired_every and not pairs:raise ValueError('No complete training counterfactual groups')
    anchors=[load(x) for x in cache['real_training']];development=[load(x) for x in cache['real_development']]
    train_sources={r['metadata']['source'] for r in train};val_sources={r['metadata']['source'] for r in validation}
    if train_sources&val_sources:raise ValueError('Source leakage')
    if {r['metadata']['input_sha256'] for r in anchors}&{r['metadata']['input_sha256'] for r in development}:raise ValueError('Real training/development overlap')
    parent=load(cache['parent']);pipeline=dict(cache['pipeline']);pipeline['version']=o.version;config={'width':96,'layers':2}
    if o.decoder_revision==2:config.update(revision=2,mappings_dir=parent['args']['mappings_dir'])
    if o.robust_edits:config['robust_edits']=True
    if o.semantic_geometry:config['semantic_geometry']=True
    if o.edit_risk_weight:config['edit_risk_weight']=o.edit_risk_weight
    if o.topology_context:config['topology_context']=True
    if o.boundary_loss_weight:config['boundary_loss_weight']=o.boundary_loss_weight
    model=FinalHeadUVDecoder(**config).cuda()
    if o.init_checkpoint:
        initial=load(o.init_checkpoint)
        if initial['final_head_uv_manifest']['parent_sha256']!=cache['parent_sha256']:raise ValueError('Initialization parent mismatch')
        from SkingToolkit.dense_uv_parser.final_head_uv_revision import initialize_revision
        initialize_revision(model,initial);del initial
    opt=torch.optim.AdamW(model.parameters(),lr=o.learning_rate,weight_decay=1e-4)
    renderer=DifferentiableRenderer(parent['args']['mappings_dir']).cuda()
    manifest={'version':o.version,'revision':'final_uv_decoder_20260907','git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'parent':cache['parent'],'parent_sha256':cache['parent_sha256'],'cache':str(o.cache.resolve()),'cache_manifest_sha256':hashlib.sha256((o.cache/'manifest.json').read_bytes()).hexdigest(),'decoder_config':config,'trainable_parameters':sum(x.numel() for x in model.parameters()),'steps':o.steps,'real_training_identities':[r['metadata'] for r in anchors],'real_development_identities':[r['metadata'] for r in development],'validation_scope':'The beard identity is now training data. Only remaining real cases and train-disjoint synthetic identities assess transfer. No automatic release.','source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}}
    signature={k:manifest[k] for k in ('parent_sha256','cache_manifest_sha256','decoder_config','steps')};signature['batch_size']=o.batch_size
    if o.decoder_revision==2:manifest['revision']='local_evidence_relations_20260907'
    if o.robust_edits:manifest['revision']='local_context_preservation_20260907'
    if o.semantic_geometry:manifest['revision']='joint_semantic_geometry_20260907'
    if o.topology_context:manifest['revision']='learned_cube_neighborhood_20260907'
    manifest['validation_scope']='Explicit real training identities are fit checks only. Remaining real cases have no gradient use; synthetic validation sources are disjoint. No automatic release.'
    if o.learning_rate!=3e-4:signature['learning_rate']=o.learning_rate
    manifest['learning_rate']=o.learning_rate
    manifest['weight_initialization']={'checkpoint':str(o.init_checkpoint.resolve()),'sha256':hashlib.sha256(o.init_checkpoint.read_bytes()).hexdigest(),'optimizer_restored':False} if o.init_checkpoint else None
    if o.decoder_revision==2 or o.anchor_every!=4:signature['anchor_every']=o.anchor_every
    if o.native_fraction:signature['native_fraction']=o.native_fraction
    manifest['native_fraction']=o.native_fraction
    if o.paired_every:signature['paired_every']=o.paired_every
    manifest['paired_every']=o.paired_every;manifest['paired_training_identities']=len(pairs)
    manifest.update(training_signature=signature,resumed_from=str(o.resume.resolve()) if o.resume else None,recovery_format=1)
    write(out/'config.json',manifest);write(out/'pipeline.json',pipeline);snapshot=out/'source';snapshot.mkdir()
    for f in root.glob('*.py'):shutil.copy2(f,snapshot/f.name)
    step=0;start=time.time();validation_warnings=[]
    if o.resume:
        # Preserve failed stability evidence across retries instead of retrying until green.
        previous=o.resume.parent/'validation_warnings.json'
        if previous.exists():validation_warnings.extend(json.loads(previous.read_text()))
        for report_path in sorted(o.resume.parent.glob('roundtrip_*.json')):
            report=json.loads(report_path.read_text())
            if upstream_rgb_warning(report):validation_warnings.append({'report':str(report_path.resolve()),'reason':'upstream_rgb_variation','passed':False})
    write(out/'validation_warnings.json',validation_warnings)
    if o.resume:step=restore_recovery(o.resume,model,opt,signature)
    def status(state,**extra):
        row={'state':state,'pid':os.getpid(),'step':step,'total_steps':o.steps,'elapsed_seconds':time.time()-start,**extra};write(out/'status.json',row);print(json.dumps(row),flush=True)
    def validate_checkpoint(path,payload):
        status('validation');metrics=evaluate(model,validation);write(out/f'evaluation_{step}.json',metrics)
        payload['final_head_uv_metrics']=metrics;atomic_save(payload,path)
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
        fresh,_=load_parser(path,torch.device('cuda'));sample=load(o.cache/'anchor_input.pt');captured={}
        def capture(module,args):captured.update(base=args[0].detach().clone(),evidence=args[1].detach().clone())
        handle=fresh.final_head_uv_decoder.register_forward_pre_hook(capture)
        try:
            with torch.no_grad():result=run_pipeline(fresh,renderer,sample['images'].cuda(),pipeline,complete=True,foreground_probability=sample['foreground'].cuda())
        finally:handle.remove()
        # Check deployment and the in-memory decoder on IDENTICAL inputs exactly.
        with torch.no_grad():same_inputs=model(captured['base'],captured['evidence'])['uv']
        if not torch.equal(same_inputs,result['uv']):raise RuntimeError('Full inference decoder differs on identical inputs')
        roundtrip=compare_roundtrip(expected,result['uv'],anchor['base'],captured['base'],anchor['evidence'],captured['evidence'])
        roundtrip['serialized_decoder_exact']=True;roundtrip['fresh_inputs_decoder_exact']=True
        write(out/f'roundtrip_{step}.json',roundtrip)
        if not roundtrip['passed']:
            if not upstream_rgb_warning(roundtrip):raise RuntimeError('Full inference geometry/body/evidence check failed; see roundtrip report')
            validation_warnings.append({'step':step,'report':str(out/f'roundtrip_{step}.json'),'reason':'upstream_rgb_variation','passed':False})
            write(out/'validation_warnings.json',validation_warnings)
            status('validation_warning',reason='Upstream RGB stability failed; keep training state, do not accept or release candidate',report=str(out/f'roundtrip_{step}.json'))
        del fresh,result,captured,same_inputs
        status('real_development_review');real=real_review(model,anchors+development,renderer,out,step)
        write(out/'latest.json',{'step':step,'checkpoint':str(path),'recovery_state':str(out/'training_state_latest.pt'),'parent_all_tensors_preserved':True,'full_inference_roundtrip_exact':roundtrip['rgba_exact'],'full_inference_roundtrip_passed':roundtrip['passed'],'validation_warning_count':len(validation_warnings),'release_promoted':False})
        status('checkpoint_saved',checkpoint=str(path),validation=metrics,anchor=real['beard'].get('annotated_uv_review'),release_promoted=False,roundtrip_passed=roundtrip['passed'],validation_warning_count=len(validation_warnings))
        return roundtrip['passed']
    def checkpoint():
        # Save the recovery state BEFORE validation, including failure-prone full inference.
        saved_rng=save_recovery(out/'training_state_latest.pt',model,opt,step,signature)
        payload={**parent,'step':step,'parent_checkpoint_step':parent.get('step'),'final_head_uv_state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'final_head_uv_config':config,'final_head_uv_manifest':manifest,'final_head_uv_step':step,'inference_pipeline':pipeline}
        path=out/f'step_{step}.pt';atomic_save(payload,path)
        write(out/'recovery.json',{'step':step,'state':str(out/'training_state_latest.pt'),'validation':'pending'})
        try:
            passed=validate_checkpoint(path,payload)
            write(out/'recovery.json',{'step':step,'state':str(out/'training_state_latest.pt'),'validation':'passed' if passed else 'upstream_rgb_warning'})
        finally:restore_rng(saved_rng)  # Validation must not change subsequent training samples.
    try:
        if o.resume:
            status('resumed',from_state=str(o.resume));checkpoint()
        else:
            status('baseline_validation');write(out/'baseline_validation.json',evaluate(model,validation))
        while step<o.steps:
            model.train();rows=random.choices(train,k=o.batch_size)
            if o.native_fraction:
                # Reserve the final slot for optional real training annotations.
                count=round((o.batch_size-1)*o.native_fraction)
                rows=random.choices(native,k=count)+random.choices(authored,k=o.batch_size-count)
            paired_step=bool(o.paired_every and step%o.paired_every==0)
            if paired_step:rows[:3]=random.choice(pairs)
            if step%o.anchor_every==0:rows[-1]=random.choice(anchors)
            augment=step%2==1 if model.revision==1 else random.random()<.75
            b=batch(rows,model,augment=augment);opt.zero_grad(set_to_none=True)
            prediction=model(b['base'],b['evidence']);loss,metrics=final_uv_loss(model,prediction,b['base'],b['target'],b['labels'],b['symmetric'])
            if paired_step:
                from SkingToolkit.dense_uv_parser.final_head_uv_revision import paired_occupancy_loss
                paired_loss=paired_occupancy_loss(model,prediction,b['target']);loss=loss+4*paired_loss;metrics['paired_occupancy']=paired_loss.detach()
            if not torch.isfinite(loss):raise RuntimeError('Non-finite final UV loss')
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(grad):raise RuntimeError('Non-finite final UV gradient')
            opt.step();step+=1
            for group in opt.param_groups:group['lr']=o.learning_rate*(.1+.9*(1+math.cos(math.pi*step/o.steps))/2)
            if step==1 or step%50==0:status('training',loss=float(loss.detach()),**{k:float(v) for k,v in metrics.items()})
            if step==o.first_eval or step%o.eval_every==0 or step==o.steps or step==o.stop_after:checkpoint()
            if step==o.stop_after and step<o.steps:
                status('stopped',recovery_state=str(out/'training_state_latest.pt'));return
        status('complete_with_validation_warnings' if validation_warnings else 'complete',release_promoted=False,parent_all_tensors_preserved=True,validation_warning_count=len(validation_warnings))
    except BaseException as e:
        status('failed',error=repr(e));raise


if __name__=='__main__':main()
