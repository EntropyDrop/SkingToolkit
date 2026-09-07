"""Stage a candidate and re-run originals through the complete inference path."""
import json,sys,subprocess,hashlib,os,argparse
from pathlib import Path
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--step',type=int,default=4000);p.add_argument('--repeat',action='store_true');p.add_argument('--train-run',default='training');p.add_argument('--threshold',type=float,default=.5);p.add_argument('--stage-only',action='store_true');p.add_argument('--beard-surface',action='store_true');p.add_argument('--protect-footprints',action='store_true');o=p.parse_args()
root=Path('/home/ds/llms/SkingToolkitDev');run=root/'dense_uv_parser/runs/v103_generalization_20260907';py='/home/ds/miniconda3/envs/sking-v61-worker/bin/python'
label=('' if o.train_run=='training' else o.train_run.replace('training_','')+'_')+str(o.step)+(('_t'+str(o.threshold).replace('.','')) if o.threshold!=.5 else '')+('_beard' if o.beard_surface else '')+('_protected' if o.protect_footprints else '')
checkpoint=run/o.train_run/f'step_{o.step}.pt';candidate=run/f'candidate_{label}';candidate.mkdir(exist_ok=True)
if not (candidate/'parser.pt').exists():
 c=torch.load(checkpoint,map_location='cpu',weights_only=False);pipeline=dict(c['inference_pipeline']);pipeline.update(version='v103',final_head_material_refine_steps=64)
 if o.beard_surface:pipeline['head_surface_routing_scope']='beard'
 if o.protect_footprints:pipeline['final_head_material_protect_inner_footprints']=True
 assert pipeline['joint_head_geometry_mode']=='disabled' and c['final_head_uv_config']['semantic_geometry']
 c['inference_pipeline']=pipeline;c['final_head_uv_config']['edit_threshold']=o.threshold;torch.save(c,candidate/'parser.pt');(candidate/'pipeline.json').write_text(json.dumps(pipeline,indent=2)+'\n')
 reloaded=torch.load(candidate/'parser.pt',map_location='cpu',weights_only=False)
 for group in ['model','final_head_uv_state']:
  assert all(torch.equal(v,reloaded[group][k]) for k,v in c[group].items())
 (candidate/'manifest.json').write_text(json.dumps({'status':'pending_review','training_checkpoint':str(checkpoint),'training_checkpoint_sha256':sha(checkpoint),'checkpoint_sha256':sha(candidate/'parser.pt'),'pipeline_sha256':sha(candidate/'pipeline.json'),'model_and_decoder_tensors_exact':True,'release_promoted':False},indent=2)+'\n')
if o.stage_only:
 print(str(candidate));sys.exit(0)
cache=json.loads((run/'cache/manifest.json').read_text());accepted=json.loads((root/'dense_uv_parser/runs/v103_semantic_revision_20260907/guarded_production_review.json').read_text())['cases'];bulk=root/'dense_uv_parser/runs/v103_all_edited_20260907T040559Z/artifacts'
rows=[]
for kind in ['real_training','real_development']:
 for rp in cache[kind]:
  meta=torch.load(rp,map_location='cpu',weights_only=False)['metadata'];name=meta['name']
  folder=Path(accepted[name]['output']) if name in accepted else next(bulk.glob(Path(meta['input']).stem+'_*'))
  m=json.loads((folder/'manifest.json').read_text());rows.append({'name':name,'input':m['input'],'input_sha256':m['input_sha256'],'probability':m['foreground_probability']['path'],'role':kind,'old_output':str(folder)})
if not o.repeat:
 for r in json.loads((run/'bulk_change_audit.json').read_text())['extra_development_selection']:
  folder=next(bulk.glob(Path(r['input']).stem+'_*'));m=json.loads((folder/'manifest.json').read_text());rows.append({'name':Path(r['input']).stem,'input':m['input'],'input_sha256':m['input_sha256'],'probability':m['foreground_probability']['path'],'role':'extra_development','old_output':str(folder)})
assert len({r['input_sha256'] for r in rows})==len(rows)
output=root/'dense_uv_parser/output_history'/f'v103_generalization_{label}_20260907{ "_repeat" if o.repeat else ""}'
(candidate/('repeat_inputs.json' if o.repeat else 'full_inputs.json')).write_text(json.dumps(rows,indent=2)+'\n')
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_PROGRESS_BARS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
active=[]
for index,part in enumerate([rows[::2],rows[1::2]]):
 log=(candidate/f'{"repeat" if o.repeat else "full"}_{index}.log').open('w')
 args=[py,'dense_uv_parser/run_local.py','batch_accessories','--checkpoint',str(candidate/'parser.pt'),'--pipeline',str(candidate/'pipeline.json'),'--output-dir',str(output),'--foreground-probabilities',*[r['probability'] for r in part],'--inputs',*[r['input'] for r in part]]
 active.append((subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT),log))
for proc,log in active:
 code=proc.wait();log.close()
 if code:raise RuntimeError('Candidate inference failed; see worker logs')
print(json.dumps({'output':str(output),'count':len(rows),'candidate':str(candidate)}))
