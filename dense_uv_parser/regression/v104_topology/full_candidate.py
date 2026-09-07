import json,subprocess,hashlib,os,argparse
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--repeat',action='store_true');o=p.parse_args()
root=Path('/home/ds/llms/SkingToolkitDev');run=root/'dense_uv_parser/runs/v104_topology_20260907';c=run/('candidate_'+o.label);c.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not (c/'parser.pt').exists():
 model=torch.load(o.checkpoint,map_location='cpu',weights_only=False);config=dict(model['inference_pipeline']);config.update(version='v104',final_head_material_refine_steps=64,final_head_material_protect_inner_footprints=True)
 assert model['final_head_uv_config']['topology_context'] and config['head_surface_routing_scope']=='beard'
 model['inference_pipeline']=config;torch.save(model,c/'parser.pt');(c/'pipeline.json').write_text(json.dumps(config,indent=2)+'\n');loaded=torch.load(c/'parser.pt',map_location='cpu',weights_only=False)
 assert all(torch.equal(v,loaded[g][k]) for g in ('model','final_head_uv_state') for k,v in model[g].items())
 (c/'manifest.json').write_text(json.dumps({'status':'pending_review','training_checkpoint':str(o.checkpoint),'training_checkpoint_sha256':sha(o.checkpoint),'checkpoint_sha256':sha(c/'parser.pt'),'pipeline_sha256':sha(c/'pipeline.json'),'model_and_decoder_tensors_exact':True,'release_promoted':False},indent=2)+'\n')
rows=json.loads((run/'review_inputs.json').read_text())
if o.repeat:rows=[r for r in rows if r['role'] in ('real_training','real_development')]
assert len(rows)==(19 if o.repeat else 43)
(c/('repeat_inputs.json' if o.repeat else 'full_inputs.json')).write_text(json.dumps(rows,indent=2)+'\n')
output=root/'dense_uv_parser/output_history'/f'v104_topology_{o.label}_20260907{"_repeat" if o.repeat else ""}'
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_DISABLE_PROGRESS_BARS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
active=[]
for i,part in enumerate([rows[::2],rows[1::2]]):
 log=(c/f'{"repeat" if o.repeat else "full"}_{i}.log').open('w');args=['/home/ds/miniconda3/envs/sking-v61-worker/bin/python','dense_uv_parser/run_local.py','batch_accessories','--checkpoint',str(c/'parser.pt'),'--pipeline',str(c/'pipeline.json'),'--output-dir',str(output),'--foreground-probabilities',*[r['probability'] for r in part],'--inputs',*[r['input'] for r in part]]
 active.append((subprocess.Popen(args,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT),log))
for proc,log in active:
 code=proc.wait();log.close()
 if code:raise RuntimeError('Full inference failed; preserve worker logs')
print(json.dumps({'count':len(rows),'output':str(output)}))
