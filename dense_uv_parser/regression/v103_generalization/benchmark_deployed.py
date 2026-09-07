"""Compare complete geometry pipelines on identical held-out rendered source skins."""
import sys,json,hashlib,time
from pathlib import Path
import torch
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.infer import load_parser
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
from SkingToolkit.dense_uv_parser.final_head_uv import evaluation_numerics
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
from SkingToolkit.renderer import DifferentiableRenderer

torch.set_num_threads(4);evaluation_numerics();run=Path('dense_uv_parser/runs/v103_generalization_20260907');load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
m=json.loads((run/'cache/manifest.json').read_text());rows=[load(p) for p in m['splits']['validation']];rows=[r for r in rows if r['metadata'].get('unpruned_parent')]
paths={'old_v103':Path('dense_uv_parser/runs/v103_semantic_revision_20260907/candidate_final_material_guarded/parser.pt'),'candidate':run/'candidate_risk_1000_beard/parser.pt'}
models={};configs={}
for key,path in paths.items():
 models[key],args=load_parser(path,torch.device('cuda'));configs[key]=dict(load(path)['inference_pipeline']);configs[key]['final_head_material_refine_steps']=0
renderer=DifferentiableRenderer(args['mappings_dir']).cuda();top=build_simple_uv_topology();mask=(top.valid&(top.part==0)&(top.layer==1)).cuda()
report={'scope':'56 train-disjoint synthetic images from 40 source identities; same raw renders, ground-truth foreground, full geometry pipeline including its initial RGB fit. Final RGB-only refit omitted because it cannot change alpha. These validation identities have been used for calibration/development, not independent real-world performance estimation.','checkpoints':{k:{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},'rows':[]}
start=time.time()
for offset in range(0,len(rows),4):
 group=rows[offset:offset+4];uv=torch.cat([r['target'] for r in group]).cuda();images=[];fg=[]
 for view in ['front_left','back_left']:
  im,t=build_dense_parser_batch(uv,renderer,view);images.append(im);fg.append(t['foreground'][:,0])
 images=torch.stack(images,1).flatten(0,1);fg=torch.stack(fg,1).flatten(0,1);truth=uv[:,3]>.5
 preds={key:run_pipeline(model,renderer,images,configs[key],complete=True,foreground_probability=fg)['uv'] for key,model in models.items()}
 for j,row in enumerate(group):
  result={'source_sha256':row['metadata']['source_sha256'],'family':row['metadata']['family'],'cohort':'native' if row['metadata']['family']=='native_texture' else 'paired'}
  for key,pred in preds.items():
   a=pred[j,3]>.5;t=truth[j];result[key]={'tp':int((a&t&mask).sum()),'fp':int((a&~t&mask).sum()),'fn':int((~a&t&mask).sum())}
  report['rows'].append(result)
 (run/'deployed_benchmark_progress.json').write_text(json.dumps({'completed':len(report['rows']),'total':len(rows),'elapsed_seconds':time.time()-start},indent=2))
 print('benchmark',len(report['rows']),len(rows),flush=True)
report['cohorts']={}
for cohort in ['native','paired']:
 report['cohorts'][cohort]={}
 for key in paths:
  sums={n:sum(r[key][n] for r in report['rows'] if r['cohort']==cohort) for n in ['tp','fp','fn']};sums['iou']=sums['tp']/max(1,sum(sums.values()));report['cohorts'][cohort][key]=sums
report['elapsed_seconds']=time.time()-start;(run/'deployed_benchmark.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['cohorts'],indent=2))
