from pathlib import Path
import sys,json,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_local import bind_checkout
bind_checkout()
import torch
from SkingToolkit.renderer import DifferentiableRenderer
from SkingToolkit.dense_uv_parser.infer import load_parser,simple_inpaint_uv
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.headwear_data import make_headwear_skin,render_headwear
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
root=Path(__file__).resolve().parents[1];out=root/'runs/v101_crown_geometry_20260906';torch.set_num_threads(4)
model,args=load_parser(root/'runs/v101_headwear_release_20260906/parser.pt',torch.device('cuda'));renderer=DifferentiableRenderer(args['mappings_dir']).cuda()
base=json.loads((root/'runs/v101_headwear_release_20260906/pipeline.json').read_text());candidate={**base,'crown_top_geometry_mode':'rendered_semantics'}
paths=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())['test'][224:256]
# Previously used test source identities, new predeclared procedural draws.
# This is a geometry regression set, not an untouched end-to-end test set.
topology=build_simple_uv_topology();top=(topology.valid&(topology.part==0)&(topology.layer==1)&(topology.face==4)).cuda();outerhead=(topology.valid&(topology.part==0)&(topology.layer==1)).cuda()
rows=[]
for i,path in enumerate(paths):
 has_caps=i%2==0
 for offset in range(1000):
  seed=155101037+i*1000+offset;uv,labels=make_headwear_skin(load_skin(path),seed)
  if (labels==6).any() and bool((uv[3].cuda()[top]>.5).any())==has_caps:break
 else:raise RuntimeError('No crown fixture')
 uv=uv[None].cuda();labels=labels[None].cuda()
 with torch.no_grad():
  images,truth=render_headwear(uv,labels,renderer,['front_left','back_left'])
  before=run_pipeline(model,renderer,images,base,complete=False,foreground_probability=images[:,3])
  after=run_pipeline(model,renderer,images,candidate,complete=False,outputs=before['outputs'],foreground_probability=images[:,3])
 row={'source':path,'seed':seed,'has_caps':has_caps,'presence':before['outputs']['headwear_presence_logits'].sigmoid()[:,1].tolist()}
 truth_alpha=uv[0,3]>.5
 for name,result in [('before',before),('after',after)]:
  skin=simple_inpaint_uv(result['conditioning'].cpu())[0].cuda();pred=skin[3]>.5
  row[name]={'top_tp':int((pred&truth_alpha&top).sum()),'top_fn':int((~pred&truth_alpha&top).sum()),'top_fp':int((pred&~truth_alpha&top).sum()),'top_tn':int((~pred&~truth_alpha&top).sum())}
  semantic=skin[None].clone();semantic[:,:3]=0;semantic[:,0]=outerhead.float()
  with torch.no_grad():rr=torch.stack([renderer.forward_view(semantic,v) for v in ['front_left','back_left']],1).flatten(0,1)
  predicted=rr[:,0]-rr[:,1]>.5;positive=truth==6;head=truth>=0
  # Compare rendered visible crown coverage, not pre-projection routing labels.
  row[name]['visible_crown_tp']=int((predicted&positive).sum());row[name]['visible_crown_n']=int(positive.sum())
  if name=='before':old=skin
  else:
   row['outside_top_equal']=bool(torch.equal(old[:,~top],skin[:,~top]));row['inner_equal']=bool(torch.equal(old[:,:16,:32],skin[:,:16,:32]))
 rows.append(row);(out/'synthetic_rows.json').write_text(json.dumps(rows,indent=2));print(i,row['before'],row['after'],flush=True)
summary={'count':len(rows),'with_caps':sum(r['has_caps'] for r in rows),'source_identity_status':'previously used test identities; new procedural draws; no training','all_outside_top_equal':all(r['outside_top_equal'] for r in rows)}
for name in ['before','after']:
 summary[name]={k:sum(r[name][k] for r in rows) for k in rows[0][name]}
 summary[name]['top_recall']=summary[name]['top_tp']/max(1,summary[name]['top_tp']+summary[name]['top_fn'])
 summary[name]['visible_crown_recall']=summary[name]['visible_crown_tp']/summary[name]['visible_crown_n']
(out/'synthetic_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
