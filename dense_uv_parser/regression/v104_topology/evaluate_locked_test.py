"""Open the fixed test split after a candidate is selected and visually reviewed."""
import sys,json,argparse,hashlib,time
from pathlib import Path
import torch
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);o=p.parse_args();run=Path('dense_uv_parser/runs/v104_topology_20260907');load=lambda p:torch.load(p,map_location='cpu',weights_only=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((o.candidate/'full_review.json').read_text())['scoped_checks_passed']
assert (o.candidate/'visual_review.json').is_file()
lock=run/'test_opened.json';record={'checkpoint':str((o.candidate/'parser.pt').resolve()),'checkpoint_sha256':sha(o.candidate/'parser.pt'),'opened_at':time.time()}
if lock.exists():assert json.loads(lock.read_text())['checkpoint_sha256']==record['checkpoint_sha256'],'Test was already opened for another checkpoint'
else:lock.write_text(json.dumps(record,indent=2)+'\n')
torch.set_num_threads(4);evaluation_numerics();cache=json.loads((run/'cache/manifest.json').read_text());rows=[load(p) for p in cache['test']]
report={'scope':'112 synthetic renders from 80 normalized UV identities excluded from v104 decoder training and checkpoint selection. Cached input from full deployed frozen parent, including material fit; measured final outer-head geometry. Not a real-edited-image accuracy estimate.','test_size':len(rows),'models':{}}
paths={'v103':Path('dense_uv_parser/runs/v103_generalization_20260907/candidate_risk_1000_beard_protected/parser.pt'),'v104':o.candidate/'parser.pt'}
for name,path in paths.items():
 c=load(path);model=FinalHeadUVDecoder(**c['final_head_uv_config']).cuda().eval();model.load_state_dict(c['final_head_uv_state']);stats={};details=[]
 for start in range(0,len(rows),8):
  group=rows[start:start+8];base=torch.cat([r['base'] for r in group]).cuda();evidence=torch.cat([r['evidence'] for r in group]).cuda()
  with torch.no_grad():pred=model(base,evidence)['uv'][:,3].cpu()>.5
  ids=model.ids[model.outer].cpu();faces=model.geometry[model.outer,5:11].argmax(1).cpu()
  for i,row in enumerate(group):
   a=pred[i].flatten()[ids];b=(row['target'][0,3].flatten()[ids]>.5);cohort='native' if row['metadata']['family']=='native_texture' else 'paired'
   for key,mask in [(cohort,torch.ones_like(a))]+[(cohort+'_face_'+str(f),faces==f) for f in range(6)]:
    s=stats.setdefault(key,{'tp':0,'fp':0,'fn':0});s['tp']+=int((a&b&mask).sum());s['fp']+=int((a&~b&mask).sum());s['fn']+=int((~a&b&mask).sum())
   details.append({'source':row['metadata']['normalized_uv_sha256'],'variant':row['metadata']['variant'],'tp':int((a&b).sum()),'fp':int((a&~b).sum()),'fn':int((~a&b).sum())})
 for s in stats.values():s['iou']=s['tp']/max(1,s['tp']+s['fp']+s['fn'])
 report['models'][name]={'checkpoint_sha256':sha(path),'cohorts':stats,'cases':details};del model,c
(run/'locked_test_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{g:v['cohorts'][g] for g in ('native','paired')} for k,v in report['models'].items()},indent=2))
