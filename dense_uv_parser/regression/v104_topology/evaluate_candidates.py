"""Select on validation and development only; never load the locked test rows."""
import json,sys,hashlib,argparse
from pathlib import Path
import numpy as np,torch
from PIL import Image
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image
from SkingToolkit.dense_uv_parser.uv_reference_repair import evaluate_annotated_review
root=Path('dense_uv_parser').resolve();run=root/'runs/v104_topology_20260907';load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
p=argparse.ArgumentParser();p.add_argument('--training',default='training');o=p.parse_args();train=run/o.training
torch.set_num_threads(4);evaluation_numerics();cache=json.loads((run/'cache/manifest.json').read_text());validation=[load(p) for p in cache['splits']['validation']];real=[load(p) for p in cache['real_training']+cache['real_development']]
reference=torch.from_numpy(np.array(Image.open('/home/ds/llms/SKING_DDJ_Dataset/entropydrop_website_generations/SKING_DDJ_v54/254NJBPVUEM759NA_result_v101.png').convert('RGBA')).copy())[:,:,3]>127
paths=[root/'runs/v103_generalization_20260907/candidate_risk_1000_beard_protected/parser.pt']+sorted(train.glob('step_*.pt'),key=lambda p:int(p.stem.split('_')[1]))
report={'scope':'Checkpoint selection on validation and known development examples; locked test targets not loaded.','test_opened':False,'checkpoints':[]}
for path in paths:
 c=load(path);m=FinalHeadUVDecoder(**c['final_head_uv_config']).cuda().eval();m.load_state_dict(c['final_head_uv_state']);observable=(m.projection.sum((0,2))>0).cpu()[m.outer.cpu()];step=c.get('final_head_uv_step');label='v103' if path==paths[0] else path.stem;stats={}
 for offset in range(0,len(validation),8):
  rows=validation[offset:offset+8];base=torch.cat([r['base'] for r in rows]).cuda();evidence=torch.cat([r['evidence'] for r in rows]).cuda()
  with torch.no_grad():pred=m(base,evidence)['uv'][:,3].cpu()>.5
  for i,r in enumerate(rows):
   meta=r['metadata'];cohort=('deployed_' if meta.get('deployed_parent') else 'replay_')+('native' if meta.get('family')=='native_texture' else 'paired' if str(meta.get('family','')).startswith('paired_') else 'authored')
   s=stats.setdefault(cohort,{'cases':0,'tp':0,'fp':0,'fn':0,'observed_tp':0,'observed_fp':0,'observed_fn':0});t=r['target'][0,3]>.5;ids=m.ids[m.outer].cpu();a=pred[i].flatten()[ids];b=t.flatten()[ids]
   s['cases']+=1;s['tp']+=int((a&b).sum());s['fp']+=int((a&~b).sum());s['fn']+=int((~a&b).sum());s['observed_tp']+=int((a&b&observable).sum());s['observed_fp']+=int((a&~b&observable).sum());s['observed_fn']+=int((~a&b&observable).sum())
 for s in stats.values():
  s['iou']=s['tp']/max(1,s['tp']+s['fp']+s['fn']);s['observed_iou']=s['observed_tp']/max(1,s['observed_tp']+s['observed_fp']+s['observed_fn'])
 checks={};development={}
 for r in real:
  with torch.no_grad():uv=m(r['base'].cuda(),r['evidence'].cuda())['uv'].cpu()
  name=r['metadata']['name'];a=np.array(tensor_to_rgba_image(uv[0]));info={'outer_head':int((a[:16,32:,3]>127).sum())}
  if name=='beard':info['beard']=evaluate_annotated_review(a,r['metadata']['input']);checks['beard']=info['beard']['passed']
  if r['metadata'].get('geometry_annotation'):
   scope=torch.from_numpy(np.array(Image.open(r['metadata']['geometry_annotation']).convert('L'))>127);errors=int((((uv[0,3]>.5)!=(r['target'][0,3]>.5))&scope).sum());info['annotation_geometry_errors']=errors;checks[name]=errors==0
  if name=='99LBZPR14PEYLZ69':info['eye_false_outer']=int((a[11:14,40:48,3]>127).sum());checks['eye']=info['eye_false_outer']==0
  if name=='254NJBPVUEM759NA':
   support=torch.from_numpy(a[:,:,3]>127);info['v101_hair_retained']=int((support[:16,32:]&reference[:16,32:]).sum());info['outside_v101_hair']=int((support[:16,32:]&~reference[:16,32:]).sum())
   # A regression guard only, not a claim that the v101 reference is ground truth.
   checks['hair_no_large_regression']=info['v101_hair_retained']>=70 and info['outside_v101_hair']<=4
  development[name]=info
 roundtrip={'passed':True} if label=='v103' else json.loads((train/f'roundtrip_{step}.json').read_text())
 score=sum((.2*(s['fp']+s['fn'])+.8*(s['observed_fp']+s['observed_fn']))*(1 if k.startswith('deployed_') else .25) for k,s in stats.items())
 record={'label':label,'checkpoint':str(path),'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'step':step,'cohorts':stats,'development':development,'checks':checks,'roundtrip_passed':roundtrip['passed'],'validation_error_score':score,'eligible':all(checks.values()) and roundtrip['passed']}
 report['checkpoints'].append(record);print(label,score,checks,flush=True);del m,c
baseline=report['checkpoints'][0];eligible=[r for r in report['checkpoints'][1:] if r['eligible']]
report['selected']=min(eligible,key=lambda r:r['validation_error_score']) if eligible else None
report['selection_rule']='Lowest deployed validation errors (view-covered candidate cells weight 1, unobserved cells weight 0.2) plus 0.25 times replay errors, after serialization/roundtrip and existing scoped development regression guards. v101 hair guard is only protection from a large known regression.'
report['baseline_validation_error_score']=baseline['validation_error_score'];report['improved_validation']=bool(report['selected'] and report['selected']['validation_error_score']<baseline['validation_error_score'])
(run/(o.training+'_selection.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['improved_validation','baseline_validation_error_score']},indent=2));print('selected',report['selected']['checkpoint'] if report['selected'] else None)
