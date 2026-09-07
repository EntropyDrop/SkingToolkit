"""Calibrate edit decisions against source-disjoint UV truth, without real images."""
import sys,json,argparse
from pathlib import Path
import torch,numpy as np
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics
from SkingToolkit.dense_uv_parser.train_final_head_uv import batch

p=argparse.ArgumentParser();p.add_argument('--train-run',default='training');o=p.parse_args()
torch.set_num_threads(4);evaluation_numerics();run=Path('dense_uv_parser/runs/v103_generalization_20260907');load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
manifest=json.loads((run/'cache/manifest.json').read_text());rows=[load(p) for p in manifest['splits']['validation']];rows=[r for r in rows if r['metadata'].get('unpruned_parent')]
cohorts={'native':[r for r in rows if r['metadata']['family']=='native_texture'],'paired':[r for r in rows if r['metadata']['family'].startswith('paired_')]}
calibration_ids=set()
for group in cohorts.values():
 ids=sorted({r['metadata']['source_sha256'] for r in group});calibration_ids.update(ids[::2])
cal=[r for r in rows if r['metadata']['source_sha256'] in calibration_ids];val=[r for r in rows if r['metadata']['source_sha256'] not in calibration_ids]
thresholds=[.5,.6,.7,.8,.85,.9,.925,.95,.975,.99,.995,.999,1.]
report={'policy':'Minimize 5 * damage to previously correct cells + remaining initial errors on calibration only. Checkpoints failing their full-inference roundtrip are excluded. Real development examples are never used to choose a threshold.','calibration_identities':len(calibration_ids),'validation_identities':len({r['metadata']['source_sha256'] for r in val}),'calibration_cases':len(cal),'validation_cases':len(val),'checkpoints':{}}
for step in sorted(int(p.stem.split('_')[1]) for p in (run/o.train_run).glob('step_*.pt')):
 c=load(run/o.train_run/f'step_{step}.pt');model=FinalHeadUVDecoder(**c['final_head_uv_config']).cuda().eval();model.load_state_dict(c['final_head_uv_state']);curves={}
 for name,group in [('calibration',cal),('validation',val)]:
  bases=[];truths=[];scores=[]
  for start in range(0,len(group),4):
   b=batch(group[start:start+4],model)
   with torch.no_grad():prediction=model(b['base'],b['evidence'])
   bases.append((b['base'].flatten(2)[:,3,model.ids][:,model.outer]>.5).cpu());truths.append((b['target'].flatten(2)[:,3,model.ids][:,model.outer]>.5).cpu());scores.append(prediction['edit_logits'][:,model.outer].sigmoid().cpu())
  base=torch.cat(bases);truth=torch.cat(truths);q=torch.cat(scores);initial_wrong=base!=truth
  curve=[]
  for t in thresholds:
   final=base^(q>=t) if t<1 else base
   damaged=int(((final!=truth)&~initial_wrong).sum());remaining=int(((final!=truth)&initial_wrong).sum());corrected=int(((final==truth)&initial_wrong).sum())
   tp=int((final&truth).sum());fp=int((final&~truth).sum());fn=int((~final&truth).sum())
   curve.append({'threshold':t,'risk':5*damaged+remaining,'damaged':damaged,'remaining_errors':remaining,'corrected':corrected,'outer_iou':tp/max(1,tp+fp+fn),'cells':base.numel()})
  curves[name]=curve
 passed=json.loads((run/o.train_run/f'roundtrip_{step}.json').read_text())['passed'];chosen=min(curves['calibration'],key=lambda r:(r['risk'],-r['threshold']))
 curves.update(roundtrip_passed=passed,selected=chosen,selected_validation=next(r for r in curves['validation'] if r['threshold']==chosen['threshold']))
 report['checkpoints'][str(step)]=curves
selected=min([(d['selected']['risk'],-d['selected']['threshold'],int(step)) for step,d in report['checkpoints'].items() if d['roundtrip_passed']])
report['selected_step']=selected[2];report['selected_threshold']=-selected[1]
(run/(o.train_run+'_calibration.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'selected_step':report['selected_step'],'selected_threshold':report['selected_threshold'],'scores':{k:{a:b for a,b in d.items() if a not in ['calibration','validation']} for k,d in report['checkpoints'].items()}},indent=2))
