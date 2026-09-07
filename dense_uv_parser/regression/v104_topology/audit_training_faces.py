import sys,json,torch
from pathlib import Path
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics
r=Path('dense_uv_parser/runs/v104_topology_20260907');load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
torch.set_num_threads(4);evaluation_numerics();c=load('dense_uv_parser/runs/v103_generalization_20260907/candidate_risk_1000_beard_protected/parser.pt');model=FinalHeadUVDecoder(**c['final_head_uv_config']).cuda().eval();model.load_state_dict(c['final_head_uv_state']);outer=model.outer.cpu();ids=model.ids.cpu();face=model.geometry[:,5:11].argmax(1).cpu();observable=(model.projection.sum((0,2))>0).cpu()
paths=[p for p in sorted((r/'cache/train').glob('*_native.pt'))];stats={}
for start in range(0,len(paths),8):
 rows=[load(p) for p in paths[start:start+8]];base=torch.cat([a['base'] for a in rows]);e=torch.cat([a['evidence'] for a in rows]);target=torch.cat([a['target'] for a in rows]);truth=target.flatten(2)[:,3,ids]>.5
 with torch.no_grad():prediction=model(base.cuda(),e.cuda())['uv'].cpu()
 for name,uv in [('parent',base),('v103',prediction)]:
  a=uv.flatten(2)[:,3,ids]>.5
  for f in range(6):
   mask=outer&(face==f);k=name+'_face_'+str(f);s=stats.setdefault(k,{'fp':0,'fn':0,'tp':0,'visible_candidate_cells':int((mask&observable).sum()),'candidate_cells':int(mask.sum())})
   s['tp']+=int((a&truth&mask).sum());s['fp']+=int((a&~truth&mask).sum());s['fn']+=int((~a&truth&mask).sum())
report={'scope':'Training data diagnostic only, not validation or test. Static candidate visibility does not assume true alpha.','cases':len(paths),'faces':stats};(r/'training_face_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
