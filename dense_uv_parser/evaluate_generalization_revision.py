"""Cohort and no-gradient real regression review for candidate selection."""
import sys,json,argparse,hashlib
from pathlib import Path
import torch,numpy as np
from PIL import Image
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics
from SkingToolkit.dense_uv_parser.train_final_head_uv import evaluate,batch
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image
from SkingToolkit.dense_uv_parser.uv_reference_repair import evaluate_annotated_review

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);o=p.parse_args()
 torch.set_num_threads(4);evaluation_numerics()
 root=Path('dense_uv_parser/runs/v103_generalization_20260907');cache=root/'cache';manifest=json.loads((cache/'manifest.json').read_text());load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
 c=load(o.checkpoint);model=FinalHeadUVDecoder(**c['final_head_uv_config']).cuda().eval();model.load_state_dict(c['final_head_uv_state'])
 rows=[load(p) for p in manifest['splits']['validation']]
 cohorts={'old_authored':[r for r in rows if r['metadata'].get('family')!='native_texture' and not r['metadata'].get('unpruned_parent')],'old_native':[r for r in rows if r['metadata'].get('family')=='native_texture' and not r['metadata'].get('unpruned_parent')],'new_native':[r for r in rows if r['metadata'].get('family')=='native_texture' and r['metadata'].get('unpruned_parent')],'new_paired':[r for r in rows if str(r['metadata'].get('family','')).startswith('paired_')]}
 report={'checkpoint':str(o.checkpoint),'checkpoint_sha256':hashlib.sha256(o.checkpoint.read_bytes()).hexdigest(),'cohorts':{name:evaluate(model,r) for name,r in cohorts.items()},'cases':{}}
 for p in manifest['real_training']+manifest['real_development']:
  r=load(p);name=r['metadata']['name'];b=batch([r],model)
  with torch.no_grad():prediction=model(b['base'],b['evidence']);uv=prediction['uv']
  im=np.array(tensor_to_rgba_image(uv[0]));dest=o.output.parent/(o.output.stem+'_uv');dest.mkdir(exist_ok=True);Image.fromarray(im).save(dest/(name+'.png'))
  info={'role':r['metadata']['kind'],'body_exact':bool(torch.equal(uv[:,:,16:],b['base'][:,:,16:])),'outer_head_cells':int((im[:16,32:,3]>127).sum())}
  if name=='99LBZPR14PEYLZ69':info['eye_outer_cells']=int((im[11:14,40:48,3]>127).sum())
  if name=='254NJBPVUEM759NA':
   ref=np.array(Image.open(Path(r['metadata']['input']).with_name(name+'_result_v101.png')));scope=ref[:16,32:,3]>127;actual=im[:16,32:,3]>127
   info['v101_hair_reference']={'reference_cells':int(scope.sum()),'retained_cells':int((scope&actual).sum()),'missing_reference_cells':int((scope&~actual).sum()),'additional_cells':int((~scope&actual).sum()),'scope':'Reference preservation diagnostic; v101 is not ground truth.'}
  if name=='beard':info['annotated_review']=evaluate_annotated_review(im,r['metadata']['input'])
  if name=='crown':info['outer_cap_cells']=int((im[:8,40:48,3]>127).sum())
  if name=='old_img46':info['outer_cap_cells']=int((im[:8,40:48,3]>127).sum())
  if name=='old_PU418PLGERMFC487_edited':info['brim_each_face']=[int((im[11,x:x+8,3]>127).sum()) for x in [32,40,48,56]]
  if name=='nose':info['nose_outer_alpha']=int(im[13,43,3])
  if name=='old_TWRLRRTHQP2UV368_edited':info['forehead_outer_alpha']=int(im[10,41,3])
  report['cases'][name]=info
 o.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report['cases'].items() if k in ['99LBZPR14PEYLZ69','254NJBPVUEM759NA','beard','crown','old_img46','old_PU418PLGERMFC487_edited']},indent=2))
if __name__=='__main__':main()
