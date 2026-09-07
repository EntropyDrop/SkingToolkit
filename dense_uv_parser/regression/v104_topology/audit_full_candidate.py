import sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.uv_reference_repair import evaluate_annotated_review
p=argparse.ArgumentParser();p.add_argument('--step',type=int,default=4000);p.add_argument('--repeat',action='store_true');p.add_argument('--label');o=p.parse_args()
root=Path('dense_uv_parser').resolve();run=root/'runs/v104_topology_20260907';label=o.label or str(o.step);candidate=run/f'candidate_{label}';output=root/'output_history'/f'v104_topology_{label}_20260907';rows=json.loads((candidate/'full_inputs.json').read_text())
image=lambda p:np.array(Image.open(p).convert('RGBA'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={json.loads(p.read_text())['input_sha256']:p for p in output.glob('*/manifest.json')};assert len(files)==len(rows)
if o.repeat:
 repeat=root/'output_history'/f'v104_topology_{label}_20260907_repeat';repeats={json.loads(p.read_text())['input_sha256']:p for p in repeat.glob('*/manifest.json')}
report={'checkpoint':str(candidate/'parser.pt'),'checkpoint_sha256':sha(candidate/'parser.pt'),'cases':{},'scope':'3 real training identities are fit checks; 40 real development images have no gradients. Includes 12 newly fixed selections beyond the 31 v103 review cases. Whole-image review is required alongside scoped checks.','release_promoted':False}
arrays={}
for row in rows:
 mp=files[row['input_sha256']];m=json.loads(mp.read_text());assert m['complete'] and m['checkpoint_sha256']==report['checkpoint_sha256'] and sha(m['input'])==row['input_sha256']
 assert m['foreground_provider']['adaptation_sha256']=='8585a698d38969e6e7562e9c7d8c5f0b5e934e8f0811cf5b8678fcf05734e732'
 a=image(mp.parent/'pred_uv.png');old=image(Path(row['old_output'])/'pred_uv.png');arrays[row['name']]=a
 r={**row,'output':str(mp.parent),'body_exact_to_old':bool(np.array_equal(a[16:],old[16:])),'outer_head':int((a[:16,32:,3]>127).sum()),'material_fit':json.loads((mp.parent/'final_head_material_refit.json').read_text())}
 if row['name']=='99LBZPR14PEYLZ69':r['eye_outer_cells']={'before':int((old[11:14,40:48,3]>127).sum()),'after':int((a[11:14,40:48,3]>127).sum())}
 if row['name']=='254NJBPVUEM759NA':
  reference=image(Path(row['input']).with_name('254NJBPVUEM759NA_result_v101.png'));scope=reference[:16,32:,3]>127
  r['v101_hair_reference']={'reference_cells':int(scope.sum()),'retained_before':int((scope&(old[:16,32:,3]>127)).sum()),'retained_after':int((scope&(a[:16,32:,3]>127)).sum()),'extra_before':int((~scope&(old[:16,32:,3]>127)).sum()),'extra_after':int((~scope&(a[:16,32:,3]>127)).sum()),'scope':'v101 support comparison, not ground truth.'}
 if row['name']=='beard':
  r['beard']=evaluate_annotated_review(a,row['input']);annotation=json.loads((root/'regression/v102_uv_symmetry_review.json').read_text());scope=np.array(Image.open(annotation['beard_scope']).convert('L'))>127;expected=image(annotation['expected_uv'])
  pairs=scope[:,:32]&scope[:,32:]&(expected[:,:32,3]>127)&(expected[:,32:,3]>127)&np.all(expected[:,:32,:3]==expected[:,32:,:3],axis=2)
  r['layer_alignment']={'pairs':int(pairs.sum()),'support_mismatches':int(((a[:,:32,3]==0)|(a[:,32:,3]==0))[pairs].sum()),'rgb_mismatches':int((np.any(a[:,:32,:3]!=a[:,32:,:3],axis=2)&pairs).sum())}
 if o.repeat and row['input_sha256'] in repeats:
  rp=repeats[row['input_sha256']];rm=json.loads(rp.read_text());assert rm['source_sha256']==m['source_sha256'] and rm['pipeline']==m['pipeline'] and rm['checkpoint_sha256']==m['checkpoint_sha256']
  b=image(rp.parent/'pred_uv.png');visible=(a[:,:,3]>0)|(b[:,:,3]>0)
  r['repeat']={'alpha_exact':bool(np.array_equal(a[:,:,3],b[:,:,3])),'body_exact':bool(np.array_equal(a[16:],b[16:])),'rgb_max_delta':int(np.abs(a[:,:,:3].astype(int)-b[:,:,:3].astype(int))[visible].max())}
 report['cases'][row['name']]=r
c=arrays['crown'];rgb=c[:,:,:3].astype(float)/255;blue=(rgb[:,:,2]>.25)&(rgb[:,:,2]-rgb[:,:,0]>.2)&(rgb[:,:,2]-rgb[:,:,1]>.08)
hat=arrays['old_PU418PLGERMFC487_edited'];h=hat[:,:,:3].astype(float)/255
phone=arrays['old_img27'][:,:,:3].astype(float)/255;green=phone[:,:,1]-np.maximum(phone[:,:,0],phone[:,:,2]);scope=np.zeros((64,64),bool);scope[:8,8:16]=1;scope[8:16,:8]=1;scope[8:16,16:32]=1
landmarks={'nose_outer_alpha':int(arrays['nose'][13,43,3]),'crown_inner_blue':int(blue[:8,8:16].sum()),'crown_outer_cap_blue':int((blue[:8,40:48]&(c[:8,40:48,3]>127)).sum()),'gold_crown_top_cells':int((arrays['old_img46'][:8,40:48,3]>127).sum()),'phone_inner_green':int(((green>.15)&scope).sum()),'glasses_forehead_outer_alpha':int(arrays['old_TWRLRRTHQP2UV368_edited'][10,41,3]),'brim_each_face':[int((hat[11,x:x+8,3]>127).sum()) for x in [32,40,48,56]],'hat_outer_upper_red':int((((h[:,:,0]-np.maximum(h[:,:,1],h[:,:,2]))[:12,32:]>.15)&(hat[:12,32:,3]>127)).sum())}
report['landmarks']=landmarks
report['checks']={'body_all_exact':all(r['body_exact_to_old'] for r in report['cases'].values()),'beard_review':report['cases']['beard']['beard']['passed'],'beard_layer_alignment':all(report['cases']['beard']['layer_alignment'][k]==0 for k in ['support_mismatches','rgb_mismatches']),'nose':landmarks['nose_outer_alpha']==0,'blue_crown':landmarks['crown_inner_blue']==0 and landmarks['crown_outer_cap_blue']>=12,'gold_crown':landmarks['gold_crown_top_cells']==0,'phone':landmarks['phone_inner_green']==0,'glasses':landmarks['glasses_forehead_outer_alpha']==0,'brim':landmarks['brim_each_face']==[8]*4 and landmarks['hat_outer_upper_red']==0,'eye_false_protrusions':report['cases']['99LBZPR14PEYLZ69']['eye_outer_cells']['after']==0,'hair_no_reference_regression':report['cases']['254NJBPVUEM759NA']['v101_hair_reference']['retained_after']>=report['cases']['254NJBPVUEM759NA']['v101_hair_reference']['retained_before'],'material_fit_preserves_geometry_and_body':all(r['material_fit']['alpha_exact'] and r['material_fit']['body_exact'] and r['material_fit']['source_mse_final']<=r['material_fit']['source_mse_before'] for r in report['cases'].values())}
if o.repeat:
 results=[r['repeat'] for r in report['cases'].values() if 'repeat' in r];assert len(results)==19
 report['repeat_count']=len(results);report['checks']['repeat_stability']=all(r['alpha_exact'] and r['body_exact'] and r['rgb_max_delta']<=1 for r in results)
report['scoped_checks_passed']=all(report['checks'].values());(candidate/'full_review.json').write_text(json.dumps(report,indent=2)+'\n')
# Visual pages: original, previous v103, candidate, front and back in each cell.
vis=candidate/'visual_review';vis.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
ordered=sorted(rows,key=lambda r:(r['name'] not in ['99LBZPR14PEYLZ69','254NJBPVUEM759NA'],rows.index(r)))
for page in range(0,len(ordered),3):
 canvas=Image.new('RGB',(1536,810),'white');d=ImageDraw.Draw(canvas)
 for ri,row in enumerate(ordered[page:page+3]):
  for ci,(label,path) in enumerate([('Input',row['input']),('Reviewed v103',Path(row['old_output'])/'simple_inpaint_render.png'),('Candidate',Path(report['cases'][row['name']]['output'])/'simple_inpaint_render.png')]):
   im=Image.open(path).convert('RGB');im=im.crop((0,0,im.width,int(im.height*.39)));im.thumbnail((512,232));canvas.paste(im,(ci*512,ri*270+30));d.text((ci*512+5,ri*270+5),row['name']+' | '+label,font=font,fill='black')
 canvas.save(vis/f'page_{page//3+1:02d}.png')
print(json.dumps({'scoped_checks_passed':report['scoped_checks_passed'],'checks':report['checks'],'landmarks':landmarks,'eye':report['cases']['99LBZPR14PEYLZ69']['eye_outer_cells'],'hair':report['cases']['254NJBPVUEM759NA']['v101_hair_reference']},indent=2))
