"""Prepare v104 on the deployed parent, with new native sources and a locked test split."""
import sys,json,hashlib,time,os,traceback
from pathlib import Path
from types import SimpleNamespace
import torch,numpy as np
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.infer import load_parser,load_view_images
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
from SkingToolkit.dense_uv_parser.final_head_uv import evaluation_numerics,image_evidence
from SkingToolkit.dense_uv_parser.prepare_generalization_revision import paired
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment
from SkingToolkit.renderer import DifferentiableRenderer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,d):t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def save(p,d):t=p.with_suffix('.tmp');torch.save(d,t);t.replace(p)
def uv_sha(source):return hashlib.sha256((load_skin(source)*255).round().byte().numpy().tobytes()).hexdigest()

def main():
 torch.set_num_threads(4);evaluation_numerics();torch.manual_seed(1040709)
 root=Path('dense_uv_parser').resolve();run=root/'runs/v104_topology_20260907';out=run/'cache';out.mkdir(exist_ok=True);start=time.time()
 old=json.loads((root/'runs/v103_generalization_20260907/cache/manifest.json').read_text())
 teacher=root/'runs/v103_generalization_20260907/candidate_risk_1000_beard_protected'
 config=json.loads((teacher/'pipeline.json').read_text());config['version']='v104';config.pop('final_head_material_refine_steps',None)
 assert config['joint_head_geometry_mode']=='disabled' and config['head_surface_routing_scope']=='beard'
 manifest={'revision':'v104_deployed_parent_native_topology','version':'v104','parent':old['parent'],'parent_sha256':old['parent_sha256'],'pipeline':config,'splits':{g:list(old['splits'][g]) for g in ('train','validation')},'real_training':[],'real_development':[],'test':[],'complete':False,'replay_policy':'Older cached parent errors are explicit corruption replay, not the deployed validation distribution.','test_policy':'New normalized-UV identities from source test split; never read by training/checkpoint selection.','teacher':str(teacher/'parser.pt'),'teacher_sha256':sha(teacher/'parser.pt')}
 split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
 normalized={};roles={g:set() for g in ('train','validation','test')}
 for g in ('train','validation'):
  for p in manifest['splits'][g]:
   s=load(p)['metadata']['source'];normalized.setdefault(s,uv_sha(s));roles[g].add(normalized[s])
 overlap=roles['train']&roles['validation']
 if overlap:
  manifest['splits']['train']=[p for p in manifest['splits']['train'] if normalized[load(p)['metadata']['source']] not in overlap]
  roles['train']-=overlap
 manifest['legacy_normalized_overlap_train_identities_removed']=len(overlap)
 used=set.union(*roles.values());selected={};all_items=[]
 # New sources cannot duplicate normalized skin content in any retained cohort.
 for g,offset,native_count,pair_count in [('train',480,384,64),('validation',120,64,16),('test',0,64,16)]:
  picked=[]
  for s in split[g][offset:]:
   key=uv_sha(s)
   if key in used:continue
   used.add(key);roles[g].add(key);normalized[s]=key;picked.append(s)
   if len(picked)==native_count+pair_count:break
  assert len(picked)==native_count+pair_count,(g,len(picked))
  selected[g]=picked;folder=out/g;folder.mkdir(exist_ok=True)
  for i,source in enumerate(picked):
   seed=(104070000+['train','validation','test'].index(g)*10000000)+i*104729
   for variant in (['native'] if i<native_count else ['bare','glasses','phones']):
    path=folder/f'{i:04d}_{variant}.pt';meta={'source':source,'source_sha256':sha(source),'normalized_uv_sha256':normalized[source],'family':'native_texture' if variant=='native' else 'paired_'+variant,'kind':'rendered_ground_truth','split':g,'seed':seed,'variant':variant,'pair_id':normalized[source] if variant!='native' else None,'deployed_parent':True,'appearance_augmented':g=='train' and i%2==1,'unaltered_source_uv':variant=='native'}
    if g=='test':manifest['test'].append(str(path))
    else:manifest['splits'][g].append(str(path))
    all_items.append({'path':path,'metadata':meta})
 assert not roles['train']&roles['validation'] and not roles['test']&(roles['train']|roles['validation'])
 manifest['normalized_source_identity_counts']={g:len(v) for g,v in roles.items()};manifest['new_source_counts']={g:len(v) for g,v in selected.items()}
 write(out/'selection.json',{'sources':selected,'normalized_uv_sha256':normalized,'test_locked_before_training':True});write(out/'manifest.json',manifest)
 assert sha(old['parent'])==old['parent_sha256']
 model,args=load_parser(old['parent'],torch.device('cuda'));renderer=DifferentiableRenderer(args['mappings_dir']).cuda()
 def progress(state,**kw):write(out/'status.json',{'state':state,'pid':os.getpid(),'elapsed_seconds':time.time()-start,**kw});print(json.dumps({'state':state,**kw}),flush=True)
 def process(items):
  skins=[]
  for item in items:
   m=item['metadata'];original=load_skin(m['source']);uv,labels=(original,torch.full((64,64),-100,dtype=torch.long)) if m['variant']=='native' else paired(original,m['seed'],m['variant'])
   skins.append(uv);item['labels']=labels
  uv=torch.stack(skins).cuda();images=[];fg=[]
  for view in ('front_left','back_left'):
   im,t=build_dense_parser_batch(uv,renderer,view);images.append(im);fg.append(t['foreground'][:,0])
  images=torch.stack(images,1).flatten(0,1);fg=torch.stack(fg,1).flatten(0,1)
  for j,item in enumerate(items):
   if item['metadata']['appearance_augmented']:
    generator=torch.Generator(device='cuda').manual_seed(item['metadata']['seed']);images[j*2:j*2+2]=appearance_augment(images[j*2:j*2+2],fg[j*2:j*2+2,None]>.5,strength=1.,generator=generator)
  result=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=fg)
  evidence=image_evidence(result['details']['rendered'],result['details']['routing']['observed_foreground'],result['details']['outputs']).cpu()
  for j,item in enumerate(items):
   save(item['path'],{'base':result['uv'][j:j+1].cpu(),'evidence':evidence[j*2:j*2+2],'target':uv[j:j+1].cpu(),'labels':item['labels'][None],'symmetric':torch.tensor([False]),'metadata':item['metadata']})
 pending=[];done=0
 for item in all_items:
  if item['path'].exists():assert load(item['path'])['metadata']==item['metadata'];done+=1;continue
  pending.append(item)
  if len(pending)==8:
   process(pending);done+=len(pending);pending=[];progress('preparing_synthetic',completed=done,total=len(all_items))
 if pending:process(pending);done+=len(pending)
 accepted=json.loads((teacher/'full_review.json').read_text())['cases'];real=out/'real';real.mkdir(exist_ok=True)
 annotation=json.loads((root/'regression/v102_uv_symmetry_review.json').read_text())
 for group in ('real_training','real_development'):
  for oldpath in old[group]:
   row=load(oldpath);meta=dict(row['metadata']);name=meta['name'];path=real/(name+'.pt')
   if not path.exists():
    a=accepted[name];cm=json.loads((Path(a['output'])/'manifest.json').read_text());images=load_view_images(SimpleNamespace(combined=meta['input']),config['routing']['views'],renderer).cuda()
    raw=torch.from_numpy(np.array(Image.open(cm['foreground_probability']['path'])).copy()).float()/255;fg=F.interpolate(torch.stack(raw.chunk(2,1))[:,None],images.shape[-2:],mode='nearest-exact')[:,0].cuda()
    result=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=fg);oldtarget=row['target'].clone();row['base']=result['uv'].cpu();row['evidence']=image_evidence(result['details']['rendered'],result['details']['routing']['observed_foreground'],result['details']['outputs']).cpu()
    if group=='real_training':
     # Reviewed annotation stays truth. Elsewhere retain the reviewed v103
     # prediction as a teacher, explicitly not an invented human label.
     row['target']=torch.from_numpy(np.array(Image.open(Path(a['output'])/'pred_uv.png').convert('RGBA')).copy()).permute(2,0,1)[None].float()/255
     mask=row['labels'][0]>=0
     if name=='beard':
      for field in ('beard_scope','hair_scope'):mask |= torch.from_numpy(np.array(Image.open(root.parent/annotation[field]).convert('L'))>127)
     if meta.get('geometry_annotation'):mask |= torch.from_numpy(np.array(Image.open(meta['geometry_annotation']).convert('L'))>127)
     row['target'][0,:,mask]=oldtarget[0,:,mask];meta['unlabelled_policy']='v103 reviewed prediction is a consistency teacher outside explicit annotation; no new real ground truth.'
    else:row['target']=row['base'].clone()
    row['metadata']={**meta,'rebased_v104_deployed_parent':True};save(path,row)
    if name=='beard':save(out/'anchor_input.pt',{'images':images.cpu(),'foreground':fg.cpu(),'input':meta['input']})
   manifest[group].append(str(path));progress('preparing_real',name=name)
 manifest['complete']=True;manifest['source_sha256']={p.name:sha(p) for p in root.glob('*.py')};write(out/'manifest.json',manifest)
 progress('complete',counts={**{g:len(v) for g,v in manifest['splits'].items()},'test':len(manifest['test'])},source_identity_counts=manifest['normalized_source_identity_counts'])
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=Path('dense_uv_parser/runs/v104_topology_20260907/cache/status.json');p.parent.mkdir(exist_ok=True,parents=True);write(p,{'state':'failed','error':repr(e),'traceback':traceback.format_exc()});raise
