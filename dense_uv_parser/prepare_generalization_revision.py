"""Source-disjoint native replay and paired glasses/phones/bare counterfactuals.
The two user-reported images are development-only and never become targets.
"""
import sys,json,hashlib,time,shutil
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
from SkingToolkit.dense_uv_parser.head_semantics_data import make_joint_skin,FACES
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.renderer import DifferentiableRenderer

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return torch.load(p,weights_only=False,map_location='cpu')
def write(p,d):t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)
def save(p,d):t=p.with_suffix('.tmp');torch.save(d,t);t.replace(p)

def paired(original,seed,variant):
    bare=make_joint_skin(original,seed,kind='bare')
    donor=make_joint_skin(original,seed,kind=variant)
    # Same RGB, face, beard and hair for all three variants. Only the added
    # accessory differs, so appearance cannot identify the glasses target.
    uv=bare['uv'].clone();labels=bare['labels'].clone()
    if variant!='bare':
        cls=6 if variant=='glasses' else 7;mask=donor['labels']==cls
        uv[:,mask]=donor['uv'][:,mask];labels[mask]=cls
    return uv,labels

def main():
 torch.set_num_threads(4);evaluation_numerics();torch.manual_seed(7030791)
 root=Path('dense_uv_parser').resolve();out=root/'runs/v103_generalization_20260907/cache';out.mkdir(exist_ok=True)
 old=json.loads((root/'runs/v103_semantic_revision_20260907/cache_native/manifest.json').read_text())
 manifest={**old,'revision':'paired_accessories_native_unpruned','complete':False,'parent_cache':str(root/'runs/v103_semantic_revision_20260907/cache_native'),'splits':{k:list(v) for k,v in old['splits'].items()},'real_training':[],'real_development':[]}
 config=dict(old['pipeline']);config['joint_head_geometry_mode']='disabled';config.pop('final_head_material_refine_steps',None)
 manifest['pipeline']=config;manifest['cache_policy']='Old pruned parser errors are retained as corruption replay. New paired and native cases use the unpruned deployment parent. Report each validation cohort separately.'
 split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
 assert sha(old['parent'])==old['parent_sha256']
 model,args=load_parser(old['parent'],torch.device('cuda'));renderer=DifferentiableRenderer(args['mappings_dir']).cuda();start=time.time()
 def process(items):
  if not items:return
  uv=torch.stack([i['uv'] for i in items]).cuda();images=[];fg=[]
  for view in ('front_left','back_left'):
   im,t=build_dense_parser_batch(uv,renderer,view);images.append(im);fg.append(t['foreground'][:,0])
  images=torch.stack(images,1).flatten(0,1);fg=torch.stack(fg,1).flatten(0,1)
  with torch.no_grad():r=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=fg)
  evidence=image_evidence(r['details']['rendered'],r['details']['routing']['observed_foreground'],r['details']['outputs']).cpu()
  for j,i in enumerate(items):
   row={'base':r['uv'][j:j+1].cpu(),'evidence':evidence[j*2:j*2+2],'target':uv[j:j+1].cpu(),'labels':i['labels'][None],'symmetric':torch.tensor([False]),'metadata':i['metadata']}
   save(i['path'],row)
 for group,offset,native_count,pair_count in [('train',320,128,32),('validation',80,32,8)]:
  folder=out/group;folder.mkdir(exist_ok=True);items=[];paths=[]
  for i in range(native_count+pair_count):
   source=split[group][offset+i];original=load_skin(source);seed=(70307910 if group=='train' else 80307910)+i*104729
   for variant in (['native'] if i<native_count else ['bare','glasses','phones']):
    family='native_texture' if variant=='native' else 'paired_'+variant
    uv,labels=(original,torch.full((64,64),-100,dtype=torch.long)) if variant=='native' else paired(original,seed,variant)
    path=folder/f'{i:04d}_{variant}.pt';meta={'source':source,'source_sha256':sha(source),'family':family,'kind':'rendered_ground_truth','split':group,'seed':seed,'variant':variant,'pair_id':sha(source) if variant!='native' else None,'unpruned_parent':True,'unaltered_source_uv':variant=='native'}
    paths.append(str(path))
    if path.exists():assert load(path)['metadata']==meta;continue
    items.append({'path':path,'uv':uv,'labels':labels,'metadata':meta})
    if len(items)==4:
     process(items);items=[];write(out/'status.json',{'state':'preparing','split':group,'done':len(list(folder.glob('*.pt'))),'total':native_count+pair_count*3,'elapsed_seconds':time.time()-start})
     print((out/'status.json').read_text(),flush=True)
  process(items);manifest['splits'][group]+=paths
  write(out/'manifest.json',manifest)
 # Rebase all existing real roles to the same unpruned deployment stage. Preserve
 # explicit training masks/targets; the two new failures stay no-gradient cases.
 accepted=json.loads((root/'runs/v103_semantic_revision_20260907/guarded_production_review.json').read_text())['cases']
 for group in ['real_training','real_development']:
  for oldpath in old[group]:
   row=load(oldpath);meta=dict(row['metadata']);name=meta['name'];folder=out/'real';folder.mkdir(exist_ok=True);path=folder/(name+'.pt')
   if not path.exists():
    cm=json.loads((Path(accepted[name]['output'])/'manifest.json').read_text())
    images=load_view_images(SimpleNamespace(combined=meta['input']),config['routing']['views'],renderer).cuda()
    raw=torch.from_numpy(np.array(Image.open(cm['foreground_probability']['path'])).astype(np.float32))/255
    fg=F.interpolate(torch.stack(raw.chunk(2,1))[:,None],images.shape[-2:],mode='nearest-exact')[:,0].cuda()
    r=run_pipeline(model,renderer,images,config,complete=True,foreground_probability=fg)
    row['base']=r['uv'].cpu();row['evidence']=image_evidence(r['details']['rendered'],r['details']['routing']['observed_foreground'],r['details']['outputs']).cpu()
    if group=='real_development':row['target']=row['base'].clone()
    meta['rebased_unpruned_parent']=True;row['metadata']=meta;save(path,row)
    if name=='beard':save(out/'anchor_input.pt',{'images':images.cpu(),'foreground':fg.cpu(),'input':meta['input']})
   manifest[group].append(str(path));print('real',name,flush=True)
 # Fresh reports retain roles, no target labels for these two raw edited images.
 for stem in ['99LBZPR14PEYLZ69','254NJBPVUEM759NA']:
  d=load(root/'runs/v103_generalization_20260907'/stem/'diagnosis.pt');path=out/'real'/(stem+'.pt')
  if not path.exists():
   r=run_pipeline(model,renderer,d['images'].cuda(),config,complete=True,foreground_probability=d['foreground'].cuda())
   row={'base':r['uv'].cpu(),'target':r['uv'].cpu(),'evidence':image_evidence(r['details']['rendered'],r['details']['routing']['observed_foreground'],r['details']['outputs']).cpu(),'labels':torch.full((1,64,64),-100,dtype=torch.long),'symmetric':torch.tensor([False]),'metadata':{'name':stem,'input':d['input'],'input_sha256':d['input_sha256'],'kind':'unlabeled_development','rebased_unpruned_parent':True}}
   save(path,row)
  manifest['real_development'].append(str(path))
 sources={g:{sha(load(p)['metadata']['source']) for p in manifest['splits'][g]} for g in ['train','validation']}
 assert not sources['train']&sources['validation']
 manifest['source_identity_counts']={g:len(v) for g,v in sources.items()}
 manifest['new_cohorts']={'train':{'native':128,'paired_identities':32,'paired_images':96},'validation':{'native':32,'paired_identities':8,'paired_images':24}}
 manifest['source_sha256']={p.name:sha(p) for p in root.glob('*.py')}
 manifest['complete']=True;write(out/'manifest.json',manifest);write(out/'status.json',{'state':'complete','elapsed_seconds':time.time()-start,'counts':{g:len(v) for g,v in manifest['splits'].items()},'source_identity_counts':manifest['source_identity_counts']})
 print((out/'status.json').read_text(),flush=True)
if __name__=='__main__':main()
