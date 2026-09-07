"""Extend actual-parser caches with stratified, source-disjoint head structures."""
import argparse,hashlib,json,sys,time
from pathlib import Path
from collections import Counter
import torch
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import evaluation_numerics
from SkingToolkit.dense_uv_parser.prepare_final_head_uv import save_case
from SkingToolkit.dense_uv_parser.final_uv_training_state import atomic_save
from SkingToolkit.dense_uv_parser.infer import load_parser
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.v103_data import make_structured_skin
from SkingToolkit.dense_uv_parser.head_semantics_data import make_joint_skin
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment
from SkingToolkit.renderer import DifferentiableRenderer

FAMILIES=('flat_beard','fringe_beard','inner_hat_ring','outer_hat_ring','open_crown','cap_crown','glasses','phones')

def author(original,seed,family,ring_required=None):
    if family in ('flat_beard','fringe_beard'):
        return make_structured_skin(original,seed,profile='flat' if family=='flat_beard' else 'fringe')
    kind='hat' if 'hat' in family else 'crown' if 'crown' in family else family
    for attempt in range(1000):
        item=make_joint_skin(original,seed+attempt*7919,kind=kind)
        labels=item['labels']
        if family=='inner_hat_ring' and not ((labels==8).any() and (labels==9).any()):continue
        if family=='outer_hat_ring' and not ((labels==10).any() and (labels==11).any()):continue
        if 'hat_ring' in family:
            # Balance full rings and partial brims, with row/depth authored randomly.
            ring=any(all(bool((labels[y,x:x+8]==12).all()) for x in (32,40,48,56)) for y in range(8,16))
            if ring_required is None:raise ValueError('Hat strata must explicitly specify brim style')
            if ring_required != ring:continue
        top=int((labels[:8,40:48]==13).sum())
        if family=='open_crown' and top!=0:continue
        if family=='cap_crown' and top==0:continue
        return item
    raise RuntimeError('Could not author family '+family)

def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--train-count',type=int,default=128);p.add_argument('--validation-count',type=int,default=32);p.add_argument('--reuse-cache',type=Path);o=p.parse_args()
    torch.set_num_threads(4);evaluation_numerics()
    root=Path('dense_uv_parser').resolve();out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    old=root/'runs/v103_final_uv_retrain_20260907/cache';manifest=json.loads((old/'manifest.json').read_text())
    assert manifest['complete'];manifest['complete']=False
    manifest['revision']='crossed_hat_strata_20260907';manifest['parent_cache']=str(old)
    manifest['reuse_cache']=str(o.reuse_cache.resolve()) if o.reuse_cache else None
    manifest['extension_counts']={'train':o.train_count,'validation':o.validation_count}
    manifest['extension_families']=FAMILIES;manifest['real_training_policy']='Only the existing annotated beard identity; other 16 real cases never enter gradients.'
    manifest['source_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}
    if (out/'manifest.json').exists():
        previous=json.loads((out/'manifest.json').read_text())
        if previous['extension_counts']!=manifest['extension_counts']:raise ValueError('Extension counts changed')
    parent=Path(manifest['parent']);assert hashlib.sha256(parent.read_bytes()).hexdigest()==manifest['parent_sha256']
    model,args=load_parser(parent,torch.device('cuda'));renderer=DifferentiableRenderer(args['mappings_dir']).cuda()
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
    start=time.time();counts={};samples={};hat_strata={}
    for group,amount,start_index in [('train',o.train_count,128),('validation',o.validation_count,32)]:
        folder=out/group;folder.mkdir(exist_ok=True);samples[group]=Counter();counts[group]=Counter();hat_strata[group]=Counter()
        for i in range(amount):
            family=FAMILIES[i%len(FAMILIES)];source=split[group][start_index+i]
            seed=(30307001 if group=='train' else 40307001)+i*104729
            ring_required=(i//len(FAMILIES))%4!=3 if 'hat_ring' in family else None
            path=folder/f'{i:05d}.pt'
            if o.reuse_cache and ring_required is None:path=o.reuse_cache.resolve()/group/path.name
            if path.exists():
                cached=torch.load(path,map_location='cpu',weights_only=False)
                assert cached['metadata']['source']==source and cached['metadata']['family']==family
                if ring_required is not None:assert cached['metadata']['ring_required']==ring_required
                labels=cached['labels'];del cached
            else:
                torch.manual_seed(seed)
                item=author(load_skin(source),seed,family,ring_required);uv=item['uv'][None].cuda();images=[];foreground=[]
                for view in ('front_left','back_left'):
                    image,target=build_dense_parser_batch(uv,renderer,view);images.append(image);foreground.append(target['foreground'][:,0])
                images=torch.cat(images);foreground=torch.cat(foreground)
                if i%3==1:images=appearance_augment(images,foreground[:,None]>.5,strength=1.4)
                with torch.no_grad():result=run_pipeline(model,renderer,images,manifest['pipeline'],complete=True,foreground_probability=foreground)
                temporary=path.with_suffix('.tmp')
                save_case(temporary,result,uv,item['labels'][None],item.get('symmetric',False),{'source':source,'family':family,'seed':seed,'kind':'rendered_ground_truth','split':group,'actual_parser_cache':True,'ring_required':ring_required})
                temporary.replace(path);labels=item['labels'];del result
            manifest['splits'][group].append(str(path));samples[group][family]+=1
            if ring_required is not None:hat_strata[group][family+('_full' if ring_required else '_partial')]+=1
            for c,n in zip(*torch.unique(labels,return_counts=True)):counts[group][int(c)]+=int(n)
            manifest['extended_class_cells']={k:dict(v) for k,v in counts.items()};manifest['extended_family_samples']={k:dict(v) for k,v in samples.items()}
            manifest['hat_crossed_strata']={k:dict(v) for k,v in hat_strata.items()}
            write(out/'manifest.json',manifest)
            print(json.dumps({'state':'preparing','split':group,'done':i+1,'total':amount,'family':family,'elapsed_seconds':time.time()-start}),flush=True)
    sources={group:{torch.load(path,map_location='cpu',weights_only=False)['metadata']['source'] for path in manifest['splits'][group]} for group in ('train','validation')}
    assert not sources['train']&sources['validation']
    if any(counts['train'].get(c,0)==0 for c in range(14)):raise RuntimeError('Missing semantic class')
    for group in ('train','validation'):
        for layer in ('inner','outer'):
            for style in ('full','partial'):
                assert hat_strata[group][layer+'_hat_ring_'+style]>0
    import shutil
    shutil.copy2(old/'anchor_input.pt',out/'anchor_input.pt')
    manifest['complete']=True;write(out/'manifest.json',manifest)
    print(json.dumps({'state':'complete','training':len(manifest['splits']['train']),'validation':len(manifest['splits']['validation']),'elapsed_seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()
