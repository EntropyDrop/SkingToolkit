"""Render identity-disjoint supervised RGBA cutouts; no flood/pseudo alpha labels."""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from SkingToolkit.renderer import DifferentiableRenderer
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.semantic_generalization import write_json


def deduplicate_manifest(output):
    path=Path(output)/'manifest.json';manifest=json.loads(path.read_text())
    if 'canonical_identity_check' in manifest:return
    hashes={source:hashlib.sha256(load_skin(source).numpy().tobytes()).hexdigest() for source in {r['source'] for r in manifest['records']}}
    ownership={};kept=[]
    for split in ('test','validation','train'):
        for record in manifest['records']:
            if record['split']!=split:continue
            key=hashes[record['source']]
            if key not in ownership:ownership[key]=(split,record['source'])
            if ownership[key]!=(split,record['source']):continue
            kept.append({**record,'canonical_rgba_sha256':key})
    manifest['records']=kept
    manifest['counts']={split:len({r['canonical_rgba_sha256'] for r in kept if r['split']==split}) for split in ('train','validation','test')}
    manifest['canonical_identity_check']='Canonical normalized RGBA hashed; no identical skin in multiple splits; test then validation take precedence'
    write_json(path,manifest)
    print(json.dumps({'deduplicated_sources':manifest['counts']}),flush=True)


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--train-sources',type=int,default=4096)
    p.add_argument('--val-sources',type=int,default=128)
    p.add_argument('--test-sources',type=int,default=128)
    o=p.parse_args();o.output_dir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    split_path=root/'dense_uv_parser/runs/dense_uv_parser_v101/source_splits.json'
    splits=json.loads(split_path.read_text());rng=np.random.default_rng(610106)
    selected={}
    for name,n in [('train',o.train_sources),('validation',o.val_sources),('test',o.test_sources)]:
        selected[name]=[splits[name][int(i)] for i in rng.choice(len(splits[name]),n,replace=False)]
    assert not(set(selected['train'])&set(selected['validation']) or set(selected['train'])&set(selected['test']) or set(selected['validation'])&set(selected['test']))
    mapping=o.output_dir/'mappings';mapping.mkdir()
    source_mapping=root.parent/'github/differentiable_minecraft_renderer/mappings_512x1024'
    views=['front_left','back_left']
    for v in views:(mapping/(v+'_mapping.pt')).symlink_to(source_mapping/(v+'_mapping.pt'))
    renderer=DifferentiableRenderer(str(mapping),bg_color=(0,0,0)).cuda()
    records=[];start=time.time()
    for name,paths in selected.items():
        dest=o.output_dir/name;dest.mkdir()
        for begin in range(0,len(paths),4):
            batch_paths=paths[begin:begin+4]
            uv=torch.stack([load_skin(f) for f in batch_paths]).cuda()
            for view in views:
                with torch.no_grad():rgba=renderer.forward_view(uv,view).float().clamp(0,1).cpu()
                alpha=rgba[:,3:4]
                rgb=(rgba[:,:3]/alpha.clamp_min(1/255)).clamp(0,1)
                straight=torch.cat([rgb,alpha],1)
                for j,source in enumerate(batch_paths):
                    source_sha=hashlib.sha256(Path(source).read_bytes()).hexdigest()
                    filename=f'{source_sha[:20]}_{view}.png'
                    file=dest/filename
                    Image.fromarray((straight[j].permute(1,2,0).numpy()*255).round().astype(np.uint8),'RGBA').save(file,compress_level=2)
                    records.append({'split':name,'source':source,'source_sha256':source_sha,'view':view,'file':str(file.resolve())})
            if begin%128==0:
                state={'state':'rendering','split':name,'sources':begin+len(batch_paths),'total_sources':len(paths),'elapsed':time.time()-start}
                write_json(o.output_dir/'status.json',state);print(json.dumps(state),flush=True)
    manifest={'recipe':'Renderer alpha and foreground RGB; PNG is straight RGBA; identity-disjoint source selection',
              'original_split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'views':views,
              'mapping_sha256':{v:hashlib.sha256((source_mapping/(v+'_mapping.pt')).read_bytes()).hexdigest() for v in views},
              'counts':{k:len(v) for k,v in selected.items()},'records':records}
    write_json(o.output_dir/'manifest.json',manifest)
    deduplicate_manifest(o.output_dir)
    write_json(o.output_dir/'status.json',{'state':'complete','images':len(records),'elapsed':time.time()-start})


if __name__=='__main__':main()
