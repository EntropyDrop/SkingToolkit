"""Cache actual frozen-parser errors, paired image evidence, and explicit targets."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from SkingToolkit.dense_uv_parser.infer import load_parser
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline,cached_real_foreground
from SkingToolkit.dense_uv_parser.final_head_uv import image_evidence,evaluation_numerics
from SkingToolkit.dense_uv_parser.head_semantics_data import make_joint_skin
from SkingToolkit.dense_uv_parser.v103_data import make_structured_skin
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment
from SkingToolkit.renderer import DifferentiableRenderer


def save_case(path,result,target,labels,symmetric,metadata):
    d=result['details']
    evidence=image_evidence(d['rendered'],d['routing']['observed_foreground'],d['outputs'])
    torch.save({'base':result['uv'].cpu(),'evidence':evidence.cpu(),'target':target.cpu(),'labels':labels.cpu(),'symmetric':torch.tensor([symmetric]),'metadata':metadata},path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--train-count',type=int,default=128);p.add_argument('--validation-count',type=int,default=32);p.add_argument('--quick',action='store_true');o=p.parse_args()
    torch.set_num_threads(4);evaluation_numerics();torch.manual_seed(1031707)
    root=Path(__file__).resolve().parent;out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    registry=json.loads((root/'v102_release.json').read_text());parent=root.parent/registry['checkpoint']
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=registry['checkpoint_sha256']:raise ValueError('Parent changed')
    pipeline=json.loads((root.parent/registry['pipeline']).read_text())
    # Match deployment parameter flags; run_pipeline already disables autograd.
    # The parent is never optimized or added to the decoder optimizer.
    model,args=load_parser(parent,torch.device('cuda'))
    renderer=DifferentiableRenderer(args['mappings_dir']).cuda()
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
    if set(split['train'])&(set(split['validation'])|set(split['test'])):raise ValueError('Identity split overlap')
    manifest={'parent':str(parent),'parent_sha256':registry['checkpoint_sha256'],'pipeline':pipeline,'splits':{},'real_training':[],'real_development':[],'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}}
    for group,count in [('train',o.train_count),('validation',o.validation_count)]:
        folder=out/group;folder.mkdir();manifest['splits'][group]=[]
        for i in range(count):
            source=split[group][i];original=load_skin(source);seed=(10307000 if group=='train' else 20307000)+i*7919
            item=make_structured_skin(original,seed) if i%2==0 else make_joint_skin(original,seed)
            uv=item['uv'][None].cuda();images=[];fg=[]
            for view in ('front_left','back_left'):
                image,t=build_dense_parser_batch(uv,renderer,view);images.append(image);fg.append(t['foreground'][:,0])
            images=torch.cat(images);fg=torch.cat(fg)
            if i%3==1:images=appearance_augment(images,fg[:,None]>.5,strength=1.4)
            with torch.no_grad():result=run_pipeline(model,renderer,images,pipeline,complete=True,foreground_probability=fg)
            path=folder/f'{i:05d}.pt';save_case(path,result,uv,item['labels'][None],item.get('symmetric',False),{'source':source,'seed':seed,'kind':'rendered_ground_truth','split':group})
            manifest['splits'][group].append(str(path))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'state':'preparing','split':group,'done':i+1,'total':count}),flush=True)
    cases=json.loads((root/'regression/v102_development_cases.json').read_text())
    for folder in sorted((root/'output_history/v101_retrained_20260906').iterdir()):
        if (folder/'diagnostics.pt').is_file():
            m=json.loads((folder/'manifest.json').read_text());cases.append({'name':'old_'+Path(m['input']).stem,'directory':str(folder),'input':m['input']})
    if o.quick:cases=[c for c in cases if c['name']=='beard']
    real=out/'real';real.mkdir()
    review=json.loads((root/'regression/v102_uv_symmetry_review.json').read_text())
    for c in cases:
        d=torch.load(Path(c['directory'])/'diagnostics.pt',weights_only=False,map_location='cpu');images=d['images'].cuda()
        fg=d['foreground'].cuda() if c['name'] in ('beard','nose','crown') else cached_real_foreground(images,c['input'],pipeline)
        with torch.no_grad():result=run_pipeline(model,renderer,images,pipeline,complete=True,foreground_probability=fg)
        target=result['uv'].clone();labels=torch.full((1,64,64),-100,dtype=torch.long)
        metadata={'input':c['input'],'input_sha256':hashlib.sha256(Path(c['input']).read_bytes()).hexdigest(),'kind':'unlabeled_development','name':c['name']}
        if c['name']=='beard':
            if metadata['input_sha256']!=review['input_sha256']:raise ValueError('Anchor input changed')
            expected=torch.from_numpy(np.array(Image.open(root.parent/review['expected_uv']).convert('RGBA')).copy()).permute(2,0,1).float()/255
            beard=np.array(Image.open(root.parent/review['beard_scope']).convert('L'))>127
            hair=np.array(Image.open(root.parent/review['hair_scope']).convert('L'))>127
            support=np.array(Image.open(root/'runs/v102_uv_symmetry_20260906/beard_support.png').convert('L'))>127
            # Use the corrected support, including its explicit mirror closure.
            from SkingToolkit.dense_uv_parser.uv_topology import build_simple_uv_topology
            topology=build_simple_uv_topology();mirror=topology.mirrored_texel.numpy().reshape(-1)
            support=(support.reshape(-1)|support.reshape(-1)[mirror]).reshape(64,64)
            mask=torch.from_numpy(beard|hair).to(target.device)
            target[0,:,mask]=expected.to(target.device)[:,mask]
            labels[0][torch.from_numpy(beard&~support&(topology.layer.numpy()==0))]=1
            labels[0][torch.from_numpy(beard&~support&(topology.layer.numpy()==1))]=0
            labels[0][torch.from_numpy(beard&support&(topology.layer.numpy()==0))]=3
            labels[0][torch.from_numpy(beard&support&(topology.layer.numpy()==1))]=5
            metadata.update(kind='explicit_partial_uv_training_annotation',annotation=str(root/'regression/v102_uv_symmetry_review.json'),annotation_sha256=hashlib.sha256((root/'regression/v102_uv_symmetry_review.json').read_bytes()).hexdigest(),expected_uv_sha256=hashlib.sha256((root.parent/review['expected_uv']).read_bytes()).hexdigest(),warning='This identity is now training data and cannot demonstrate generalization.')
            manifest['real_training'].append(str(real/(c['name']+'.pt')))
        else:manifest['real_development'].append(str(real/(c['name']+'.pt')))
        path=real/(c['name']+'.pt');save_case(path,result,target,labels,c['name']=='beard',metadata)
        if c['name']=='beard':torch.save({'images':images.cpu(),'foreground':fg.cpu(),'input':c['input']},out/'anchor_input.pt')
        print(json.dumps({'state':'preparing_real','name':c['name'],'training':c['name']=='beard'}),flush=True)
    manifest['complete']=True
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('cache_complete',flush=True)


if __name__=='__main__':main()
