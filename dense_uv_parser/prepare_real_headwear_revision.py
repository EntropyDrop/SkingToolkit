"""Turn two explicit user headwear corrections into partial training annotations."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,'dense_uv_parser')
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image

root=Path('dense_uv_parser').resolve();run=root/'runs/v103_semantic_revision_20260907'
source=run/'cache_crossed';out=run/'cache_real_headwear';out.mkdir(exist_ok=False)
manifest=json.loads((source/'manifest.json').read_text());assert manifest['complete']
manifest['complete']=False;manifest['parent_cache']=str(source)
manifest['revision']='explicit_real_headwear_training_20260907'
annotations=root/'regression/v103_headwear_training_20260907';annotations.mkdir(exist_ok=False)
records=[];selected={'crown','old_PU418PLGERMFC487_edited'};development=[]
for path in manifest['real_development']:
    row=torch.load(path,map_location='cpu',weights_only=False);name=row['metadata']['name']
    if name not in selected:development.append(path);continue
    metadata=dict(row['metadata']);assert hashlib.sha256(Path(metadata['input']).read_bytes()).hexdigest()==metadata['input_sha256']
    target=row['base'].clone();labels=torch.full((1,64,64),-100,dtype=torch.long)
    scope=np.zeros((64,64),bool);uv=np.array(tensor_to_rgba_image(target[0]));rgb=uv[:,:,:3].astype(float)/255
    if name=='crown':
        # User identified these blue crown ends as continuing onto the outer top.
        # The previous accepted v102 correction provides that reviewed geometry.
        scope[:8,40:48]=True
        blue=(rgb[:,:,2]>.25)&(rgb[:,:,2]-rgb[:,:,0]>.2)&(rgb[:,:,2]-rgb[:,:,1]>.08)
        positive=scope&blue&(uv[:,:,3]>127)
        assert int(positive.sum())==12
        labels[0][torch.from_numpy(positive)]=13
        labels[0][torch.from_numpy(scope&(uv[:,:,3]==0))]=0
        description='Preserve the 12 reviewed blue outer-top crown cells and transparent gaps. Other colours in this region remain semantically unlabelled.'
    else:
        scope[11,32:64]=True
        assert bool((uv[11,32:64,3]==255).all())
        labels[0,11,32:64]=12
        description='The user explicitly requires the brim to continue around all four head faces: outer row 11, columns 32 through 63.'
    mask_path=annotations/(name+'_scope.png');target_path=annotations/(name+'_copy_target.png')
    Image.fromarray(scope.astype(np.uint8)*255).save(mask_path);tensor_to_rgba_image(target[0]).save(target_path)
    metadata.update(kind='explicit_partial_uv_training_annotation',training_role=True,geometry_annotation=str(mask_path),copy_target=str(target_path),annotation_description=description,unlabelled_policy='Outside reviewed scope, the previous v102 UV is a copy-consistency target, not newly claimed human ground truth.',warning='This identity has moved into training; it is no longer a gradient-free development example.')
    row.update(target=target,labels=labels,metadata=metadata)
    output=out/(name+'.pt');torch.save(row,output);manifest['real_training'].append(str(output))
    records.append({**metadata,'scope_sha256':hashlib.sha256(mask_path.read_bytes()).hexdigest(),'target_sha256':hashlib.sha256(target_path.read_bytes()).hexdigest(),'source_cache':path,'scoped_texels':int(scope.sum())})
manifest['real_development']=development
assert len(manifest['real_training'])==3 and len(development)==14
manifest['real_training_policy']='Three real training identities (beard, blue crown, ring brim); fourteen remaining real cases have no gradient use.'
manifest['real_headwear_annotations']=str(annotations/'annotations.json')
(annotations/'annotations.json').write_text(json.dumps({'scope':'Partial user-directed geometry labels plus explicit copy-consistency targets. No file-name/identity lookup is used at inference.','items':records},indent=2)+'\n')
shutil.copy2(source/'anchor_input.pt',out/'anchor_input.pt')
manifest['complete']=True;(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'training_synthetic':len(manifest['splits']['train']),'validation_synthetic':len(manifest['splits']['validation']),'real_training':len(manifest['real_training']),'real_development':len(manifest['real_development']),'annotations':str(annotations)}))
