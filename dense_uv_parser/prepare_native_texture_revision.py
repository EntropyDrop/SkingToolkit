"""Preserve source skin head patterns instead of replacing every head procedurally."""
import json,hashlib,sys,time,shutil
from pathlib import Path
import torch
sys.path.insert(0,"dense_uv_parser")
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import evaluation_numerics
from SkingToolkit.dense_uv_parser.prepare_final_head_uv import save_case
from SkingToolkit.dense_uv_parser.infer import load_parser
from SkingToolkit.dense_uv_parser.accessory_pipeline import run_pipeline
from SkingToolkit.dense_uv_parser.skin_dataset import load_skin
from SkingToolkit.dense_uv_parser.utils import build_dense_parser_batch
from SkingToolkit.renderer import DifferentiableRenderer

torch.set_num_threads(4);evaluation_numerics()
root=Path("dense_uv_parser/runs/v103_semantic_revision_20260907").resolve()
previous=root/"cache_real_headwear";out=root/"cache_native";out.mkdir(exist_ok=True)
manifest=json.loads((previous/"manifest.json").read_text());manifest["complete"]=False
manifest.update(revision="native_source_texture_replay",parent_cache=str(previous),native_texture_policy="Keep original RGBA skin unchanged. Semantic labels unknown; no guessed hair/beard labels. These rendered source skins are synthetic inputs, not real edited development images.")
source_split=json.loads(Path("dense_uv_parser/runs/dense_uv_parser_v101_retrain_20260906/source_splits.json").read_text())
assert hashlib.sha256(Path(manifest["parent"]).read_bytes()).hexdigest()==manifest["parent_sha256"]
model,args=load_parser(manifest["parent"],torch.device("cuda"));renderer=DifferentiableRenderer(args["mappings_dir"]).cuda()
start=time.time()
for group,offset,count in [("train",256,64),("validation",64,16)]:
    folder=out/group;folder.mkdir(exist_ok=True)
    for i,source in enumerate(source_split[group][offset:offset+count]):
        path=folder/f"{i:05d}.pt"
        if path.exists():
            row=torch.load(path,map_location="cpu",weights_only=False);assert row["metadata"]["source"]==source
        else:
            torch.manual_seed(70307001+i+(10000 if group=="validation" else 0))
            uv=load_skin(source)[None].cuda();images=[];foreground=[]
            for view in ("front_left","back_left"):
                image,target=build_dense_parser_batch(uv,renderer,view);images.append(image);foreground.append(target["foreground"][:,0])
            with torch.no_grad():result=run_pipeline(model,renderer,torch.cat(images),manifest["pipeline"],complete=True,foreground_probability=torch.cat(foreground))
            temp=path.with_suffix(".tmp")
            save_case(temp,result,uv,torch.full((1,64,64),-100,dtype=torch.long),False,{"source":source,"source_sha256":hashlib.sha256(Path(source).read_bytes()).hexdigest(),"family":"native_texture","kind":"rendered_ground_truth","split":group,"unaltered_source_uv":True})
            temp.replace(path)
        manifest["splits"][group].append(str(path))
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"state":"preparing","split":group,"done":i+1,"total":count,"elapsed_seconds":time.time()-start}),flush=True)
sources={g:{torch.load(p,map_location="cpu",weights_only=False)["metadata"]["source"] for p in manifest["splits"][g]} for g in ("train","validation")}
assert len(sources["train"])==320 and len(sources["validation"])==80 and not sources["train"]&sources["validation"]
shutil.copy2(previous/"anchor_input.pt",out/"anchor_input.pt")
manifest["complete"]=True
(out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"state":"complete","elapsed_seconds":time.time()-start}),flush=True)
