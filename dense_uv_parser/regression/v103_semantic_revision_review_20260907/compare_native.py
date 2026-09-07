"""Compare authored and native-texture validation separately for all candidates."""
import sys,json,hashlib
from pathlib import Path
import torch
sys.path.insert(0,"dense_uv_parser")
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.final_head_uv import FinalHeadUVDecoder,evaluation_numerics
from SkingToolkit.dense_uv_parser.train_final_head_uv import evaluate,load
torch.set_num_threads(4);evaluation_numerics()
root=Path("dense_uv_parser/runs/v103_semantic_revision_20260907")
cache=root/"cache_native/manifest.json"
rows=[load(p) for p in json.loads(cache.read_text())["splits"]["validation"]]
groups={"authored_64":[r for r in rows if r["metadata"].get("family")!="native_texture"],"native_texture_16":[r for r in rows if r["metadata"].get("family")=="native_texture"]}
assert len(groups["authored_64"])==64 and len(groups["native_texture_16"])==16
report={"cache_manifest_sha256":hashlib.sha256(cache.read_bytes()).hexdigest(),"scope":"Source-disjoint synthetic validation; original source textures are rendered inputs. No real training identity included.","models":{}}
for name,path in [("previous_failed_v103",Path("dense_uv_parser/runs/v103_final_uv_retrain_20260907/training_completed/step_6000.pt")),("before_native_replay",root/"training_real/step_3000.pt"),("native_replay",root/"training_native/step_3000.pt")]:
    ck=load(path);model=FinalHeadUVDecoder(**ck["final_head_uv_config"]).cuda().eval();model.load_state_dict(ck["final_head_uv_state"])
    report["models"][name]={"checkpoint":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"groups":{n:evaluate(model,rs) for n,rs in groups.items()}}
    del model,ck
(root/"native_validation_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
