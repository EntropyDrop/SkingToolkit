"""Evaluate both final-UV checkpoints against exactly the same validation cache."""
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
cache=root/"cache_real_headwear/manifest.json"
rows=[load(p) for p in json.loads(cache.read_text())["splits"]["validation"]]
report={"cache_manifest_sha256":hashlib.sha256(cache.read_bytes()).hexdigest(),"scope":"Same 64 train-disjoint synthetic validation source identities for both candidates; no real training identity in this score.","models":{}}
for name,path in [("previous_failed_v103",Path("dense_uv_parser/runs/v103_final_uv_retrain_20260907/training_completed/step_6000.pt")),("improved_v103",root/"training_real/step_3000.pt")]:
    ck=load(path);model=FinalHeadUVDecoder(**ck["final_head_uv_config"]).cuda().eval();model.load_state_dict(ck["final_head_uv_state"])
    report["models"][name]={"checkpoint":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"evaluation":evaluate(model,rows)}
    del model,ck
parent=load(root/"training_real/step_3000.pt");base=load("dense_uv_parser/runs/v102_alignment_release_20260906/parser.pt")
report["all_parent_model_tensors_exact"]=parent["model"].keys()==base["model"].keys() and all(torch.equal(v,parent["model"][k]) for k,v in base["model"].items())
(root/"same_cache_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
