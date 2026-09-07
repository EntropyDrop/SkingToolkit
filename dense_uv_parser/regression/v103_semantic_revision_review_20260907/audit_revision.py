"""Review cached real outputs of the recovered final-UV training; no inference edits."""
import sys,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,"dense_uv_parser")
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image

import argparse
p=argparse.ArgumentParser();p.add_argument("--run",default="training");o=p.parse_args()
root=Path("dense_uv_parser/runs/v103_semantic_revision_20260907")
run=root/o.run
history=run
manifest=json.loads((root/"cache/manifest.json").read_text())
load=lambda p:torch.load(p,map_location="cpu",weights_only=False)
base={}
for path in manifest["real_training"]+manifest["real_development"]:
    row=load(path);name=row["metadata"]["name"]
    base[name]=np.array(tensor_to_rgba_image(row["base"][0]))
    folder=run/"review_baseline"/name;folder.mkdir(parents=True,exist_ok=True)
    Image.fromarray(base[name]).save(folder/"uv.png")

def landmarks(name,u):
    rgb=u[:,:,:3].astype(float)/255
    if name=="nose":return {"nose_outer_alpha":int(u[13,43,3])}
    if name=="crown":
        blue=(rgb[:,:,2]>.25)&(rgb[:,:,2]-rgb[:,:,0]>.2)&(rgb[:,:,2]-rgb[:,:,1]>.08)
        return {"inner_cap_blue":int(blue[:8,8:16].sum()),"outer_cap_blue":int((blue[:8,40:48]&(u[:8,40:48,3]>127)).sum())}
    if name=="old_img46":return {"crown_top_cells":int((u[:8,40:48,3]>127).sum())}
    if name=="old_img27":
        green=rgb[:,:,1]-np.maximum(rgb[:,:,0],rgb[:,:,2]);mask=np.zeros((64,64),bool);mask[:8,8:16]=1;mask[8:16,:8]=1;mask[8:16,16:32]=1
        return {"inner_strong_green":int(((green>.15)&mask).sum())}
    if "TWRLRR" in name:return {"forehead_outer_alpha":int(u[10,41,3])}
    if "PU418" in name:
        red=rgb[:,:,0]-np.maximum(rgb[:,:,1],rgb[:,:,2])
        return {"brim_each_face":[int((u[11,x:x+8,3]>127).sum()) for x in [32,40,48,56]],"upper_hat_outer_red":int(((red[:12,32:]>.15)&(u[:12,32:,3]>127)).sum())}
    return {}

config=json.loads((run/"config.json").read_text())
trained=[x["name"] for x in config["real_training_identities"]]
development=[x["name"] for x in config["real_development_identities"]]
report={"scope":"Training identities measure fitting only. Remaining real cases are repeatedly used development cases, not independent held-out tests. Landmark coordinates are evaluation annotations only.","training_identities":trained,"development_identities":development,"checkpoints":{}}
for step in (200,1000,2000,3000,4000,5000,6000):
    folder=run if (run/f"real_{step}.json").exists() else history
    path=folder/f"real_{step}.json"
    if not path.exists():continue
    records=json.loads(path.read_text());rows={}
    for name,b in base.items():
        p=np.array(Image.open(folder/f"real_{step}"/name/"uv.png").convert("RGBA"))
        rows[name]={"base":landmarks(name,b),"candidate":landmarks(name,p),"body_exact":bool(np.array_equal(b[16:],p[16:])),"role":records[name]["role"]}
    c=lambda name:rows[name]["candidate"]
    checks={
        "beard_training_fit":records["beard"]["annotated_uv_review"]["passed"],
        "body_exact":all(r["body_exact"] for r in rows.values()),
        "nose_inner":c("nose")["nose_outer_alpha"]==0,
        "crown_cap_inner_clear":c("crown")["inner_cap_blue"]==0,
        "crown_cap_extension_retained":c("crown")["outer_cap_blue"]>=rows["crown"]["base"]["outer_cap_blue"],
        "gold_crown_top_open":c("old_img46")["crown_top_cells"]==0,
        "phone_no_strong_green_leak":c("old_img27")["inner_strong_green"]<=rows["old_img27"]["base"]["inner_strong_green"],
        "glasses_forehead_clear":c("old_TWRLRRTHQP2UV368_edited")["forehead_outer_alpha"]==0,
        "brim_continuous":c("old_PU418PLGERMFC487_edited")["brim_each_face"]==[8]*4,
        "hat_band_inner":c("old_PU418PLGERMFC487_edited")["upper_hat_outer_red"]==0,
    }
    report["checkpoints"][str(step)]={"checks":checks,"beard":records["beard"]["annotated_uv_review"],"cases":rows,"synthetic":json.loads((folder/f"evaluation_{step}.json").read_text()),"roundtrip":json.loads((folder/f"roundtrip_{step}.json").read_text())}
    print(json.dumps({"step":step,"checks":checks,"beard":records["beard"]["annotated_uv_review"]}))
report["status"]=json.loads((run/"status.json").read_text())
(run/"semantic_revision_review.json").write_text(json.dumps(report,indent=2)+"\n")
