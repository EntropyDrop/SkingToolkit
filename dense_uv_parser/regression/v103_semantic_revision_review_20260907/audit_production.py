"""Audit exported original-image predictions without modifying any output."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,"dense_uv_parser")
from run_local import bind_checkout
bind_checkout()
from SkingToolkit.dense_uv_parser.uv_reference_repair import evaluate_annotated_review

root=Path("dense_uv_parser/runs/v103_semantic_revision_20260907")
p=argparse.ArgumentParser();p.add_argument("--run",default="training_real");p.add_argument("--output-dir",default="dense_uv_parser/output_history/v103_semantic_revision_20260907");p.add_argument("--report",default="production_review.json");p.add_argument("--checkpoint",type=Path);p.add_argument("--refit",action="store_true");p.add_argument("--repeat-output-dir",type=Path);o=p.parse_args()
if o.refit and not o.repeat_output_dir:p.error('Refit review requires an independent full-inference repeat')
run=root/o.run
out=Path(o.output_dir)
config=json.loads((run/"config.json").read_text())
cached=json.loads((run/"semantic_revision_review.json").read_text())["checkpoints"]["3000"]
image=lambda p:np.array(Image.open(p).convert("RGBA"))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checkpoint=o.checkpoint or run/"step_3000.pt"
files={json.loads(p.read_text())["input_sha256"]:p for p in out.glob("*/manifest.json")}
identities=config["real_training_identities"]+config["real_development_identities"]
assert len(files)==len(identities)==17
report={"scope":"Three real training identities measure fit only. Fourteen repeatedly used real development identities have no gradient use and are not independent held-out tests. Foreground probabilities are checksum-bound trained-model cache outputs; parser inference is fresh from original raw images.","checkpoint":str(checkpoint.resolve()),"checkpoint_sha256":sha(checkpoint),"cases":{},"release_promoted":False}
fresh={}
for meta in identities:
    name=meta["name"];mp=files[meta["input_sha256"]];m=json.loads(mp.read_text());folder=mp.parent
    assert m["complete"] and m["checkpoint_sha256"]==report["checkpoint_sha256"]
    assert sha(m["input"])==meta["input_sha256"]
    assert m["foreground_provider"]["adaptation_sha256"]=="8585a698d38969e6e7562e9c7d8c5f0b5e934e8f0811cf5b8678fcf05734e732"
    assert sha(m["foreground_probability"]["path"])==m["foreground_probability"]["sha256"]
    p=image(folder/"pred_uv.png");c=image(run/"real_3000"/name/"uv.png");base=image(run/"review_baseline"/name/"uv.png")
    fresh[name]=p
    visible=(p[:,:,3]>0)|(c[:,:,3]>0)
    delta=np.abs(p[:,:,:3].astype(int)-c[:,:,:3].astype(int))
    row={"input":m["input"],"output":str(folder.resolve()),"role":cached["cases"][name]["role"],"png_sha256":sha(folder/"pred_uv.png"),"alpha_exact_to_cached":bool(np.array_equal(p[:,:,3],c[:,:,3])),"body_exact_to_cached_parent":bool(np.array_equal(p[16:],base[16:])),"visible_rgb_max_byte_delta_to_cached":int(delta[visible].max()),"visible_rgb_changed_texels_to_cached":int((delta.max(2)[visible]>0).sum())}
    if name=="beard":
        row["beard"]=evaluate_annotated_review(p,m["input"])
        annotation=json.loads(Path("dense_uv_parser/regression/v102_uv_symmetry_review.json").read_text())
        scope=np.array(Image.open(annotation["beard_scope"]).convert("L"))>127
        expected=image(annotation["expected_uv"])
        # Only annotation-confirmed co-visible matching material pairs.
        eligible=scope[:,:32]&scope[:,32:]&(expected[:,:32,3]>0)&(expected[:,32:,3]>0)&np.all(expected[:,:32,:3]==expected[:,32:,:3],axis=2)
        d=np.abs(p[:,:32,:3].astype(int)-p[:,32:,:3].astype(int))
        row["layer_alignment"]={"annotated_pairs":int(eligible.sum()),"support_mismatches":int(((p[:,:32,3]==0)|(p[:,32:,3]==0))[eligible].sum()),"rgb_mismatches":int((d.max(2)[eligible]>0).sum()),"rgb_max_byte_difference":int(d[eligible].max())}
    report["cases"][name]=row
report["cached_landmark_checks"]=cached["checks"]
rgb=fresh["crown"][:,:,:3].astype(float)/255
blue=(rgb[:,:,2]>.25)&(rgb[:,:,2]-rgb[:,:,0]>.2)&(rgb[:,:,2]-rgb[:,:,1]>.08)
hat=fresh["old_PU418PLGERMFC487_edited"];hrgb=hat[:,:,:3].astype(float)/255
phone=fresh["old_img27"][:,:,:3].astype(float)/255;green=phone[:,:,1]-np.maximum(phone[:,:,0],phone[:,:,2])
phone_scope=np.zeros((64,64),bool);phone_scope[:8,8:16]=1;phone_scope[8:16,:8]=1;phone_scope[8:16,16:32]=1
report["fresh_landmarks"]={"nose_outer_alpha":int(fresh["nose"][13,43,3]),"crown_inner_cap_blue":int(blue[:8,8:16].sum()),"crown_outer_cap_blue":int((blue[:8,40:48]&(fresh["crown"][:8,40:48,3]>127)).sum()),"gold_crown_top_cells":int((fresh["old_img46"][:8,40:48,3]>127).sum()),"phone_inner_strong_green":int(((green>.15)&phone_scope).sum()),"glasses_forehead_outer_alpha":int(fresh["old_TWRLRRTHQP2UV368_edited"][10,41,3]),"brim_each_face":[int((hat[11,x:x+8,3]>127).sum()) for x in [32,40,48,56]],"upper_hat_outer_red":int((((hrgb[:,:,0]-np.maximum(hrgb[:,:,1],hrgb[:,:,2]))[:12,32:]>.15)&(hat[:12,32:,3]>127)).sum())}
f=report["fresh_landmarks"]
fresh_pass=f["nose_outer_alpha"]==0 and f["crown_inner_cap_blue"]==0 and f["crown_outer_cap_blue"]>=cached["cases"]["crown"]["base"]["outer_cap_blue"] and f["gold_crown_top_cells"]==0 and f["phone_inner_strong_green"]<=cached["cases"]["old_img27"]["base"]["inner_strong_green"] and f["glasses_forehead_outer_alpha"]==0 and f["brim_each_face"]==[8]*4 and f["upper_hat_outer_red"]==0
report["checks"]={"all_alpha_exact_to_cached":all(x["alpha_exact_to_cached"] for x in report["cases"].values()),"all_body_exact_to_cached_parent":all(x["body_exact_to_cached_parent"] for x in report["cases"].values()),"all_visible_rgb_within_one_byte_of_cached":all(x["visible_rgb_max_byte_delta_to_cached"]<=1 for x in report["cases"].values()),"cached_landmarks_pass":all(cached["checks"].values()),"beard_symmetry_and_hair":report["cases"]["beard"]["beard"]["passed"],"beard_layer_alignment":all(report["cases"]["beard"]["layer_alignment"][k]==0 for k in ("support_mismatches","rgb_mismatches"))}
report["checks"]["fresh_landmarks_pass"]=fresh_pass
if o.refit:
    report["raw_decoder_rgb_comparison_within_one_byte"]=report["checks"].pop("all_visible_rgb_within_one_byte_of_cached")
    report["raw_decoder_rgb_comparison_scope"]="Informational only: this explicit pipeline adds source colour fitting after decoding, so raw decoder RGB is not the deployment output. The same 1-byte budget is checked against a full-pipeline repeat instead."
    repeat={json.loads(p.read_text())["input_sha256"]:p for p in o.repeat_output_dir.glob("*/manifest.json")}
    assert len(repeat)==17
    for meta in identities:
        name=meta["name"];row=report["cases"][name];a=fresh[name];mp=repeat[meta["input_sha256"]];m=json.loads(mp.read_text())
        assert m["complete"] and m["checkpoint_sha256"]==report["checkpoint_sha256"]
        first=json.loads((Path(row["output"])/"manifest.json").read_text())
        assert m["source_sha256"]==first["source_sha256"] and m["pipeline"]==first["pipeline"] and m["foreground_probability"]["sha256"]==first["foreground_probability"]["sha256"]
        b=image(mp.parent/"pred_uv.png");visible=(a[:,:,3]>0)|(b[:,:,3]>0)
        row["independent_repeat"]={"output":str(mp.parent.resolve()),"alpha_exact":bool(np.array_equal(a[:,:,3],b[:,:,3])),"body_exact":bool(np.array_equal(a[16:],b[16:])),"visible_rgb_max_byte_delta":int(np.abs(a[:,:,:3].astype(int)-b[:,:,:3].astype(int))[visible].max())}
        row["final_material_fit"]=json.loads((Path(row["output"])/"final_head_material_refit.json").read_text())
    report["checks"].update(repeat_alpha_and_body_exact=all(r["independent_repeat"]["alpha_exact"] and r["independent_repeat"]["body_exact"] for r in report["cases"].values()),repeat_visible_rgb_within_one_byte=all(r["independent_repeat"]["visible_rgb_max_byte_delta"]<=1 for r in report["cases"].values()),refit_preserves_geometry_body_and_reconstruction=all(r["final_material_fit"]["alpha_exact"] and r["final_material_fit"]["body_exact"] and r["final_material_fit"]["source_mse_final"]<=r["final_material_fit"]["source_mse_before"] for r in report["cases"].values()))
report["passed"]=all(report["checks"].values())
report["passed_scope"]="Existing scoped checks only. Whole-image visual review is a separate required assessment."
(root/o.report).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps({"passed":report["passed"],"checks":report["checks"],"beard":report["cases"]["beard"],"rgb_deltas":{n:x["visible_rgb_max_byte_delta_to_cached"] for n,x in report["cases"].items()}},indent=2))
