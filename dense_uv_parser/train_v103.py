"""Fine-tune v102 head semantics and face identity with four-view UV losses."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import torch
from torch.utils.data import DataLoader
from SkingToolkit.dense_uv_parser.infer import load_parser
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment, write_json
from SkingToolkit.dense_uv_parser.train_v102 import joint_loss, segmentation_loss, real_review
from SkingToolkit.dense_uv_parser.v103_data import StructuredHeadDataset
from SkingToolkit.dense_uv_parser.v103_losses import VIEWS, render_structured_batch, structured_uv_loss, pool_uv
from SkingToolkit.renderer import DifferentiableRenderer


@torch.no_grad()
def evaluate(model,renderer,dataset):
    model.eval()
    total,correct = torch.zeros(14,device="cuda"),torch.zeros(14,device="cuda")
    losses=[];surface_correct=0;surface_count=0
    for batch in DataLoader(dataset,batch_size=4,num_workers=2):
        batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
        images,truth,faces,index,valid=render_structured_batch(batch,renderer)
        logits,_,surface=model.predict_joint_head_semantics(images,images[:,3]>.5,return_surface=True)
        loss,metrics=structured_uv_loss(logits,index,valid,batch["labels"],batch["mode"],batch["symmetric"])
        losses.append(float(loss))
        pooled,counts=pool_uv(logits.float().softmax(1),index,valid)
        target=batch["labels"].flatten(1);pred=pooled.argmax(1)
        selected=(counts>0)&(batch["mode"][:,None]==0)&(target>0)
        total+=torch.bincount(target[selected],minlength=14)
        correct+=torch.bincount(target[selected&(pred==target)],minlength=14)
        surface_correct+=int(((surface.argmax(1)==faces)&valid).sum());surface_count+=int(valid.sum())
    return {"uv_loss":sum(losses)/max(len(losses),1),"visible_uv_class_recall":(correct/total.clamp_min(1)).tolist(),"visible_uv_class_counts":total.tolist(),"surface_accuracy":surface_correct/max(surface_count,1),"scope":"Synthetic validation on train-disjoint source identities, not real generalization evidence."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--steps",type=int,default=6000)
    p.add_argument("--batch-size",type=int,default=4)
    p.add_argument("--lr",type=float,default=5e-5)
    p.add_argument("--eval-every",type=int,default=1000)
    p.add_argument("--first-eval",type=int,default=200)
    p.add_argument("--validation-count",type=int,default=128)
    p.add_argument("--skip-real",action="store_true")
    o=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1030906)
    torch.backends.cudnn.benchmark=True
    root=Path(__file__).resolve().parent;out=o.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    registry=json.loads((root/"v102_release.json").read_text())
    parent=root.parent/registry["checkpoint"]
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=registry["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint hash mismatch")
    checkpoint=torch.load(parent,map_location="cpu",weights_only=False)
    model,args=load_parser(parent,torch.device("cuda"))
    model.requires_grad_(False);model.head_semantics_head.requires_grad_(True)
    renderer=DifferentiableRenderer(args["mappings_dir"]).cuda()
    split_path=root/"runs/dense_uv_parser_v101_retrain_20260906/source_splits.json"
    split=json.loads(split_path.read_text())
    if set(split["train"]) & (set(split["validation"])|set(split["test"])):
        raise ValueError("Training identity split overlap")
    review=json.loads((root/"regression/v102_uv_symmetry_review.json").read_text())
    excluded={review[k] for k in ("input","reference","corrected_result")}
    excluded|={str(root.parent/review["expected_uv"])}
    if set(split["train"]) & excluded:
        raise ValueError("User-reviewed example in gradient training")
    train=StructuredHeadDataset(split["train"],32768,103090600)
    val=StructuredHeadDataset(split["validation"],o.validation_count,103091700)
    pipeline=json.loads((root.parent/registry["pipeline"]).read_text())
    write_json(out/"pipeline.json",pipeline)
    manifest={"version":"v103","state":"candidate_not_released","parent":str(parent),"parent_sha256":registry["checkpoint_sha256"],"git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),"options":{k:str(v) if isinstance(v,Path) else v for k,v in vars(o).items()},"views":VIEWS,"trainable":"head_semantics_head including its surface classifier; all other parent tensors frozen","trainable_parameters":sum(x.numel() for x in model.parameters() if x.requires_grad),"source_splits_sha256":hashlib.sha256(split_path.read_bytes()).hexdigest(),"split_counts":{k:len(v) for k,v in split.items()},"excluded_review_assets":sorted(excluded),"data":"50% authored structured beard/fringe cases, 50% legacy joint/ownership/headwear replay. No user image or corrected UV used for gradients.","loss":"pixel joint semantics + 0.4 visible face loss + 0.8 cell-balanced UV loss; UV loss includes GT-conditioned mirror and mixed-layer terms","acceptance":"Real automatic UV review and prior accessory regressions required before release. Training completion is not acceptance.","source_sha256":{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob("*.py")}}
    write_json(out/"config.json",manifest)
    snapshot=out/"source";snapshot.mkdir()
    for f in root.glob("*.py"):shutil.copy2(f,snapshot/f.name)
    opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=o.lr,weight_decay=1e-4)
    step=0;epoch=0;start=time.time()
    def status(state,**extra):
        row={"state":state,"version":"v103","pid":os.getpid(),"step":step,"total_steps":o.steps,"elapsed_seconds":time.time()-start,**extra}
        write_json(out/"status.json",row);print(json.dumps(row),flush=True)
    def validate():
        status("validation")
        metrics=evaluate(model,renderer,val);write_json(out/f"evaluation_{step}.json",metrics)
        state=model.state_dict()
        for name,value in checkpoint["model"].items():
            if name.startswith("head_semantics_head."):continue
            if not torch.equal(value,state[name].cpu()):raise RuntimeError("Frozen tensor changed: "+name)
        payload={**{k:v for k,v in checkpoint.items() if k not in ("model","optimizer")},"model":state,"step":step,"v103_manifest":manifest,"v103_metrics":metrics,"inference_pipeline":pipeline}
        path=out/f"step_{step}.pt";torch.save(payload,out/"checkpoint.tmp");(out/"checkpoint.tmp").replace(path)
        torch.save({"optimizer":opt.state_dict(),"step":step,"epoch":epoch,"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all()},out/"optimizer_latest.pt")
        write_json(out/"latest.json",{"step":step,"checkpoint":str(path),"non_head_parent_tensors_unchanged":True})
        if not o.skip_real:
            status("real_uv_review");real_review(model,renderer,pipeline,out,step)
        status("checkpoint_saved",checkpoint=str(path),validation=metrics)
    try:
        status("baseline_validation")
        write_json(out/"baseline_validation.json",evaluate(model,renderer,val))
        while step<o.steps:
            train.epoch=epoch
            for batch in DataLoader(train,batch_size=o.batch_size,shuffle=True,num_workers=4,pin_memory=True):
                if step>=o.steps:break
                model.eval();model.head_semantics_head.train()
                batch={k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
                with torch.no_grad():
                    images,truth,faces,index,valid=render_structured_batch(batch,renderer)
                    fg=images[:,3]>.5
                    if step%2==0:images=appearance_augment(images,fg[:,None],strength=1.4)
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    logits,presence,surface=model.predict_joint_head_semantics(images,fg,return_surface=True)
                    y0,y1,x0,x1=head_bounds(*truth.shape[-2:])
                    pixel=joint_loss(logits[:,:,y0:y1,x0:x1],presence,truth[:,y0:y1,x0:x1],batch["mode"].repeat_interleave(4))
                    face=segmentation_loss(surface[:,:,y0:y1,x0:x1],faces[:,y0:y1,x0:x1],[1,2,2,2,2,2,2])
                    uv,uv_metrics=structured_uv_loss(logits,index,valid,batch["labels"],batch["mode"],batch["symmetric"])
                    loss=pixel+.4*face+.8*uv
                if not torch.isfinite(loss):raise RuntimeError("Non-finite loss")
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.head_semantics_head.parameters(),1.)
                if not torch.isfinite(grad):raise RuntimeError("Non-finite gradient")
                opt.step();step+=1
                for group in opt.param_groups:group["lr"]=o.lr*(.1+.9*(1+math.cos(math.pi*step/o.steps))/2)
                if step==1 or step%50==0:
                    status("training",loss=float(loss.detach()),pixel_loss=float(pixel.detach()),face_loss=float(face.detach()),**{k:float(v) for k,v in uv_metrics.items()},peak_gpu_memory_mib=torch.cuda.max_memory_allocated()/2**20)
                if step==o.first_eval or step%o.eval_every==0 or step==o.steps:validate()
            epoch+=1
        status("complete",non_head_parent_tensors_unchanged=True,release_promoted=False)
    except BaseException as e:
        status("failed",error=repr(e));raise


if __name__=="__main__":main()
