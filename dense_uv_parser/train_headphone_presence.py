"""Train an object-level headphone gate using spatial semantic evidence only."""
from pathlib import Path
import argparse,inspect,json,hashlib,time,os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from SkingToolkit.dense_uv_parser.model import DenseUVParserNet
from SkingToolkit.dense_uv_parser.semantic import attach_semantic_runtime
from SkingToolkit.dense_uv_parser.ownership_data import OwnershipDataset,render_ownership
from SkingToolkit.dense_uv_parser.semantic_generalization import appearance_augment,write_json
from SkingToolkit.dense_uv_parser.accessory_pipeline import cached_real_foreground
from SkingToolkit.dense_uv_parser.accessories import head_bounds
from SkingToolkit.renderer import DifferentiableRenderer


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--steps',type=int,default=1200);opt=p.parse_args();out=opt.output_dir;out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).parent;torch.set_num_threads(4);torch.manual_seed(1010602)
    source=root/'runs/v101_ownership_main_20260906/step_3000.pt';ckpt=torch.load(source,map_location='cpu',weights_only=False)
    cfg={k:v for k,v in ckpt['model_config'].items() if k in inspect.signature(DenseUVParserNet).parameters};cfg['predict_headphone_presence']=True
    model=DenseUVParserNet(**cfg).cuda();missing,extra=model.load_state_dict(ckpt['model'],strict=False)
    if extra or any(not x.startswith('ownership_head.presence.') for x in missing):raise ValueError((missing,extra))
    model.requires_grad_(False);model.ownership_head.presence.requires_grad_(True)
    attach_semantic_runtime(model,'siglip2','google/siglip2-base-patch16-224','cuda',local_files_only=True)
    renderer=DifferentiableRenderer(ckpt['args']['mappings_dir']).cuda();pipeline=json.loads((root/'runs/v101_layer_audit_20260906/validation/pipeline.json').read_text())
    split=json.loads((root/'runs/dense_uv_parser_v101_retrain_20260906/source_splits.json').read_text())
    train=OwnershipDataset(split['train'],32768,45101000);val=OwnershipDataset(split['validation'],256,59101000);test=OwnershipDataset(split['test'],256,69101000)
    optimizer=torch.optim.AdamW(model.ownership_head.presence.parameters(),lr=.001,weight_decay=.001)
    manifest={'parent':str(source),'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method':'Only MLP over pooled semantic probability maps is trained. No RGB or image-specific rule; existing ownership and v101 weights frozen.','steps':opt.steps,'pipeline':pipeline,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')}};write_json(out/'config.json',manifest)
    @torch.no_grad()
    def evaluate(dataset):
        model.eval();tp=fp=tn=fn=0
        for batch in DataLoader(dataset,batch_size=16):
            images,labels=render_ownership(batch['uv'].cuda(),batch['ownership'].cuda(),renderer,['front_left','back_left'])
            _,prediction=model.predict_ownership(images,images[:,3]>.5,return_presence=True);gt=(labels==4).flatten(1).any(1);pred=prediction.sigmoid()>=.9
            tp+=int((pred&gt).sum());fp+=int((pred&~gt).sum());tn+=int((~pred&~gt).sum());fn+=int((~pred&gt).sum())
        return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'negative_fpr':fp/max(1,fp+tn)}
    @torch.no_grad()
    def real():
        model.eval();stats={}
        for folder in sorted((root/'output_history/v101_retrained_20260906').iterdir()):
            m=json.loads((folder/'manifest.json').read_text());d=torch.load(folder/'diagnostics.pt',weights_only=False)
            images=d['images'].cuda();fg=cached_real_foreground(images,m['input'],pipeline)
            _,prediction=model.predict_ownership(images,fg>=.5,return_presence=True)
            stats[Path(m['input']).stem]=prediction.sigmoid().tolist()
        return stats
    start=time.time();step=0
    try:
        for batch in DataLoader(train,batch_size=8,shuffle=True,num_workers=4,pin_memory=True):
            if step>=opt.steps:break
            model.eval();model.ownership_head.presence.train()
            with torch.no_grad():images,labels=render_ownership(batch['uv'].cuda(),batch['ownership'].cuda(),renderer,['front_left','back_left']);fg=images[:,3]>.5
            if step%2==0:images=appearance_augment(images,fg[:,None],strength=1.4)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                _,prediction=model.predict_ownership(images,fg,return_presence=True);truth=(labels==4).flatten(1).any(1).float();loss=F.binary_cross_entropy_with_logits(prediction.float(),truth,weight=torch.where(truth>.5,1.,2.))
            if not torch.isfinite(loss):raise RuntimeError('Non-finite gate loss')
            loss.backward();optimizer.step();step+=1
            if step%50==0:
                status={'state':'training','step':step,'loss':float(loss.detach()),'elapsed_seconds':time.time()-start};write_json(out/'status.json',status);print(json.dumps(status),flush=True)
            if step%300==0 or step==opt.steps:
                metrics={'step':step,'validation':evaluate(val),'real':real()};write_json(out/f'evaluation_{step}.json',metrics)
                payload={**{k:v for k,v in ckpt.items() if k!='model'},'model':model.state_dict(),'model_config':{**ckpt['model_config'],'predict_headphone_presence':True},'inference_pipeline':pipeline,'presence_manifest':manifest,'presence_metrics':metrics}
                torch.save(payload,out/f'step_{step}.pt');print(json.dumps(metrics),flush=True)
        for name,value in ckpt['model'].items():
            if not torch.equal(value,model.state_dict()[name].cpu()):raise ValueError('Frozen tensor changed: '+name)
        write_json(out/'heldout.json',evaluate(test));write_json(out/'status.json',{'state':'complete','step':step,'elapsed_seconds':time.time()-start})
    except BaseException as e:write_json(out/'status.json',{'state':'failed','step':step,'error':repr(e)});raise


if __name__=='__main__':main()
