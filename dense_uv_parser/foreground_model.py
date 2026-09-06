"""Local pretrained BiRefNet plus versioned Minecraft decoder checkpoints."""
import hashlib,io,json
from pathlib import Path
import torch


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def load_foreground_model(model_dir,checkpoint=None,device='cuda'):
    from transformers import AutoConfig,AutoModelForImageSegmentation
    from safetensors.torch import load_file
    path=Path(model_dir)
    config=AutoConfig.from_pretrained(str(path),trust_remote_code=True,local_files_only=True)
    model=AutoModelForImageSegmentation.from_config(config,trust_remote_code=True)
    model.load_state_dict(load_file(path/'model.safetensors'),strict=True)
    metadata=None
    if checkpoint:
        checkpoint_bytes=Path(checkpoint).read_bytes()
        payload=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=False)
        metadata={**payload['manifest'],'checkpoint_sha256':hashlib.sha256(checkpoint_bytes).hexdigest()}
        if metadata['base_sha256']!=sha(path/'model.safetensors'):raise ValueError('Foreground base checkpoint mismatch')
        missing,extra=model.load_state_dict(payload['decoder'],strict=False)
        if extra or any(not name.startswith('bb.') for name in missing):raise ValueError((missing,extra))
    return model.eval().to(device),metadata


def foreground_logits(model,rgb):
    mean=rgb.new_tensor([.485,.456,.406])[None,:,None,None]
    std=rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    return model((rgb-mean)/std)[-1]
