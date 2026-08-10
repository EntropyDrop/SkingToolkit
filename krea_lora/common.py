from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import torch


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "mc_preview.json"


def load_config(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config_path = Path(path or DEFAULT_CONFIG).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["_config_path"] = str(config_path)
    return config


def ensure_parent(path: str | os.PathLike[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {path}") from exc
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    output = ensure_parent(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    output = ensure_parent(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(output)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose bf16, fp16, or fp32") from exc


def pack_latents(latents: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """Pack BCHW Qwen Image latents into Krea2's BxSequencexChannels layout."""
    batch, channels, height, width = latents.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Latent size {height}x{width} is not divisible by patch size {patch_size}")
    latents = latents.view(
        batch,
        channels,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(
        batch,
        (height // patch_size) * (width // patch_size),
        channels * patch_size * patch_size,
    )


def prompt_cache_key(kind: str, prompt_id: str) -> str:
    if not prompt_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in prompt_id):
        raise ValueError(f"Unsafe prompt_id {prompt_id!r}; use letters, digits, '_' or '-'")
    return f"{kind}.{prompt_id}"


def relative_or_absolute(path: str | os.PathLike[str], base: Path) -> Path:
    value = Path(path).expanduser()
    return value if value.is_absolute() else (base / value).resolve()

