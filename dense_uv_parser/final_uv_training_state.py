"""Recoverable training state and checks for final-UV inference reproducibility."""
import random
from pathlib import Path

import numpy as np
import torch

from SkingToolkit.dense_uv_parser.uv_layout import tensor_to_rgba_image


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def restore_rng(state):
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def save_recovery(path, model, optimizer, step, signature):
    """One atomic file pairs the model, optimizer, schedule position and RNG."""
    state = {
        "format_version": 1,
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": step,
        "signature": signature,
        "rng": rng_state(),
    }
    atomic_save(state, path)
    return state["rng"]


def restore_recovery(path, model, optimizer, signature):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1 or state["signature"] != signature:
        raise ValueError("Recovery state does not match cache, parent or training schedule")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return int(state["step"])


def compare_roundtrip(cached, fresh, cached_base, fresh_base, cached_evidence, fresh_evidence):
    """Keep exact geometry/body checks; bound and report upstream RGB drift.

    The parent's iterative material fit can change sub-byte RGB values between
    executions. Allow at most 1/255 in float RGB AND one exported RGB level.
    This is not a tolerance for alpha, body edits, or serialization changes.
    """
    tensors = (cached, fresh, cached_base, fresh_base, cached_evidence, fresh_evidence)
    if not all(bool(torch.isfinite(x).all()) for x in tensors):
        return {"passed": False, "reason": "non-finite roundtrip tensors"}
    left = np.array(tensor_to_rgba_image(cached[0]))
    right = np.array(tensor_to_rgba_image(fresh[0]))
    byte_delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    changed = (left != right).any(2)
    report = {
        "rgba_exact": bool(np.array_equal(left, right)),
        "alpha_exact": bool(torch.equal(cached[:, 3], fresh[:, 3])),
        "base_alpha_exact": bool(torch.equal(cached_base[:, 3], fresh_base[:, 3])),
        "body_exact": bool(torch.equal(cached[:, :, 16:], fresh[:, :, 16:])),
        "cached_body_preserved": bool(torch.equal(cached[:, :, 16:], cached_base[:, :, 16:])),
        "fresh_body_preserved": bool(torch.equal(fresh[:, :, 16:], fresh_base[:, :, 16:])),
        "evidence_exact": bool(torch.equal(cached_evidence, fresh_evidence)),
        "base_rgb_max_delta": float((cached_base[:, :3] - fresh_base[:, :3]).abs().max()),
        "final_rgb_max_delta": float((cached[:, :3] - fresh[:, :3]).abs().max()),
        "png_rgb_max_delta": int(byte_delta[:, :, :3].max()),
        "different_texels": int(changed.sum()),
        "rgb_float_tolerance": 1 / 255,
        "rgb_byte_tolerance": 1,
        "differences": [{"xy": [int(x), int(y)], "cached": left[y, x].tolist(), "fresh": right[y, x].tolist()} for y, x in np.argwhere(changed)],
    }
    exact_fields = ("alpha_exact", "base_alpha_exact", "body_exact", "cached_body_preserved", "fresh_body_preserved", "evidence_exact")
    report["passed"] = all(report[k] for k in exact_fields) and max(report["base_rgb_max_delta"], report["final_rgb_max_delta"]) <= 1 / 255 and report["png_rgb_max_delta"] <= 1
    return report
