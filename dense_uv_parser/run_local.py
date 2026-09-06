"""Run a toolkit entry point using only this checkout, regardless of its name.

Example: python dense_uv_parser/run_local.py semantic_generalization --help
"""
import importlib.machinery
import runpy
import sys
import types
from pathlib import Path


def bind_checkout():
    root = Path(__file__).resolve().parents[1]
    existing = sys.modules.get("SkingToolkit")
    if existing is not None:
        if list(existing.__path__) != [str(root)]:
            raise RuntimeError("SkingToolkit was already imported from another checkout")
    else:
        package = types.ModuleType("SkingToolkit")
        package.__path__ = [str(root)]
        package.__package__ = "SkingToolkit"
        package.__spec__ = importlib.machinery.ModuleSpec(
            "SkingToolkit", loader=None, is_package=True
        )
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules["SkingToolkit"] = package
    return root


if __name__ == "__main__":
    bind_checkout()
    entry = sys.argv.pop(1) if len(sys.argv) > 1 else "semantic_generalization"
    if entry not in {"semantic_generalization", "infer", "train", "test_semantic_generalization", "train_accessories", "test_accessories", "batch_accessories", "test_affine_routing", "test_semantic_parser", "test_foreground", "foreground_provider", "prepare_foreground_data", "train_foreground", "test_matting_data", "foreground_batch", "released_inference", "train_ownership", "test_ownership", "train_headphone_presence", "train_headwear", "test_headwear", "train_headwear_presence"}:
        raise SystemExit("Unsupported local entry point")
    runpy.run_module(f"SkingToolkit.dense_uv_parser.{entry}", run_name="__main__", alter_sys=True)
