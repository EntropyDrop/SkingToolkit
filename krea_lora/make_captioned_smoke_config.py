#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated Qwen-captioned one-step smoke config.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--paired-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paired-output", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    with Path(args.source).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with Path(args.paired_source).open("r", encoding="utf-8") as handle:
        paired_config = json.load(handle)
    root = Path(args.root).resolve()
    paired_config["data"]["dataset_dir"] = str(root / "paired")
    paired_config["data"]["max_images"] = 2
    paired_config["data"]["validation_fraction"] = 0.5
    config["data"].update(paired_config["data"])
    config["data"]["paired_dataset_dir"] = str(root / "paired")
    config["data"]["dataset_dir"] = str(root / "data")
    config["data"]["max_images"] = 2
    config["data"]["validation_fraction"] = 0.5
    config["checkpoint_preview"]["test_images"] = config["checkpoint_preview"]["test_images"][:1]
    config["checkpoint_preview"]["steps"] = 2
    config["training"]["output_dir"] = str(root / "run")
    config["training"]["gradient_accumulation_steps"] = 1
    config["training"]["num_workers"] = 0
    config["training"]["rank"] = 4
    config["training"]["lora_alpha"] = 4
    config["training"]["layerwise_casting"] = True
    config["training"]["max_train_steps"] = 1
    config["training"]["save_every"] = 1
    config["inference"]["output_path"] = str(root / "generated.png")
    config["inference"]["steps"] = 2
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    paired_output = Path(args.paired_output)
    paired_output.parent.mkdir(parents=True, exist_ok=True)
    with paired_output.open("w", encoding="utf-8") as handle:
        json.dump(paired_config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
