#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the isolated one-step smoke-test configuration.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    with Path(args.source).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    root = Path(args.root).resolve()
    config["data"]["dataset_dir"] = str(root / "data")
    config["data"]["max_images"] = 8
    config["data"]["validation_fraction"] = 0.125
    config["training"]["output_dir"] = str(root / "run")
    config["training"]["gradient_accumulation_steps"] = 1
    config["training"]["rank"] = 4
    config["training"]["lora_alpha"] = 4
    config["training"]["layerwise_casting"] = True
    config["training"]["max_train_steps"] = 1
    config["training"]["save_every"] = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
