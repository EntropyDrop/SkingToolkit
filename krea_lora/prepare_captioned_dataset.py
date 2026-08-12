#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from caption_prompt import build_captioned_prompt, caption_instruction_hash
from common import load_config, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-image-prompt DDJ target metadata from Qwen captions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    paired_dir = Path(data["paired_dataset_dir"]).expanduser().resolve()
    dataset_dir = Path(data["dataset_dir"]).expanduser().resolve()
    metadata_path = dataset_dir / "metadata.jsonl"
    checkpoint_path = dataset_dir / "checkpoint_metadata.jsonl"
    if metadata_path.exists() and not args.force:
        raise FileExistsError(f"{metadata_path} already exists; pass --force to rebuild")
    captions = read_jsonl(dataset_dir / "qwen_captions.jsonl")
    current_hash = caption_instruction_hash(config)
    caption_map: dict[tuple[str, str], dict] = {}
    for row in captions:
        if row.get("instruction_hash") == current_hash and row.get("description"):
            caption_map[(str(row["kind"]), str(row["id"]))] = row

    output_rows: list[dict] = []
    missing: list[str] = []
    for row in read_jsonl(paired_dir / "metadata.jsonl"):
        sample_id = str(row["id"])
        caption = caption_map.get(("train", sample_id))
        if caption is None:
            missing.append(sample_id)
            continue
        output_rows.append(
            {
                "id": sample_id,
                "image": str(row["image"]),
                "source_image": str(row["source_image"]),
                "description": str(caption["description"]),
                "caption": build_captioned_prompt(config, str(caption["description"])),
                "prompt_id": sample_id,
                "split": str(row["split"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
            }
        )
    if missing and not args.allow_partial:
        raise RuntimeError(
            f"Missing {len(missing)} training captions; rerun caption_ddj_sources.py. First IDs: {missing[:8]}"
        )
    checkpoint_rows: list[dict] = []
    for (kind, sample_id), caption in caption_map.items():
        if kind != "checkpoint":
            continue
        checkpoint_rows.append(
            {
                "id": sample_id,
                "source_image": str(caption["source_image"]),
                "description": str(caption["description"]),
                "caption": build_captioned_prompt(config, str(caption["description"])),
                "prompt_id": sample_id,
            }
        )
    checkpoint_rows.sort(key=lambda row: row["id"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(metadata_path, output_rows)
    write_jsonl(checkpoint_path, checkpoint_rows)
    write_json(
        dataset_dir / "summary.json",
        {
            "paired_dataset_dir": str(paired_dir),
            "rows": len(output_rows),
            "train": sum(row["split"] == "train" for row in output_rows),
            "validation": sum(row["split"] == "validation" for row in output_rows),
            "checkpoint_prompts": len(checkpoint_rows),
            "missing_captions": len(missing),
            "instruction_hash": current_hash,
            "conditioning": "per-image Qwen caption -> Krea text encoder",
        },
    )
    print({"rows": len(output_rows), "checkpoint_prompts": len(checkpoint_rows), "missing": len(missing)})


if __name__ == "__main__":
    main()
