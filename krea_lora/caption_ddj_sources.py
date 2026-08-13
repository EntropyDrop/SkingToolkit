#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm.auto import tqdm

from caption_prompt import caption_instruction_hash, checkpoint_prompt_id
from checkpoint_preview import checkpoint_test_image_paths
from common import load_config, read_jsonl
from qwen_captioner import QwenCaptioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption DDJ reference images with frozen Qwen3.6-27B FP8.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None, help="Limit training rows; checkpoint images are retained.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    paired_dir = Path(data["paired_dataset_dir"]).expanduser().resolve()
    dataset_dir = Path(data["dataset_dir"]).expanduser().resolve()
    output_path = dataset_dir / "qwen_captions.jsonl"
    error_path = dataset_dir / "qwen_caption_errors.jsonl"
    instruction_hash = caption_instruction_hash(config)
    if args.force:
        output_path.unlink(missing_ok=True)
        error_path.unlink(missing_ok=True)

    if not output_path.exists() and not args.force:
        for candidate in config["captioning"].get("reuse_caption_files", []):
            reusable_path = Path(candidate).expanduser().resolve()
            if reusable_path.is_file():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(reusable_path, output_path)
                print(f"reused Qwen captions: {reusable_path} -> {output_path}")
                break

    paired_rows = read_jsonl(paired_dir / "metadata.jsonl")
    if args.limit:
        paired_rows = paired_rows[: args.limit]
    requested: list[dict[str, str]] = []
    checkpoint_paths = checkpoint_test_image_paths(config.get("checkpoint_preview", {}).get("test_images", []))
    for index, path in enumerate(checkpoint_paths):
        requested.append(
            {
                "kind": "checkpoint",
                "id": checkpoint_prompt_id(str(path), index),
                "source_image": str(path),
            }
        )
    requested.extend(
        {
            "kind": "train",
            "id": str(row["id"]),
            "source_image": str(row["source_image"]),
        }
        for row in paired_rows
    )

    existing: set[tuple[str, str, str]] = set()
    if output_path.is_file():
        for row in read_jsonl(output_path):
            if row.get("description"):
                existing.add((str(row["kind"]), str(row["id"]), str(row.get("instruction_hash", ""))))
    pending = [
        row for row in requested if (row["kind"], row["id"], instruction_hash) not in existing
    ]
    if not pending:
        print(f"All {len(requested)} Qwen captions already exist: {output_path}")
        return

    captioner = QwenCaptioner(config["captioning"], device=args.device)
    failures = 0
    batch_size = max(1, int(config["captioning"].get("batch_size", 1)))
    try:
        progress = tqdm(total=len(pending), desc=f"Qwen3.6 character captions (batch={batch_size})")
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            try:
                descriptions = captioner.describe_many([row["source_image"] for row in batch])
                if len(descriptions) != len(batch):
                    raise RuntimeError(f"Qwen returned {len(descriptions)} captions for a batch of {len(batch)}")
                completed = list(zip(batch, descriptions, strict=True))
            except Exception as batch_exc:
                if len(batch) == 1:
                    completed = []
                    failures += 1
                    append_jsonl(
                        error_path,
                        {**batch[0], "error": repr(batch_exc), "instruction_hash": instruction_hash},
                    )
                    print(f"caption failed for {batch[0]['source_image']}: {batch_exc}")
                else:
                    print(f"caption batch failed; retrying {len(batch)} images individually: {batch_exc}")
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    completed = []
                    for row in batch:
                        try:
                            completed.append((row, captioner.describe(row["source_image"])))
                        except Exception as exc:
                            failures += 1
                            append_jsonl(error_path, {**row, "error": repr(exc), "instruction_hash": instruction_hash})
                            print(f"caption failed for {row['source_image']}: {exc}")
            for row, description in completed:
                append_jsonl(
                    output_path,
                    {
                        **row,
                        "description": description,
                        "instruction_hash": instruction_hash,
                        "model_path": str(Path(config["captioning"]["model_path"]).expanduser().resolve()),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            progress.update(len(batch))
        progress.close()
    finally:
        captioner.close()
    if failures:
        raise RuntimeError(f"{failures} Qwen captions failed; successes are preserved for resume")
    print({"captions": len(pending), "batch_size": batch_size, "output": str(output_path)})


if __name__ == "__main__":
    main()
