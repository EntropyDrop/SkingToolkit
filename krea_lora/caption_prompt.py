from __future__ import annotations

import hashlib


def build_captioned_prompt(config: dict, description: str) -> str:
    prompt_config = config["prompt"]
    description = " ".join(description.strip().split())
    if not description:
        raise ValueError("Character description is empty")
    return f"{str(prompt_config['format_prompt']).strip()} {str(prompt_config['identity_prefix']).strip()} {description}"


def caption_instruction_hash(config: dict) -> str:
    captioning = config["captioning"]
    payload = "\n".join(
        [
            str(captioning["model_path"]),
            str(captioning["instruction"]),
            str(captioning.get("max_new_tokens", 140)),
            str(captioning.get("max_words", 100)),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def checkpoint_prompt_id(path: str, index: int) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"test_{index:02d}_{digest}"
