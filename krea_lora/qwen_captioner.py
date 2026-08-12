from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class QwenCaptioner:
    """Frozen Qwen3.6 multimodal captioner used before Krea training/inference."""

    def __init__(self, config: dict, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        model_path = Path(config["model_path"]).expanduser().resolve()
        kernel_path = Path(config["fp8_kernel_dir"]).expanduser().resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        if not kernel_path.is_dir():
            raise FileNotFoundError(kernel_path)

        from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor
        from transformers.integrations import finegrained_fp8 as transformers_fp8

        if str(kernel_path) not in sys.path:
            sys.path.insert(0, str(kernel_path))
        import finegrained_fp8 as local_fp8

        transformers_fp8._load_finegrained_fp8_kernel = lambda: transformers_fp8.FineGrainedFP8(
            matmul=local_fp8.matmul_2d,
            batched_matmul=local_fp8.matmul_batched,
            grouped_matmul=local_fp8.matmul_grouped,
        )
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        quantization_config = model_config.quantization_config
        excluded_modules = quantization_config.get("modules_to_not_convert", [])
        quantization_config["modules_to_not_convert"] = [
            name for name in excluded_modules if not name.endswith(".mlp.gate")
        ]
        self.model: Any | None = AutoModelForMultimodalLM.from_pretrained(
            str(model_path),
            config=model_config,
            device_map=device,
            dtype="auto",
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()

    @torch.inference_mode()
    def describe(self, image_or_path: Image.Image | str | Path) -> str:
        if self.model is None:
            raise RuntimeError("Qwen captioner has already been closed")
        if isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGB")
        else:
            with Image.open(image_or_path) as opened:
                image = opened.convert("RGB")
        image.thumbnail(
            (int(self.config.get("analysis_size", 448)),) * 2,
            Image.Resampling.LANCZOS,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": str(self.config["instruction"])},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=int(self.config.get("max_new_tokens", 140)),
            do_sample=False,
        )
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated, strict=True)]
        description = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        description = " ".join(description.strip().split())
        max_words = int(self.config.get("max_words", 100))
        words = description.split()
        if len(words) > max_words:
            candidate = " ".join(words[:max_words]).rstrip(" ,;:")
            sentence_end = max(candidate.rfind("."), candidate.rfind("!"), candidate.rfind("?"))
            if sentence_end >= len(candidate) // 2:
                candidate = candidate[: sentence_end + 1]
            elif not candidate.endswith((".", "!", "?")):
                candidate += "."
            description = candidate
        if len(description) < 20:
            raise RuntimeError("Qwen3.6 returned an unexpectedly short character description")
        return description

    def close(self) -> None:
        model = self.model
        self.model = None
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
