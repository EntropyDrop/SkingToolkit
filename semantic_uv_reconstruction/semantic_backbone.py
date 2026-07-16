"""Frozen open-vocabulary visual semantics for semantic UV reconstruction.

The dependency on ``transformers`` is intentionally lazy.  Dataset utilities,
unit tests, and the geometry-only fallback therefore remain usable on machines
without a Hugging Face training environment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigLIP2VisionBackbone(nn.Module):
    """Frozen SigLIP2 vision tower with differentiable image preprocessing.

    FixRes SigLIP2 checkpoints are used deliberately: a differentiable,
    aspect-preserving letterbox can be shared by source renders and predicted
    renders.  That lets the render-semantic cycle propagate gradients all the
    way back to the predicted UV atlas.
    """

    def __init__(
        self,
        model_name="google/siglip2-base-patch16-224",
        token_channels=128,
        local_files_only=False,
    ):
        super().__init__()
        if "naflex" in model_name.lower():
            raise ValueError(
                "NaFlex checkpoints are not supported by the differentiable render-cycle "
                "adapter. Use a FixRes SigLIP2 checkpoint such as "
                "google/siglip2-base-patch16-224."
            )
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as error:
            raise ImportError(
                "SigLIP2 training requires Hugging Face Transformers. Install the remote "
                "training environment with: pip install -U transformers sentencepiece "
                "safetensors"
            ) from error

        processor = AutoImageProcessor.from_pretrained(
            model_name,
            local_files_only=bool(local_files_only),
            use_fast=False,
        )
        full_model = AutoModel.from_pretrained(
            model_name, local_files_only=bool(local_files_only)
        )
        if not hasattr(full_model, "vision_model"):
            raise ValueError(f"{model_name} does not expose a SigLIP vision tower.")
        self.vision_model = full_model.vision_model
        config = getattr(full_model.config, "vision_config", self.vision_model.config)
        image_size = getattr(config, "image_size", None)
        patch_size = getattr(config, "patch_size", None)
        hidden_size = getattr(config, "hidden_size", None)
        if image_size is None:
            processor_size = getattr(processor, "size", {})
            if isinstance(processor_size, dict):
                image_size = processor_size.get("height") or processor_size.get("shortest_edge")
        if not isinstance(image_size, int) or not isinstance(patch_size, int):
            raise ValueError(
                f"{model_name} is not a supported fixed-resolution SigLIP2 vision model."
            )
        if not isinstance(hidden_size, int):
            raise ValueError(f"Cannot determine hidden_size for {model_name}.")

        image_mean = getattr(processor, "image_mean", (0.5, 0.5, 0.5))
        image_std = getattr(processor, "image_std", (0.5, 0.5, 0.5))
        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.raw_feature_dim = int(hidden_size)
        self.token_channels = int(token_channels)
        # Cache only immutable Python index tuples. A CUDA tensor kept here can
        # outlive a torch.compile CUDA Graph invocation and point at storage
        # overwritten by the next replay.
        self._valid_token_index_cache = {}
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, token_channels),
        )
        self.global_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, token_channels),
        )

        self.vision_model.requires_grad_(False)
        self.vision_model.eval()

    def train(self, mode=True):
        super().train(mode)
        # The projection adapters train; the pretrained vision tower never does.
        self.vision_model.eval()
        return self

    def _letterbox(self, images):
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected Nx3xHxW images, got {tuple(images.shape)}.")
        height, width = images.shape[-2:]
        scale = min(self.image_size / height, self.image_size / width)
        resized_height = max(1, min(self.image_size, round(height * scale)))
        resized_width = max(1, min(self.image_size, round(width * scale)))
        resized = F.interpolate(
            images.float(),
            size=(resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pad_height = self.image_size - resized_height
        pad_width = self.image_size - resized_width
        left = pad_width // 2
        right = pad_width - left
        top = pad_height // 2
        bottom = pad_height - top
        pixels = F.pad(resized, (left, right, top, bottom), value=0.5)
        pixels = (pixels - self.image_mean) / self.image_std
        return pixels, (top, left, resized_height, resized_width)

    def _valid_token_indices(self, content_rect, token_count, device):
        cache_key = (*content_rect, int(token_count))
        cached = self._valid_token_index_cache.get(cache_key)
        if cached is None:
            top, left, resized_height, resized_width = content_rect
            patch_grid = self.image_size // self.patch_size
            patch_tokens = patch_grid * patch_grid
            if token_count not in (patch_tokens, patch_tokens + 1):
                cached = tuple(range(token_count))
            else:
                row_start = top // self.patch_size
                row_stop = min(
                    patch_grid,
                    (top + resized_height + self.patch_size - 1) // self.patch_size,
                )
                column_start = left // self.patch_size
                column_stop = min(
                    patch_grid,
                    (left + resized_width + self.patch_size - 1) // self.patch_size,
                )
                patch_indices = tuple(
                    row * patch_grid + column
                    for row in range(row_start, row_stop)
                    for column in range(column_start, column_stop)
                )
                cached = (
                    (0, *(index + 1 for index in patch_indices))
                    if token_count == patch_tokens + 1
                    else patch_indices
                )
            self._valid_token_index_cache[cache_key] = cached
        # This tensor belongs only to the current eager/compiled invocation.
        # Never retain it in Python state across CUDA Graph replays.
        return torch.tensor(cached, dtype=torch.long, device=device)

    def forward(self, images):
        pixels, content_rect = self._letterbox(images)
        outputs = self.vision_model(
            pixel_values=pixels,
        )
        raw_tokens = outputs.last_hidden_state
        raw_global = outputs.pooler_output

        # Every image in this call has the same spatial shape and therefore the
        # same letterbox mask. Remove padded patch tokens instead of carrying a
        # key-padding mask into every UV cross-attention layer. Besides reducing
        # memory length, an unmasked attention call can use PyTorch's fused SDPA
        # kernels on supported CUDA devices.
        valid_token_indices = self._valid_token_indices(
            content_rect, raw_tokens.shape[1], raw_tokens.device
        )
        raw_tokens = raw_tokens.index_select(1, valid_token_indices)
        patch_mask = torch.ones(
            raw_tokens.shape[:2], dtype=torch.bool, device=raw_tokens.device
        )

        return {
            "tokens": self.token_projection(raw_tokens),
            "token_mask": patch_mask,
            "tokens_compact": True,
            "global": self.global_projection(raw_global),
            "raw_global": raw_global,
        }
