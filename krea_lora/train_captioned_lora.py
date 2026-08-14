#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.krea2.pipeline_krea2 import calculate_shift
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from captioned_preview import CaptionedCheckpointPreviewer
from common import load_config, read_jsonl, resolve_dtype, write_json


class CaptionedLatentDataset(Dataset):
    def __init__(self, rows: list[dict], latent_dir: Path, prompt_dir: Path):
        self.items: list[tuple[Path, Path, str]] = []
        missing = 0
        for row in rows:
            if row.get("split") != "train":
                continue
            latent_path = latent_dir / f"{row['id']}.safetensors"
            prompt_path = prompt_dir / f"{row['prompt_id']}.safetensors"
            if latent_path.is_file() and prompt_path.is_file():
                self.items.append((latent_path, prompt_path, str(row["prompt_id"])))
            else:
                missing += 1
        if missing:
            print(f"warning: {missing} training rows are missing latent or prompt cache and will be skipped")
        if not self.items:
            raise RuntimeError("No fully cached captioned training samples were found")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        latent_path, prompt_path, prompt_id = self.items[index]
        latent = load_file(latent_path, device="cpu")["latents"]
        prompt = load_file(prompt_path, device="cpu")
        return latent, prompt["embeds"], prompt["mask"], prompt_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standard noise-to-target Krea2 LoRA with Qwen captions.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-lora", default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument(
        "--timestep-min-fraction",
        type=float,
        default=None,
        help="Lowest scheduler index fraction to sample: 0 is highest noise, 1 is lowest noise.",
    )
    parser.add_argument(
        "--timestep-max-fraction",
        type=float,
        default=None,
        help="Highest scheduler index fraction to sample: use a late interval for detail fine-tuning.",
    )
    return parser.parse_args()


def save_lora(transformer: torch.nn.Module, accelerator: Accelerator, output_dir: Path, metadata: dict) -> None:
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    state_dict = get_peft_model_state_dict(accelerator.unwrap_model(transformer))
    Krea2Pipeline.save_lora_weights(
        output_dir,
        transformer_lora_layers=state_dict,
        safe_serialization=True,
    )
    write_json(output_dir / "adapter_metadata.json", metadata)


def save_previews(
    previewer: CaptionedCheckpointPreviewer,
    transformer: torch.nn.Module,
    output_dir: Path,
    checkpoint: int | str,
) -> None:
    results = previewer.save(transformer, output_dir / "tests")
    write_json(
        output_dir / "tests" / "preview.json",
        {
            "checkpoint": checkpoint,
            "conditioning": "Qwen3.6 caption -> Krea prompt embedding",
            "generation_mode": previewer.mode,
            "strength": previewer.strength if previewer.mode == "img2img" else None,
            "steps": previewer.steps,
            "guidance_scale": 0.0,
            "raw_previews": previewer.save_raw,
            "crisp_postprocess": previewer.crisp_postprocess,
            "sharpen_radius": previewer.sharpen_radius,
            "sharpen_percent": previewer.sharpen_percent,
            "sharpen_threshold": previewer.sharpen_threshold,
            "contrast": previewer.contrast,
            "saturation": previewer.saturation,
            "posterize_bits": previewer.posterize_bits,
            "results": results,
        },
    )
    print(f"checkpoint previews saved: {output_dir / 'tests'}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["training"]
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        train_config["learning_rate"] = args.learning_rate
    if args.lr_warmup_steps is not None:
        if args.lr_warmup_steps < 0:
            raise ValueError("--lr-warmup-steps must be non-negative")
        train_config["lr_warmup_steps"] = args.lr_warmup_steps
    if args.timestep_min_fraction is not None:
        train_config["timestep_min_fraction"] = args.timestep_min_fraction
    if args.timestep_max_fraction is not None:
        train_config["timestep_max_fraction"] = args.timestep_max_fraction
    timestep_min_fraction = float(train_config.get("timestep_min_fraction", 0.0))
    timestep_max_fraction = float(train_config.get("timestep_max_fraction", 1.0))
    if not 0.0 <= timestep_min_fraction < timestep_max_fraction <= 1.0:
        raise ValueError(
            "training timestep fractions must satisfy "
            "0 <= timestep_min_fraction < timestep_max_fraction <= 1"
        )
    if int(train_config["batch_size"]) != 1:
        raise ValueError("Captioned prompt embeddings have variable length; training.batch_size must be 1")
    model_path = Path(model_config["path"]).expanduser().resolve()
    dataset_dir = Path(data_config["dataset_dir"]).expanduser().resolve()
    output_dir = Path(args.output_dir or train_config["output_dir"]).expanduser().resolve()
    max_train_steps = int(args.max_train_steps or train_config["max_train_steps"])
    rows = read_jsonl(dataset_dir / "metadata.jsonl")
    dataset = CaptionedLatentDataset(rows, dataset_dir / "latents", dataset_dir / "prompt_cache")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=int(train_config.get("num_workers", 1)),
        pin_memory=True,
        drop_last=True,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config["gradient_accumulation_steps"]),
        mixed_precision=str(train_config.get("mixed_precision", "bf16")),
        project_config=ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(output_dir / "logs")),
    )
    set_seed(int(train_config["seed"]), device_specific=True)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "resolved_config.json", config)

    weight_dtype = resolve_dtype(model_config.get("dtype", "bf16"))
    transformer = Krea2Transformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=weight_dtype,
        local_files_only=True,
    )
    transformer.requires_grad_(False)
    if bool(train_config.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()

    lora_rank = int(train_config["rank"])
    lora_alpha = int(train_config["lora_alpha"])
    resume_state = None
    if args.resume_lora:
        state = Krea2Pipeline.lora_state_dict(args.resume_lora)
        if isinstance(state, tuple):
            state = state[0]
        resume_state = {
            key.removeprefix("transformer."): value
            for key, value in state.items()
            if key.startswith("transformer.")
        }
        ranks = {
            int(value.shape[0])
            for key, value in resume_state.items()
            if key.endswith("lora_A.weight")
        }
        if len(ranks) != 1:
            raise ValueError(f"Could not infer one resumed LoRA rank: {sorted(ranks)}")
        inferred_rank = ranks.pop()
        if inferred_rank != lora_rank:
            if lora_alpha == lora_rank:
                lora_alpha = inferred_rank
            print(f"overriding configured LoRA rank {lora_rank} with resumed rank {inferred_rank}")
            lora_rank = inferred_rank
    transformer.add_adapter(
        LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=float(train_config.get("lora_dropout", 0.0)),
            init_lora_weights="gaussian",
            target_modules=list(train_config["target_modules"]),
        )
    )
    if bool(train_config.get("layerwise_casting", False)):
        transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=weight_dtype,
        )
    if resume_state is not None:
        incompatible = set_peft_model_state_dict(transformer, resume_state, adapter_name="default")
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected resumed LoRA keys: {incompatible.unexpected_keys[:8]}")
        print(f"resumed LoRA initialization: {args.resume_lora}")

    trainable_parameters = [parameter for parameter in transformer.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    total_count = sum(parameter.numel() for parameter in transformer.parameters())
    if accelerator.is_main_process:
        print(
            f"trainable parameters: {trainable_count:,} / {total_count:,} "
            f"({100 * trainable_count / total_count:.4f}%)"
        )
        print("conditioning: per-image Qwen caption, standard Gaussian noise -> MC target flow")
        print(
            "timestep index fraction: "
            f"{timestep_min_fraction:.3f}..{timestep_max_fraction:.3f} "
            "(0=highest noise, 1=lowest noise)"
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_config["learning_rate"]),
        betas=(float(train_config.get("adam_beta1", 0.9)), float(train_config.get("adam_beta2", 0.999))),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
        eps=float(train_config.get("adam_epsilon", 1e-8)),
    )
    lr_scheduler = get_scheduler(
        str(train_config.get("lr_scheduler", "cosine")),
        optimizer=optimizer,
        num_warmup_steps=int(train_config.get("lr_warmup_steps", 100)),
        num_training_steps=max_train_steps,
    )
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path,
        subfolder="scheduler",
        local_files_only=True,
    )
    transformer, optimizer, loader, lr_scheduler = accelerator.prepare(transformer, optimizer, loader, lr_scheduler)

    first_latent, first_embeds, _, _ = dataset[0]
    sequence_length, channels = first_latent.shape
    unwrapped_config = accelerator.unwrap_model(transformer).config
    if channels != int(unwrapped_config.in_channels):
        raise ValueError(f"Latent channels {channels} != transformer in_channels {unwrapped_config.in_channels}")
    height = int(rows[0]["height"])
    width = int(rows[0]["width"])
    vae_scale_factor = int(model_config.get("vae_scale_factor", 8))
    patch_size = int(model_config.get("patch_size", 2))
    grid_height = height // (vae_scale_factor * patch_size)
    grid_width = width // (vae_scale_factor * patch_size)
    if grid_height * grid_width != sequence_length:
        raise ValueError(f"Resolution implies {grid_height * grid_width} tokens, cache has {sequence_length}")
    mu = calculate_shift(
        sequence_length,
        noise_scheduler.config.get("base_image_seq_len", 256),
        noise_scheduler.config.get("max_image_seq_len", 6400),
        noise_scheduler.config.get("base_shift", 0.5),
        noise_scheduler.config.get("max_shift", 1.15),
    )
    noise_scheduler.set_timesteps(
        noise_scheduler.config.num_train_timesteps,
        device=accelerator.device,
        mu=mu,
    )
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    schedule_sigmas = noise_scheduler.sigmas.to(accelerator.device)
    weighting_scheme = str(train_config.get("weighting_scheme", "logit_normal"))

    checkpoint_rows_path = dataset_dir / "checkpoint_metadata.jsonl"
    checkpoint_rows = read_jsonl(checkpoint_rows_path) if checkpoint_rows_path.is_file() else []
    previewer = None
    if checkpoint_rows and accelerator.is_main_process:
        preview_config = config["checkpoint_preview"]
        previewer = CaptionedCheckpointPreviewer(
            rows=checkpoint_rows,
            prompt_cache_dir=dataset_dir / "checkpoint_prompt_cache",
            model_path=model_path,
            device=accelerator.device,
            dtype=weight_dtype,
            width=int(preview_config.get("width", width)),
            height=int(preview_config.get("height", height)),
            vae_scale_factor=vae_scale_factor,
            patch_size=patch_size,
            steps=int(preview_config.get("steps", 28)),
            seed=int(preview_config.get("seed", train_config["seed"])),
            mode=str(preview_config.get("mode", "img2img")),
            strength=float(preview_config.get("strength", 0.9)),
            white_background_threshold=int(preview_config.get("white_background_threshold", 250)),
            crisp_postprocess=bool(preview_config.get("crisp_postprocess", True)),
            sharpen_radius=float(preview_config.get("sharpen_radius", 0.6)),
            sharpen_percent=int(preview_config.get("sharpen_percent", 80)),
            sharpen_threshold=int(preview_config.get("sharpen_threshold", 3)),
            contrast=float(preview_config.get("contrast", 1.16)),
            saturation=float(preview_config.get("saturation", 1.10)),
            posterize_bits=int(preview_config.get("posterize_bits", 4)),
            save_raw=bool(preview_config.get("save_raw", True)),
        )
        print(
            f"checkpoint previews: {len(checkpoint_rows)} Qwen-captioned image(s), "
            f"{previewer.steps} steps, {previewer.mode}, CFG 0"
        )

    metadata = {
        "base_model": str(model_path),
        "task": "Qwen-captioned standard noise-to-target Minecraft preview",
        "conditioning_schema": "qwen_caption_text_to_image",
        "rank": str(lora_rank),
        "lora_alpha": str(lora_alpha),
        "resolution": f"{width}x{height}",
        "initial_prompt_tokens": str(first_embeds.shape[0]),
        "layerwise_casting": str(bool(train_config.get("layerwise_casting", False))),
        "timestep_min_fraction": str(timestep_min_fraction),
        "timestep_max_fraction": str(timestep_max_fraction),
    }
    global_step = 0
    progress = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process, desc="Krea2 captioned LoRA")

    if checkpoint_rows:
        accelerator.wait_for_everyone()
        checkpoint_dir = output_dir / "checkpoint-0"
        save_lora(transformer, accelerator, checkpoint_dir, metadata)
        if accelerator.is_main_process:
            if previewer is None:
                raise RuntimeError("Main process did not initialize captioned checkpoint previews")
            save_previews(previewer, accelerator.unwrap_model(transformer), checkpoint_dir, 0)
        accelerator.wait_for_everyone()

    transformer.train()
    while global_step < max_train_steps:
        for clean_latents, embeds, masks, _prompt_ids in loader:
            with accelerator.accumulate(transformer):
                clean_latents = clean_latents.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                embeds = embeds.to(accelerator.device, dtype=weight_dtype, non_blocking=True)
                masks = masks.to(accelerator.device, non_blocking=True)
                noise = torch.randn_like(clean_latents)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=weighting_scheme,
                    batch_size=clean_latents.shape[0],
                    logit_mean=float(train_config.get("logit_mean", 0.0)),
                    logit_std=float(train_config.get("logit_std", 1.0)),
                    mode_scale=float(train_config.get("mode_scale", 1.29)),
                    device=accelerator.device,
                )
                u = timestep_min_fraction + u * (timestep_max_fraction - timestep_min_fraction)
                indices = (u * noise_scheduler.config.num_train_timesteps).long().clamp_(
                    0, noise_scheduler.config.num_train_timesteps - 1
                )
                timesteps = schedule_timesteps[indices]
                sigmas = schedule_sigmas[indices].to(dtype=clean_latents.dtype).view(-1, 1, 1)
                noisy_latents = (1.0 - sigmas) * clean_latents + sigmas * noise
                position_ids = Krea2Pipeline.prepare_position_ids(
                    embeds.shape[1],
                    grid_height,
                    grid_width,
                    accelerator.device,
                )
                prediction = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=embeds,
                    timestep=(timesteps / noise_scheduler.config.num_train_timesteps).to(weight_dtype),
                    position_ids=position_ids,
                    encoder_attention_mask=masks,
                    return_dict=False,
                )[0]
                target = noise - clean_latents
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=weighting_scheme, sigmas=sigmas)
                loss = (
                    weighting.float() * F.mse_loss(prediction.float(), target.float(), reduction="none")
                ).reshape(clean_latents.shape[0], -1).mean(dim=1).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_parameters,
                        float(train_config.get("max_grad_norm", 1.0)),
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                progress.set_postfix(loss=f"{loss.detach().item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")
                save_every = int(train_config.get("save_every", 250))
                if save_every > 0 and global_step % save_every == 0:
                    accelerator.wait_for_everyone()
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    save_lora(transformer, accelerator, checkpoint_dir, metadata)
                    if accelerator.is_main_process and previewer is not None:
                        save_previews(previewer, accelerator.unwrap_model(transformer), checkpoint_dir, global_step)
                    accelerator.wait_for_everyone()
            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    final_dir = output_dir / "final"
    save_lora(transformer, accelerator, final_dir, metadata)
    if accelerator.is_main_process and previewer is not None:
        save_previews(previewer, accelerator.unwrap_model(transformer), final_dir, "final")
    if accelerator.is_main_process:
        write_json(
            output_dir / "training_state.json",
            {
                "global_step": global_step,
                "trainable_parameters": trainable_count,
                "dataset_items": len(dataset),
                "conditioning_schema": "qwen_caption_text_to_image",
                "timestep_min_fraction": timestep_min_fraction,
                "timestep_max_fraction": timestep_max_fraction,
                "final_lora": str(final_dir),
            },
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
