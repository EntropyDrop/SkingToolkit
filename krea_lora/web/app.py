from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field


PROJECT_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
CONFIG_PATH = Path(
    os.getenv("KREA_CONFIG", PROJECT_DIR / "configs" / "ddj_captioned.json")
).expanduser().resolve()
with CONFIG_PATH.open("r", encoding="utf-8") as handle:
    CONFIG = json.load(handle)

DEFAULT_LORA = PROJECT_DIR / "runs" / "ddj_captioned_front_left_back_left_lora" / "best"
LORA_PATH = Path(os.getenv("KREA_LORA_PATH", DEFAULT_LORA)).expanduser().resolve()
RUNTIME_DIR = Path(os.getenv("KREA_WEB_RUNTIME", PROJECT_DIR / "runs" / "captioned_web")).expanduser().resolve()
INPUT_DIR = RUNTIME_DIR / "inputs"
OUTPUT_DIR = RUNTIME_DIR / "outputs"
LOG_DIR = RUNTIME_DIR / "logs"
for directory in (INPUT_DIR, OUTPUT_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Krea MC Captioned LoRA", version="2.0.0")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
jobs: dict[str, dict[str, Any]] = {}
active_job_id: str | None = None
state_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    reference_image: str = Field(min_length=32, max_length=14_500_000)
    prompt: str = Field(min_length=1)
    supplemental_description: str = ""
    mode: Literal["txt2img", "img2img"] = "txt2img"
    strength: float = Field(default=0.95, ge=0.05, le=1.0)
    seed: int | None = Field(default=None, ge=-1, le=2**63 - 1)
    steps: int = Field(default=28, ge=8, le=40)
    guidance_scale: float = Field(default=0.0, ge=0.0, le=3.0)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in {"input_path", "output_path", "log_path"}}


def decode_reference(data_url: str) -> Image.Image:
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="参考图格式不正确") from exc
    if not header.startswith("data:image/") or ";base64" not in header:
        raise HTTPException(status_code=422, detail="参考图必须是 JPG、PNG 或 WebP")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=422, detail="参考图 Base64 无效") from exc
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="参考图不能超过 10MB")
    try:
        with Image.open(BytesIO(raw)) as opened:
            if opened.width * opened.height > 40_000_000:
                raise HTTPException(status_code=413, detail="参考图不能超过 4000 万像素")
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.load()
            return image
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="无法读取参考图") from exc


def run_generation(job_id: str, request: GenerateRequest) -> dict[str, Any]:
    job = jobs[job_id]
    command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "generate_captioned.py"),
        "--config",
        str(CONFIG_PATH),
        "--source",
        str(job["input_path"]),
        "--lora",
        str(LORA_PATH),
        "--output",
        str(job["output_path"]),
        "--mode",
        request.mode,
        "--strength",
        str(request.strength),
        "--seed",
        str(job["seed"]),
        "--steps",
        str(request.steps),
        "--guidance-scale",
        str(request.guidance_scale),
        "--format-prompt",
        request.prompt,
    ]
    if request.supplemental_description.strip():
        command.extend(["--description-suffix", request.supplemental_description.strip()])

    progress_pattern = re.compile(r"(\d{1,3})%")
    with Path(job["log_path"]).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith("Qwen description:"):
                job["reference_description"] = line.split(":", 1)[1].strip()
                job.update(phase="Qwen 描述完成，正在加载 Krea-2-Raw + MC LoRA", progress=0.35)
            elif "Loading pipeline components" in line:
                job.update(phase="正在加载 Krea-2-Raw + MC LoRA", progress=max(job["progress"], 0.38))
            else:
                matches = progress_pattern.findall(line)
                if matches:
                    percent = min(int(matches[-1]), 100)
                    job.update(phase=f"MC LoRA 正在生成 {percent}%", progress=0.45 + 0.5 * percent / 100)
        return_code = process.wait()
    if return_code:
        tail = Path(job["log_path"]).read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"生成进程退出码 {return_code}\n{tail}")
    metadata_path = Path(job["output_path"]).with_suffix(".json")
    if not Path(job["output_path"]).is_file() or not metadata_path.is_file():
        raise RuntimeError("生成结束但没有输出 PNG/JSON")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "image_url": f"/outputs/{Path(job['output_path']).name}",
        "qwen_description": metadata.get("qwen_description", ""),
        "effective_prompt": metadata.get("prompt", ""),
    }


async def execute(job_id: str, request: GenerateRequest) -> None:
    global active_job_id
    try:
        jobs[job_id].update(status="running", phase="Qwen3.6-27B 正在识别角色", progress=0.03)
        result = await asyncio.to_thread(run_generation, job_id, request)
        jobs[job_id].update(
            status="completed",
            phase="生成完成",
            progress=1.0,
            completed_at=now(),
            **result,
        )
    except Exception as exc:
        jobs[job_id].update(status="failed", phase="生成失败", error=str(exc), completed_at=now())
    finally:
        async with state_lock:
            if active_job_id == job_id:
                active_job_id = None


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return {
        "prompt": CONFIG["prompt"]["format_prompt"],
        "model": str(CONFIG["model"]["path"]),
        "caption_model": str(CONFIG["captioning"]["model_path"]),
        "lora": str(LORA_PATH),
        "default_mode": "txt2img",
        "steps": int(CONFIG["inference"].get("steps", 28)),
        "guidance_scale": float(CONFIG["inference"].get("guidance_scale", 0.0)),
        "crisp_postprocess": bool(CONFIG["inference"].get("crisp_postprocess", True)),
    }


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "ok": CONFIG_PATH.is_file() and LORA_PATH.is_dir(),
        "busy": active_job_id is not None,
        "config_exists": CONFIG_PATH.is_file(),
        "lora_exists": LORA_PATH.is_dir(),
        "lora": str(LORA_PATH),
    }


@app.post("/api/generate", status_code=202)
async def generate(request: GenerateRequest) -> dict[str, Any]:
    global active_job_id
    image = decode_reference(request.reference_image)
    async with state_lock:
        if active_job_id is not None:
            raise HTTPException(status_code=409, detail="GPU 正在生成，请等待当前任务完成")
        job_id = uuid.uuid4().hex[:12]
        seed = request.seed if request.seed is not None and request.seed >= 0 else int.from_bytes(os.urandom(8), "big")
        input_path = INPUT_DIR / f"{job_id}.png"
        output_path = OUTPUT_DIR / f"mc-{job_id}-{seed}.png"
        image.save(input_path, optimize=True)
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "phase": "等待执行",
            "progress": 0.0,
            "seed": seed,
            "mode": request.mode,
            "strength": request.strength if request.mode == "img2img" else None,
            "steps": request.steps,
            "created_at": now(),
            "input_path": str(input_path),
            "output_path": str(output_path),
            "log_path": str(LOG_DIR / f"{job_id}.log"),
        }
        active_job_id = job_id
    asyncio.create_task(execute(job_id, request))
    return public_job(jobs[job_id])


@app.get("/api/jobs/{job_id}")
async def job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="任务不存在")
    return public_job(jobs[job_id])
