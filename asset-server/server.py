"""
Wisp Asset Generation Server — port 3004
Generates game backgrounds/assets with SDXL on RTX 5090.
"""
import sys
sys.path.insert(0, "/opt/eagleai-photos/backend/services")

import asyncio
import gc
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
PENDING_DIR = ASSETS_DIR / "pending"
APPROVED_DIR = ASSETS_DIR / "approved"
REJECTED_DIR = ASSETS_DIR / "rejected"
META_FILE = ASSETS_DIR / "meta.json"

for d in [PENDING_DIR, APPROVED_DIR, REJECTED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GAME_BG_DIR = Path("/opt/gry/Wisp/public/assets/bg")
GAME_BG_DIR.mkdir(parents=True, exist_ok=True)

MUSIC_DIR         = ASSETS_DIR / "music"
MUSIC_PENDING_DIR = MUSIC_DIR / "pending"
MUSIC_APPROVED_DIR= MUSIC_DIR / "approved"
GAME_MUSIC_DIR    = Path("/opt/gry/Wisp - NOWA wersja w budowie/public/assets/music")
for d in [MUSIC_PENDING_DIR, MUSIC_APPROVED_DIR, GAME_MUSIC_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MUSIC_SCRIPT = BASE_DIR / "generate_music.py"
CONDA_PYTHON = "/root/miniconda3/envs/wisp-music/bin/python3"
_music_generating = False

VIDEO_DIR         = ASSETS_DIR / "video"
VIDEO_PENDING_DIR = VIDEO_DIR / "pending"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PENDING_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_SCRIPT = BASE_DIR / "generate_video.py"
ASSETS_PYTHON = "/root/miniconda3/envs/eagleai-photos/bin/python3"
_video_generating = False

app = FastAPI(title="Wisp Asset Server")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/game-assets", StaticFiles(directory=str(GAME_BG_DIR)), name="game-assets")
app.mount("/music-assets", StaticFiles(directory=str(MUSIC_DIR)), name="music-assets")
app.mount("/video-assets", StaticFiles(directory=str(VIDEO_DIR)), name="video-assets")

# ── SDXL pipeline (lazy load) ────────────────────────────────────────────────
_pipe = None
_pipe_lock = asyncio.Lock()
_generating = False

WORLDS = {
    1: {"name": "Sunny Forest (1-10)", "hint": "sunny enchanted forest, magical glowing trees, sunrays, vibrant green"},
    2: {"name": "Flower Meadow (11-20)", "hint": "magical flower meadow, colorful mushrooms, rainbow light, fairy tale"},
    3: {"name": "Golden Sunrise (21-30)", "hint": "golden sunrise forest, warm amber light, autumn colors, magical"},
    4: {"name": "Winter Wonderland (31-40)", "hint": "bright sparkling winter forest, snow crystals, ice magic, pastel blue"},
}

LAYERS = {
    "full":       "full scene background, all elements",
    "sky":        "sky only, no ground, atmospheric background layer",
    "midground":  "forest midground, trees and foliage, transparent bottom",
    "foreground": "foreground plants and flowers, close up, transparent background",
}

# ── Metadata ─────────────────────────────────────────────────────────────────
def load_meta() -> dict:
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {}

def save_meta(meta: dict):
    META_FILE.write_text(json.dumps(meta, indent=2))

# ── SDXL ─────────────────────────────────────────────────────────────────────
async def get_pipe():
    global _pipe
    async with _pipe_lock:
        if _pipe is None:
            logger.info("Loading SDXL model...")
            from diffusers import StableDiffusionXLPipeline
            _pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to("cuda")
            _pipe.set_progress_bar_config(disable=True)
            logger.info("SDXL ready")
    return _pipe

def unload_pipe():
    global _pipe
    if _pipe is not None:
        del _pipe
        _pipe = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("SDXL unloaded from VRAM")

# ── Generation task ───────────────────────────────────────────────────────────
async def run_generation(job_id: str, prompt: str, world: int, layer: str,
                         count: int, seeds: list[int], width: int, height: int,
                         label: str):
    global _generating
    meta = load_meta()
    meta[job_id]["status"] = "generating"
    save_meta(meta)

    try:
        pipe = await get_pipe()
        world_hint = WORLDS.get(world, {}).get("hint", "")
        layer_hint = LAYERS.get(layer, "")
        full_prompt = (
            f"{prompt}, {world_hint}, {layer_hint}, "
            "2d game background, hand painted, vibrant colors, beautiful, "
            "children game art, soft lighting, magical atmosphere, "
            "high quality, detailed illustration"
        )
        neg = (
            "dark, scary, horror, ugly, text, watermark, blurry, "
            "photorealistic, photograph, 3d render, deformed, human, person"
        )

        files = []
        for i, seed in enumerate(seeds):
            gen = torch.Generator(device="cpu").manual_seed(seed)
            img = pipe(
                prompt=full_prompt,
                negative_prompt=neg,
                width=width,
                height=height,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=gen,
            ).images[0]
            fname = f"{job_id}_{i}.png"
            img.save(str(PENDING_DIR / fname))
            files.append(fname)
            logger.info(f"Generated {fname}")

        meta = load_meta()
        meta[job_id]["status"] = "pending"
        meta[job_id]["files"] = files
        meta[job_id]["finished_at"] = datetime.now().isoformat()
        save_meta(meta)

    except Exception as e:
        logger.exception("Generation failed")
        meta = load_meta()
        if job_id in meta:
            meta[job_id]["status"] = "error"
            meta[job_id]["error"] = str(e)
            save_meta(meta)
    finally:
        _generating = False
        unload_pipe()

# ── API models ────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    world: int = 1
    layer: str = "full"
    count: int = 4
    seed: int = 42
    width: int = 576
    height: int = 1024
    label: str = ""

class ApproveRequest(BaseModel):
    job_id: str
    file_index: int
    target: str  # e.g. "world1", "level5", "world2_sky"

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def panel():
    return FileResponse(str(BASE_DIR / "panel.html"))

@app.get("/api/status")
async def status():
    return {"generating": _generating, "pipe_loaded": _pipe is not None}

@app.get("/api/worlds")
async def worlds():
    return WORLDS

@app.post("/api/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    global _generating
    if _generating:
        raise HTTPException(409, "Generation already in progress, please wait")

    _generating = True
    job_id = str(uuid.uuid4())[:8]
    seeds = [req.seed + i * 100 for i in range(req.count)]

    meta = load_meta()
    meta[job_id] = {
        "job_id": job_id,
        "prompt": req.prompt,
        "world": req.world,
        "layer": req.layer,
        "count": req.count,
        "seeds": seeds,
        "width": req.width,
        "height": req.height,
        "label": req.label,
        "status": "queued",
        "files": [],
        "created_at": datetime.now().isoformat(),
    }
    save_meta(meta)

    background_tasks.add_task(
        run_generation, job_id, req.prompt, req.world, req.layer,
        req.count, seeds, req.width, req.height, req.label
    )
    return {"job_id": job_id, "message": "Generation started"}

@app.get("/api/jobs")
async def list_jobs(status: Optional[str] = None):
    meta = load_meta()
    jobs = list(meta.values())
    if status:
        jobs = [j for j in jobs if j.get("status") == status]
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    meta = load_meta()
    if job_id not in meta:
        raise HTTPException(404)
    return meta[job_id]

@app.post("/api/approve")
async def approve(req: ApproveRequest):
    meta = load_meta()
    if req.job_id not in meta:
        raise HTTPException(404, "Job not found")
    job = meta[req.job_id]
    if req.file_index >= len(job.get("files", [])):
        raise HTTPException(400, "Invalid file index")

    fname = job["files"][req.file_index]
    src = PENDING_DIR / fname
    if not src.exists():
        raise HTTPException(404, "File not found")

    # Copy to approved dir
    dst_name = f"{req.target}_{fname}"
    shutil.copy(str(src), str(APPROVED_DIR / dst_name))

    # Copy to game public dir
    game_dst = GAME_BG_DIR / f"{req.target}.png"
    shutil.copy(str(src), str(game_dst))

    meta[req.job_id]["approved"] = meta[req.job_id].get("approved", [])
    meta[req.job_id]["approved"].append({"index": req.file_index, "target": req.target})
    save_meta(meta)

    return {"ok": True, "game_path": f"/assets/bg/{req.target}.png"}

@app.post("/api/reject/{job_id}/{file_index}")
async def reject(job_id: str, file_index: int):
    meta = load_meta()
    if job_id not in meta:
        raise HTTPException(404)
    job = meta[job_id]
    if file_index >= len(job.get("files", [])):
        raise HTTPException(400)

    fname = job["files"][file_index]
    src = PENDING_DIR / fname
    if src.exists():
        shutil.move(str(src), str(REJECTED_DIR / fname))

    meta[job_id]["rejected"] = meta[job_id].get("rejected", [])
    meta[job_id]["rejected"].append(file_index)
    save_meta(meta)
    return {"ok": True}

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    meta = load_meta()
    if job_id not in meta:
        raise HTTPException(404)
    job = meta[job_id]
    for fname in job.get("files", []):
        for d in [PENDING_DIR, APPROVED_DIR, REJECTED_DIR]:
            p = d / fname
            if p.exists():
                p.unlink()
    del meta[job_id]
    save_meta(meta)
    return {"ok": True}

@app.get("/api/game-assets")
async def game_assets():
    files = sorted(GAME_BG_DIR.glob("*.png"))
    return [{"name": f.stem, "url": f"/game-assets/{f.name}", "size": f.stat().st_size} for f in files]

class FileApproveRequest(BaseModel):
    filename: str
    target: str

class FileRejectRequest(BaseModel):
    filename: str

@app.post("/api/approve-file")
async def approve_file(req: FileApproveRequest):
    src = PENDING_DIR / req.filename
    if not src.exists():
        raise HTTPException(404, "File not found")
    dst_name = f"{req.target}_{req.filename}"
    shutil.copy(str(src), str(APPROVED_DIR / dst_name))
    game_dst = GAME_BG_DIR / f"{req.target}.png"
    shutil.copy(str(src), str(game_dst))
    meta = load_meta()
    fmeta = meta.setdefault("_file_approvals", {})
    fmeta[req.filename] = req.target
    save_meta(meta)
    return {"ok": True, "game_path": f"/assets/bg/{req.target}.png"}

@app.post("/api/reject-file")
async def reject_file(req: FileRejectRequest):
    src = PENDING_DIR / req.filename
    if src.exists():
        shutil.move(str(src), str(REJECTED_DIR / req.filename))
    meta = load_meta()
    fmeta = meta.setdefault("_file_rejections", [])
    if req.filename not in fmeta:
        fmeta.append(req.filename)
    save_meta(meta)
    return {"ok": True}

@app.get("/api/pending-files")
async def pending_files():
    """Skanuje folder pending i zwraca grafiki z metadanymi zatwierdzeń."""
    meta = load_meta()
    approved_map: dict = dict(meta.get("_file_approvals", {}))
    rejected_set: set = set(meta.get("_file_rejections", []))
    for job in meta.values():
        if not isinstance(job, dict):
            continue
        for a in job.get("approved", []):
            idx = a["index"]
            fname = job["files"][idx] if idx < len(job.get("files", [])) else None
            if fname:
                approved_map[fname] = a["target"]
        for idx in job.get("rejected", []):
            if idx < len(job.get("files", [])):
                rejected_set.add(job["files"][idx])

    import re
    result = []
    for f in sorted(PENDING_DIR.glob("*.png")):
        m = re.match(r"level(\d+)_(w\d)", f.name)
        level = int(m.group(1)) if m else None
        world = m.group(2) if m else None
        result.append({
            "name": f.name,
            "url": f"/assets/pending/{f.name}",
            "level": level,
            "world": world,
            "approved_as": approved_map.get(f.name),
            "rejected": f.name in rejected_set,
        })
    return result

# ── Music API ─────────────────────────────────────────────────────────────────

class MusicGenerateRequest(BaseModel):
    prompt: str
    duration: int = 30
    world: str = "w1"
    label: str = ""

def load_music_meta() -> dict:
    mf = MUSIC_DIR / "meta.json"
    return json.loads(mf.read_text()) if mf.exists() else {}

def save_music_meta(meta: dict):
    (MUSIC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

async def run_music_generation(job_id: str, prompt: str, duration: int, world: str, label: str):
    global _music_generating
    out_path = str(MUSIC_PENDING_DIR / f"{job_id}.mp3")
    status_path = str(MUSIC_DIR / f"{job_id}_status.json")

    meta = load_music_meta()
    meta[job_id]["status"] = "generating"
    save_music_meta(meta)

    try:
        proc = await asyncio.create_subprocess_exec(
            CONDA_PYTHON, str(MUSIC_SCRIPT),
            job_id, prompt, str(duration), out_path, status_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            meta = load_music_meta()
            meta[job_id]["status"] = "pending"
            meta[job_id]["file"] = f"{job_id}.mp3"
            meta[job_id]["finished_at"] = datetime.now().isoformat()
            save_music_meta(meta)
        else:
            raise RuntimeError(stderr.decode()[-500:])
    except Exception as e:
        logger.exception("Music generation failed")
        meta = load_music_meta()
        if job_id in meta:
            meta[job_id]["status"] = "error"
            meta[job_id]["error"] = str(e)[-300:]
            save_music_meta(meta)
    finally:
        _music_generating = False
        Path(status_path).unlink(missing_ok=True)

@app.post("/api/music/generate")
async def music_generate(req: MusicGenerateRequest, background_tasks: BackgroundTasks):
    global _music_generating
    if _music_generating:
        raise HTTPException(409, "Generowanie muzyki już trwa, poczekaj")
    _music_generating = True
    job_id = "mus_" + str(uuid.uuid4())[:8]
    meta = load_music_meta()
    meta[job_id] = {
        "job_id": job_id, "prompt": req.prompt, "duration": req.duration,
        "world": req.world, "label": req.label,
        "status": "queued", "file": None,
        "created_at": datetime.now().isoformat(),
    }
    save_music_meta(meta)
    background_tasks.add_task(run_music_generation, job_id, req.prompt, req.duration, req.world, req.label)
    return {"job_id": job_id}

@app.get("/api/music/jobs")
async def music_jobs():
    meta = load_music_meta()
    jobs = sorted(meta.values(), key=lambda j: j.get("created_at",""), reverse=True)
    return jobs

@app.get("/api/music/status/{job_id}")
async def music_status(job_id: str):
    sf = MUSIC_DIR / f"{job_id}_status.json"
    if sf.exists():
        return json.loads(sf.read_text())
    meta = load_music_meta()
    return meta.get(job_id, {"status": "unknown"})

@app.post("/api/music/approve/{job_id}")
async def music_approve(job_id: str, target: str):
    meta = load_music_meta()
    if job_id not in meta:
        raise HTTPException(404)
    job = meta[job_id]
    fname = job.get("file")
    if not fname:
        raise HTTPException(400, "Brak pliku")
    src = MUSIC_PENDING_DIR / fname
    if not src.exists():
        raise HTTPException(404, "Plik nie istnieje")
    dst_name = f"{target}.mp3"
    shutil.copy(str(src), str(MUSIC_APPROVED_DIR / dst_name))
    shutil.copy(str(src), str(GAME_MUSIC_DIR / dst_name))
    meta[job_id]["approved_as"] = target
    meta[job_id]["status"] = "approved"
    save_music_meta(meta)
    return {"ok": True, "path": f"assets/music/{dst_name}"}

@app.delete("/api/music/jobs/{job_id}")
async def music_delete(job_id: str):
    meta = load_music_meta()
    if job_id not in meta:
        raise HTTPException(404)
    fname = meta[job_id].get("file")
    if fname:
        for d in [MUSIC_PENDING_DIR, MUSIC_APPROVED_DIR]:
            (d / fname).unlink(missing_ok=True)
    del meta[job_id]
    save_music_meta(meta)
    return {"ok": True}

# ── Video API ─────────────────────────────────────────────────────────────────
def load_video_meta() -> dict:
    mf = VIDEO_DIR / "meta.json"
    return json.loads(mf.read_text()) if mf.exists() else {}

def save_video_meta(meta: dict):
    (VIDEO_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

class VideoGenRequest(BaseModel):
    prompt: str
    num_frames: int = 49
    width: int = 832
    height: int = 480
    guidance_scale: float = 5.0
    label: str = ""
    upscale: str = ""   # "" | "fhd" | "2k" | "4k"

async def run_video_generation(job_id: str, req: VideoGenRequest):
    global _video_generating
    out_path = str(VIDEO_PENDING_DIR / f"{job_id}.mp4")
    status_path = str(VIDEO_DIR / f"{job_id}_status.json")

    meta = load_video_meta()
    meta[job_id]["status"] = "generating"
    save_video_meta(meta)

    try:
        proc = await asyncio.create_subprocess_exec(
            ASSETS_PYTHON, str(VIDEO_SCRIPT),
            job_id, req.prompt,
            str(req.num_frames), str(req.width), str(req.height),
            str(req.guidance_scale), out_path, status_path, req.upscale,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            meta = load_video_meta()
            meta[job_id]["status"] = "pending"
            meta[job_id]["file"] = f"{job_id}.mp4"
            meta[job_id]["finished_at"] = datetime.now().isoformat()
            save_video_meta(meta)
        else:
            raise RuntimeError(stderr.decode()[-800:])
    except Exception as e:
        logger.exception("Video generation failed")
        meta = load_video_meta()
        if job_id in meta:
            meta[job_id]["status"] = "error"
            meta[job_id]["error"] = str(e)[-400:]
            save_video_meta(meta)
    finally:
        _video_generating = False
        Path(status_path).unlink(missing_ok=True)

@app.post("/api/video/generate")
async def video_generate(req: VideoGenRequest, background_tasks: BackgroundTasks):
    global _video_generating
    if _video_generating:
        raise HTTPException(409, "Generacja wideo już trwa")
    _video_generating = True
    job_id = "vid_" + uuid.uuid4().hex[:8]
    meta = load_video_meta()
    meta[job_id] = {
        "job_id": job_id,
        "prompt": req.prompt,
        "num_frames": req.num_frames,
        "width": req.width,
        "height": req.height,
        "label": req.label,
        "upscale": req.upscale,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "finished_at": None,
    }
    save_video_meta(meta)
    background_tasks.add_task(run_video_generation, job_id, req)
    return {"job_id": job_id}

@app.get("/api/video/jobs")
async def video_jobs():
    meta = load_video_meta()
    return sorted(meta.values(), key=lambda j: j.get("created_at", ""), reverse=True)

@app.get("/api/video/status/{job_id}")
async def video_status(job_id: str):
    sf = VIDEO_DIR / f"{job_id}_status.json"
    if sf.exists():
        return json.loads(sf.read_text())
    meta = load_video_meta()
    return meta.get(job_id, {"status": "unknown"})

@app.delete("/api/video/jobs/{job_id}")
async def video_delete(job_id: str):
    meta = load_video_meta()
    if job_id not in meta:
        raise HTTPException(404)
    fname = meta[job_id].get("file")
    if fname:
        (VIDEO_PENDING_DIR / fname).unlink(missing_ok=True)
    del meta[job_id]
    save_video_meta(meta)
    return {"ok": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3004, log_level="info")
