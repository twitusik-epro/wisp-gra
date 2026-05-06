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
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
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

MUSIC_SCRIPT = BASE_DIR / "generate_music_acestep.py"
CONDA_PYTHON = "/root/miniconda3/envs/wisp-music/bin/python3"
_music_generating = False

VIDEO_DIR         = ASSETS_DIR / "video"
VIDEO_PENDING_DIR = VIDEO_DIR / "pending"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_PENDING_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_SCRIPT    = BASE_DIR / "generate_video.py"
VIDEO_14B_SCRIPT= BASE_DIR / "generate_video_14b.py"
VIDEO_I2V_SCRIPT= BASE_DIR / "generate_video_i2v.py"
SVD_SCRIPT      = BASE_DIR / "generate_video_svd.py"
ASSETS_PYTHON  = "/root/miniconda3/envs/eagleai-photos/bin/python3"
UPLOADS_DIR    = VIDEO_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
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
    duration: float = 5
    top_k: int = 250
    world: str = "w1"
    label: str = ""

def load_music_meta() -> dict:
    mf = MUSIC_DIR / "meta.json"
    return json.loads(mf.read_text()) if mf.exists() else {}

def save_music_meta(meta: dict):
    (MUSIC_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

async def run_music_generation(job_id: str, prompt: str, duration: float, top_k: int, world: str, label: str):
    global _music_generating
    out_path = str(MUSIC_PENDING_DIR / f"{job_id}.mp3")
    status_path = str(MUSIC_DIR / f"{job_id}_status.json")

    meta = load_music_meta()
    meta[job_id]["status"] = "generating"
    save_music_meta(meta)

    try:
        proc = await asyncio.create_subprocess_exec(
            CONDA_PYTHON, str(MUSIC_SCRIPT),
            job_id, prompt, str(duration), str(top_k), out_path, status_path,
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
        "top_k": req.top_k, "world": req.world, "label": req.label,
        "status": "queued", "file": None,
        "created_at": datetime.now().isoformat(),
    }
    save_music_meta(meta)
    background_tasks.add_task(run_music_generation, job_id, req.prompt, req.duration, req.top_k, req.world, req.label)
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
    model: str = "wan"        # "wan" | "wan_i2v" | "svd"
    wan_quality: str = "1.3B" # "1.3B" | "14B"
    prompt: str = ""
    image_path: str = ""      # I2V/SVD — server-side path after upload
    num_frames: int = 81      # wan: 4k+1 format; svd: 14 or 25
    width: int = 480
    height: int = 832
    max_area: int = 399360    # I2V: 480*832=399360 | 720P: 921600
    guidance_scale: float = 5.0
    negative_prompt: str = "blurry, low quality, distorted, watermark, text, static, ugly, deformed, flickering"
    motion_bucket: int = 127  # SVD only, 0-255
    label: str = ""
    upscale: str = ""         # "" | "fhd" | "2k" | "4k"
    seed: int = -1            # -1 = random

async def _release_ollama():
    """Tell Ollama to evict all loaded models from VRAM, then wait until free."""
    import json as _json
    import urllib.request as _ur
    try:
        # Get list of currently loaded models
        resp = _ur.urlopen("http://localhost:11434/api/ps", timeout=5)
        ps = _json.loads(resp.read())
        models = [m["model"] for m in ps.get("models", [])]
    except Exception:
        models = []

    for model_name in models:
        try:
            data = _json.dumps({"model": model_name, "keep_alive": 0}).encode()
            req_http = _ur.Request(
                "http://localhost:11434/api/generate",
                data=data, method="POST",
                headers={"Content-Type": "application/json"}
            )
            _ur.urlopen(req_http, timeout=10)
        except Exception:
            pass

    if models:
        # Wait up to 15s for Ollama to actually release VRAM
        import subprocess as _sp
        for _ in range(15):
            await asyncio.sleep(1)
            try:
                resp = _ur.urlopen("http://localhost:11434/api/ps", timeout=3)
                still = _json.loads(resp.read()).get("models", [])
                if not still:
                    break
            except Exception:
                break

async def _free_vram_for_14b():
    """Stop eagleai-photos to free ~13GB VRAM, release Ollama."""
    await _release_ollama()
    proc = await asyncio.create_subprocess_exec(
        "systemctl", "stop", "eagleai-photos.service",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()
    await asyncio.sleep(2)

async def _restore_after_14b():
    """Restart eagleai-photos after 14B generation finishes."""
    proc = await asyncio.create_subprocess_exec(
        "systemctl", "start", "eagleai-photos.service",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL
    )
    await proc.wait()

async def run_video_generation(job_id: str, req: VideoGenRequest):
    global _video_generating
    out_path = str(VIDEO_PENDING_DIR / f"{job_id}.mp4")
    status_path = str(VIDEO_DIR / f"{job_id}_status.json")

    meta = load_video_meta()
    meta[job_id]["status"] = "generating"
    save_video_meta(meta)

    try:
        if req.model == "svd":
            proc = await asyncio.create_subprocess_exec(
                ASSETS_PYTHON, str(SVD_SCRIPT),
                job_id, req.image_path,
                str(req.num_frames), str(req.width), str(req.height),
                str(req.motion_bucket), str(req.seed),
                out_path, status_path, req.upscale,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        elif req.model == "wan_i2v":
            await _release_ollama()
            proc = await asyncio.create_subprocess_exec(
                ASSETS_PYTHON, str(VIDEO_I2V_SCRIPT),
                job_id, req.prompt, req.image_path,
                str(req.num_frames), str(req.max_area),
                str(req.guidance_scale), req.negative_prompt,
                out_path, status_path, req.upscale, str(req.seed),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        elif req.wan_quality == "14B":
            await _release_ollama()
            proc = await asyncio.create_subprocess_exec(
                ASSETS_PYTHON, str(VIDEO_14B_SCRIPT),
                job_id, req.prompt,
                str(req.num_frames), str(req.width), str(req.height),
                str(req.guidance_scale), req.negative_prompt, out_path, status_path, req.upscale, str(req.seed),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                ASSETS_PYTHON, str(VIDEO_SCRIPT),
                job_id, req.prompt,
                str(req.num_frames), str(req.width), str(req.height),
                str(req.guidance_scale), req.negative_prompt, out_path, status_path, req.upscale, str(req.seed),
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
        pass  # eagleai-photos stays running (14B uses only ~4GB VRAM via block offloading)


async def _monitor_orphan_video(job_id: str, pid: int):
    """Wait for orphaned video subprocess (survived server restart) and update meta."""
    global _video_generating
    _video_generating = True
    logger.info(f"Monitoring orphan PID={pid} for video job {job_id}")
    try:
        while True:
            try:
                os.kill(pid, 0)
                await asyncio.sleep(5)
            except ProcessLookupError:
                break
        await asyncio.sleep(1)
        mp4 = VIDEO_PENDING_DIR / f"{job_id}.mp4"
        meta = load_video_meta()
        (VIDEO_DIR / f"{job_id}_status.json").unlink(missing_ok=True)
        if mp4.exists() and mp4.stat().st_size > 100_000:
            meta[job_id]["status"] = "pending"
            meta[job_id]["file"] = f"{job_id}.mp4"
            meta[job_id]["finished_at"] = datetime.now().isoformat()
            logger.info(f"Orphan job {job_id} recovered successfully")
        else:
            meta[job_id]["status"] = "error"
            meta[job_id]["error"] = "Generation failed (orphaned process)"
            logger.info(f"Orphan job {job_id} failed")
        save_video_meta(meta)
    except Exception:
        logger.exception(f"Error monitoring orphan {job_id}")
    finally:
        _video_generating = False


@app.on_event("startup")
async def startup_recovery():
    """On startup, recover video jobs stuck in 'generating' (server was restarted mid-job)."""
    import subprocess as _sp2
    meta = load_video_meta()
    changed = False
    for job_id, job in list(meta.items()):
        if job.get("status") != "generating":
            continue
        try:
            result = _sp2.run(["pgrep", "-f", job_id], capture_output=True, text=True)
            if result.returncode == 0:
                pid = int(result.stdout.strip().split("\n")[0])
                asyncio.ensure_future(_monitor_orphan_video(job_id, pid))
                logger.info(f"Startup recovery: monitoring orphan PID={pid} for {job_id}")
            else:
                meta[job_id]["status"] = "error"
                meta[job_id]["error"] = "Server restarted during generation"
                (VIDEO_DIR / f"{job_id}_status.json").unlink(missing_ok=True)
                changed = True
                logger.info(f"Startup recovery: marked {job_id} as error (process gone)")
        except Exception:
            logger.exception(f"Startup recovery error for {job_id}")
    if changed:
        save_video_meta(meta)


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
        "model": req.model,
        "wan_quality": req.wan_quality,
        "prompt": req.prompt,
        "image_path": req.image_path,
        "num_frames": req.num_frames,
        "width": req.width,
        "height": req.height,
        "motion_bucket": req.motion_bucket,
        "label": req.label,
        "upscale": req.upscale,
        "seed": req.seed,
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

@app.post("/api/video/upload-image")
async def upload_image(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(400, "Nieobsługiwany format. Użyj JPG, PNG lub WEBP.")
    fname = f"upload_{uuid.uuid4().hex[:8]}{ext}"
    dest = UPLOADS_DIR / fname
    dest.write_bytes(await file.read())
    return {"path": str(dest), "url": f"/video-assets/uploads/{fname}"}

@app.get("/api/video/frame/{job_id}")
async def extract_frame(job_id: str):
    meta = load_video_meta()
    if job_id not in meta:
        raise HTTPException(404)
    job = meta[job_id]
    if job.get("status") != "pending" or not job.get("file"):
        raise HTTPException(400, "Film nie jest gotowy")
    video_path = VIDEO_PENDING_DIR / job["file"]
    if not video_path.exists():
        raise HTTPException(404, "Plik nie istnieje")
    frame_fname = f"frame_{job_id}.jpg"
    frame_path = UPLOADS_DIR / frame_fname
    import subprocess
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vframes", "1", "-q:v", "2", str(frame_path)
    ], capture_output=True, check=True)
    return {"path": str(frame_path), "url": f"/video-assets/uploads/{frame_fname}"}

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

class MergeRequest(BaseModel):
    video_job_id: str
    music_job_id: str
    crossfade: bool = False

@app.post("/api/video/merge")
async def video_merge(req: MergeRequest):
    vmeta = load_video_meta()
    mmeta = load_music_meta()
    vj = vmeta.get(req.video_job_id)
    mj = mmeta.get(req.music_job_id)
    if not vj or vj.get("status") != "pending" or not vj.get("file"):
        raise HTTPException(400, "Wideo nie jest gotowe")
    if not mj or mj.get("status") not in ("pending", "approved") or not mj.get("file"):
        raise HTTPException(400, "Muzyka nie jest gotowa")
    video_path = VIDEO_PENDING_DIR / vj["file"]
    music_path = MUSIC_PENDING_DIR / mj["file"]
    if not video_path.exists():
        raise HTTPException(404, "Plik wideo nie istnieje")
    if not music_path.exists():
        raise HTTPException(404, "Plik muzyki nie istnieje")

    # get video duration for crossfade fade-out start point
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
    ], capture_output=True, text=True)
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else None

    merged_id = "mrg_" + uuid.uuid4().hex[:8]
    out_path = VIDEO_PENDING_DIR / f"{merged_id}.mp4"

    if req.crossfade and duration:
        fade_dur = 0.5
        fade_out_start = max(0.0, duration - fade_dur)
        audio_filter = (
            f"[1:a]afade=t=in:st=0:d={fade_dur},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_dur}[aout]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", audio_filter,
            "-map", "0:v:0", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(out_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-map", "0:v:0", "-map", "1:a:0",
            str(out_path)
        ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise HTTPException(500, result.stderr.decode()[-300:])
    vmeta[merged_id] = {
        "job_id": merged_id,
        "model": vj.get("model", "wan"),
        "prompt": f"[MONTAŻ] {vj.get('prompt','')} + {mj.get('prompt','')[:40]}",
        "num_frames": vj.get("num_frames", 0),
        "width": vj.get("width", 0),
        "height": vj.get("height", 0),
        "upscale": vj.get("upscale", ""),
        "seed": vj.get("seed", -1),
        "label": vj.get("label", ""),
        "status": "pending",
        "file": f"{merged_id}.mp4",
        "created_at": datetime.now().isoformat(),
        "finished_at": datetime.now().isoformat(),
    }
    save_video_meta(vmeta)
    return {"job_id": merged_id, "url": f"/video-assets/pending/{merged_id}.mp4"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3004, log_level="info")
