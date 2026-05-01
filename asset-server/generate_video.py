"""
Wan2.1-T2V-1.3B video generation.
Args: job_id prompt num_frames width height guidance_scale out_path status_path [upscale]
"""
import sys, json, time, warnings, threading, os
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_DIR = "/opt/models/Wan2.1-T2V-1.3B"
FPS = 16

job_id         = sys.argv[1]
prompt         = sys.argv[2]
num_frames     = int(sys.argv[3])
width          = int(sys.argv[4])
height         = int(sys.argv[5])
guidance_scale = float(sys.argv[6])
out_path       = sys.argv[7]
status_f       = sys.argv[8]
upscale        = sys.argv[9] if len(sys.argv) > 9 else ""
seed_arg       = int(sys.argv[10]) if len(sys.argv) > 10 else -1

def upd(s):
    with open(status_f, 'w') as f:
        json.dump(s, f)

def ticker(label, p_start, p_end, interval, stop_evt):
    p = p_start
    step = max(1, (p_end - p_start) // 15)
    while not stop_evt.is_set() and p < p_end:
        time.sleep(interval)
        if stop_evt.is_set(): break
        p = min(p_end, p + step)
        upd({"status": label, "progress": p})

import random as _random
actual_seed = seed_arg if seed_arg >= 0 else _random.randint(0, 2**32 - 1)

def snap_frames(n):
    """Snap to nearest valid Wan2.1 frame count (4k+1)."""
    k = max(2, round((n - 1) / 4))
    return 4 * k + 1

upd({"status": "loading", "progress": 5})

import torch
import numpy as np

_stop_load = threading.Event()
threading.Thread(target=ticker, args=("loading", 5, 38, 2, _stop_load), daemon=True).start()

import wan
from wan.configs import WAN_CONFIGS

# dtype comes from config.param_dtype — no dtype arg in constructor
# t5_cpu=True: T5 text encoder stays on CPU (~6GB VRAM saved), needed since autophotos.ai uses ~10GB
model = wan.WanT2V(
    config=WAN_CONFIGS['t2v-1.3B'],
    checkpoint_dir=MODEL_DIR,
    device_id=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_usp=False,
    t5_cpu=True,
)

_stop_load.set()

valid_frames = snap_frames(num_frames)

upd({"status": "generating", "progress": 40, "seed": actual_seed})
_stop_gen = threading.Event()
threading.Thread(target=ticker, args=("generating", 40, 85, 4, _stop_gen), daemon=True).start()

# size=(width, height) tuple, returns tensor (C, N, H, W) in range [-1, 1]
video = model.generate(
    prompt,
    size=(width, height),
    frame_num=valid_frames,
    sampling_steps=50,
    sample_solver='unipc',
    shift=5.0,
    guide_scale=guidance_scale,
    seed=actual_seed,
    offload_model=True,
)

_stop_gen.set()
upd({"status": "saving", "progress": 90})

# (C, N, H, W) [-1,1] → (N, H, W, C) uint8
video = video.cpu()
video = (video.clamp(-1, 1) + 1) / 2          # → [0, 1]
video = (video * 255).to(torch.uint8)          # → [0, 255]
video = video.permute(1, 2, 3, 0).numpy()      # → (N, H, W, C)
out_frames = [video[i] for i in range(video.shape[0])]

import imageio.v2 as imageio
imageio.mimwrite(out_path, out_frames, fps=FPS, codec='libx264', quality=7, pixelformat='yuv420p')

del model
import gc
gc.collect()
torch.cuda.empty_cache()

# ─── Upscaling ffmpeg (lanczos) ───────────────────────────────────────────────
if upscale:
    upd({"status": "upscaling", "progress": 92})
    SCALES = {
        "fhd": {"landscape": "1920:1080", "portrait": "1080:1920", "square": "1080:1080"},
        "2k":  {"landscape": "2560:1440", "portrait": "1440:2560", "square": "1440:1440"},
        "4k":  {"landscape": "3840:2160", "portrait": "2160:3840", "square": "2160:2160"},
    }
    orient = "landscape" if width > height else ("portrait" if height > width else "square")
    target = SCALES.get(upscale, {}).get(orient)
    if target:
        import subprocess
        tmp = out_path.replace(".mp4", "_base.mp4")
        os.rename(out_path, tmp)
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp,
            "-vf", f"scale={target}:flags=lanczos",
            "-c:v", "libx264", "-preset", "fast", "-crf", "17",
            "-c:a", "copy", out_path
        ], capture_output=True, check=True)
        os.unlink(tmp)

upd({"status": "done", "progress": 100, "file": out_path, "seed": actual_seed})
print(f"OK: {out_path}")
