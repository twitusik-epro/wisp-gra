"""
Generuje wideo CogVideoX-2b.
Uruchamiany przez server.py jako subprocess w środowisku eagleai-photos.
Argumenty: job_id prompt num_frames width height guidance_scale out_path status_path
"""
import sys, json, time, warnings, threading, os
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

job_id         = sys.argv[1]
prompt         = sys.argv[2]
num_frames     = int(sys.argv[3])
width          = int(sys.argv[4])
height         = int(sys.argv[5])
guidance_scale = float(sys.argv[6])
out_path       = sys.argv[7]
status_f       = sys.argv[8]
upscale        = sys.argv[9] if len(sys.argv) > 9 else ""  # "" | "fhd" | "2k" | "4k"

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

upd({"status": "loading", "progress": 5})

import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

upd({"status": "loading", "progress": 15})

_stop_load = threading.Event()
threading.Thread(target=ticker, args=("loading", 15, 38, 2, _stop_load), daemon=True).start()

pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16,
)
# Sequential offload: każdy moduł ładowany do GPU tylko na czas forward pass → szczyt ~4-6 GB VRAM
pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()    # dekoduje wideo w plasterkach
pipe.vae.enable_tiling()     # przetwarza w kafelkach

_stop_load.set()
upd({"status": "generating", "progress": 40})

_stop_gen = threading.Event()
threading.Thread(target=ticker, args=("generating", 40, 85, 3, _stop_gen), daemon=True).start()

output = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    guidance_scale=guidance_scale,
    num_inference_steps=50,
    use_dynamic_cfg=True,
)

_stop_gen.set()
upd({"status": "saving", "progress": 90})

export_to_video(output.frames[0], out_path, fps=8)

del pipe
import gc
gc.collect()
torch.cuda.empty_cache()

# ─── Upscaling ffmpeg (lanczos) ───────────────────────────────────
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

upd({"status": "done", "progress": 100, "file": out_path})
print(f"OK: {out_path}")
