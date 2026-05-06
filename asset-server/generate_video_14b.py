"""
Wan2.1-T2V-14B video generation with dit_cpu=True block offloading.
DiT blocks stay on CPU and move to GPU one at a time via accelerate hooks.
Peak VRAM: ~4-6GB (VAE + 1 block + activations).
Args: job_id prompt num_frames width height guidance_scale negative_prompt out_path status_path [upscale] [seed]
"""
import sys, json, time, warnings, threading, os, gc
warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/opt/Wan2.1")

MODEL_DIR = "/opt/models/Wan2.1-T2V-14B"
FPS = 16

job_id         = sys.argv[1]
prompt         = sys.argv[2]
num_frames     = int(sys.argv[3])
width          = int(sys.argv[4])
height         = int(sys.argv[5])
guidance_scale  = float(sys.argv[6])
negative_prompt = sys.argv[7]
out_path        = sys.argv[8]
status_f        = sys.argv[9]
upscale         = sys.argv[10] if len(sys.argv) > 10 else ""
seed_arg        = int(sys.argv[11]) if len(sys.argv) > 11 else -1

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
    k = max(2, round((n - 1) / 4))
    return 4 * k + 1

upd({"status": "loading", "progress": 3})

import torch

_stop_load = threading.Event()
threading.Thread(target=ticker, args=("loading", 3, 38, 3, _stop_load), daemon=True).start()

import wan
from wan.configs import WAN_CONFIGS

# Load model with dit_cpu=True — DiT blocks stay on CPU, never loaded to GPU
# patch_embedding / text_embedding / head etc. will be moved to GPU manually
model = wan.WanT2V(
    config=WAN_CONFIGS['t2v-14B'],
    checkpoint_dir=MODEL_DIR,
    device_id=0,
    t5_fsdp=False,
    dit_fsdp=False,
    use_usp=False,
    t5_cpu=True,
    dit_cpu=True,   # patched into text2video.py — keeps blocks on CPU
)

# Move non-block parts of WanModel to GPU (they're small: ~500MB total)
# WanModel.forward() uses self.patch_embedding.weight.device to detect device
# and auto-moves self.freqs to match.
gpu = torch.device("cuda:0")
m = model.model  # WanModel instance (on CPU)
m.patch_embedding.to(gpu)
m.text_embedding.to(gpu)
m.time_embedding.to(gpu)
m.time_projection.to(gpu)
m.head.to(gpu)

# Manual block offloading via PyTorch hooks (no accelerate — avoids meta-device conflicts).
# pre_forward: CPU → GPU; post_forward: GPU → CPU + empty cache.
# Each block stays on CPU between calls; peak VRAM ≈ VAE + 1 block + activations (~4-6 GB).
_block_call_counter = [0]

def _apply_offload_hooks(block, device):
    def _pre(module, args):
        module.to(device)
    def _post(module, args, output):
        module.to('cpu')
        _block_call_counter[0] += 1
        if _block_call_counter[0] % 40 == 0:
            torch.cuda.empty_cache()
        return output
    block.register_forward_pre_hook(_pre)
    block.register_forward_hook(_post)

for block in m.blocks:
    _apply_offload_hooks(block, gpu)

_stop_load.set()

valid_frames = snap_frames(num_frames)

upd({"status": "generating", "progress": 40, "seed": actual_seed})
_stop_gen = threading.Event()
threading.Thread(target=ticker, args=("generating", 40, 85, 8, _stop_gen), daemon=True).start()

# generate() will NOT call self.model.to(self.device) (patched out via dit_cpu=True)
# Accelerate hooks on each block handle GPU movement per-block
video = model.generate(
    prompt,
    size=(width, height),
    frame_num=valid_frames,
    sampling_steps=50,
    sample_solver='unipc',
    shift=5.0,
    guide_scale=guidance_scale,
    n_prompt=negative_prompt,
    seed=actual_seed,
    offload_model=False,
)

_stop_gen.set()
upd({"status": "saving", "progress": 90})

video = video.cpu()
video = (video.clamp(-1, 1) + 1) / 2
video = (video * 255).to(torch.uint8)
video = video.permute(1, 2, 3, 0).numpy()
out_frames = [video[i] for i in range(video.shape[0])]

import imageio.v2 as imageio
imageio.mimwrite(out_path, out_frames, fps=FPS, codec='libx264', quality=7, pixelformat='yuv420p')

del model
gc.collect()
torch.cuda.empty_cache()

# ─── Upscaling ────────────────────────────────────────────────────────────────
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
