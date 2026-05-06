"""
ACE-Step music generation (replaces MusicGen-small).
Args: job_id prompt duration_sec infer_steps out_path status_path
"""
import sys, json, os, subprocess, gc, threading, time
sys.path.insert(0, "/opt/ACE-Step")

job_id     = sys.argv[1]
prompt     = sys.argv[2]
duration   = float(sys.argv[3])
infer_steps= int(sys.argv[4]) if len(sys.argv) > 4 else 30
out_path   = sys.argv[5]
status_f   = sys.argv[6]

def upd(s):
    with open(status_f, 'w') as f:
        json.dump(s, f)

def ticker(label, p_start, p_end, interval, stop_evt):
    p = p_start
    step = max(1, (p_end - p_start) // 20)
    while not stop_evt.is_set() and p < p_end:
        time.sleep(interval)
        if stop_evt.is_set():
            break
        p = min(p_end, p + step)
        upd({"status": label, "progress": p})

upd({"status": "loading", "progress": 3})

import torch
from acestep.pipeline_ace_step import ACEStepPipeline

_stop_load = threading.Event()
threading.Thread(target=ticker, args=("loading", 3, 38, 2, _stop_load), daemon=True).start()

pipe = ACEStepPipeline(
    checkpoint_dir="",
    dtype="bfloat16",
    torch_compile=False,
    cpu_offload=False,
)

_stop_load.set()
upd({"status": "generating", "progress": 40})

_stop_gen = threading.Event()
threading.Thread(target=ticker, args=("generating", 40, 88, 1.5, _stop_gen), daemon=True).start()

tmp_wav = out_path.replace(".mp3", "_tmp.wav")

pipe(
    audio_duration=duration,
    prompt=prompt,
    lyrics="[instrumental]",
    infer_step=infer_steps,
    guidance_scale=7.0,
    scheduler_type="euler",
    cfg_type="apg",
    omega_scale=10.0,
    manual_seeds="-1",
    guidance_interval=0.5,
    guidance_interval_decay=0.0,
    min_guidance_scale=3.0,
    use_erg_tag=True,
    use_erg_lyric=True,
    use_erg_diffusion=True,
    oss_steps="",
    guidance_scale_text=0.0,
    guidance_scale_lyric=0.0,
    save_path=tmp_wav,
)

_stop_gen.set()
upd({"status": "saving", "progress": 92})

subprocess.run(
    ["ffmpeg", "-y", "-i", tmp_wav, "-b:a", "192k", out_path],
    capture_output=True, check=True
)
os.unlink(tmp_wav)

del pipe
gc.collect()
torch.cuda.empty_cache()

upd({"status": "done", "progress": 100, "file": out_path})
print(f"OK: {out_path}")
