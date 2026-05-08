"""
ACE-Step music generation (replaces MusicGen-small).
Args: job_id prompt duration_sec infer_steps guidance_scale cfg_type seed audio_path audio_strength out_path status_path
"""
import sys, json, os, subprocess, gc, threading, time, math
sys.path.insert(0, "/opt/ACE-Step")

job_id          = sys.argv[1]
prompt          = sys.argv[2]
duration        = float(sys.argv[3])
infer_steps     = int(sys.argv[4]) if len(sys.argv) > 4 else 30
guidance_scale  = float(sys.argv[5]) if len(sys.argv) > 5 else 7.0
cfg_type        = sys.argv[6] if len(sys.argv) > 6 else "apg"
seed            = sys.argv[7] if len(sys.argv) > 7 else "-1"
audio_path      = sys.argv[8] if len(sys.argv) > 8 else ""
audio_strength  = float(sys.argv[9]) if len(sys.argv) > 9 else 0.5
out_path        = sys.argv[10]
status_f        = sys.argv[11]

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

# ── Extension path: pure ffmpeg, skip AI model entirely ──────────────────────
use_ref = bool(audio_path and os.path.exists(audio_path))

if use_ref:
    upd({"status": "generating", "progress": 20})

    _r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True)
    try:
        ref_duration = float(_r.stdout.strip())
    except Exception:
        ref_duration = duration

    upd({"status": "generating", "progress": 35})

    # Build a TRUE seamless loop from the reference:
    #
    #   body  = original[cf : ref_dur-cf]          (unchanged middle section)
    #   tail  = original[ref_dur-cf : end]          (last `cf` seconds)
    #   head  = original[0 : cf]                    (first `cf` seconds)
    #   trans = acrossfade(tail, head, d=cf)        (tail fades into head)
    #   loop  = body ++ trans                        (total = ref_dur - cf)
    #
    # When loop is concatenated with itself the boundary is:
    #   ...trans ends at original[cf]  →  body starts at original[cf]  ✓ seamless!
    #
    # `cf` is at most 1/3 of track length so body is always positive.

    # cf_trim  = how much to bite off each end for the transition region
    # cf_fade  = actual acrossfade d (90% of cf_trim; safety margin against
    #            MP3 container/frame rounding making tail slightly < cf_trim)
    # NOTE: asplit + acrossfade + concat in one filter_complex deadlocks ffmpeg;
    #       use 3 separate -i copies of the same file to avoid it.
    cf_trim = min(2.0, ref_duration / 3.0)
    cf_fade = round(cf_trim * 0.9, 6)
    body_start = cf_trim
    body_end   = ref_duration - cf_trim

    seamless_wav = out_path.replace(".mp3", "_seamless.wav")

    flt = (
        f"[0:a]atrim=start={body_start}:end={body_end},asetpts=PTS-STARTPTS[body];"
        f"[1:a]atrim=start={body_end},asetpts=PTS-STARTPTS[tail];"
        f"[2:a]atrim=end={cf_trim},asetpts=PTS-STARTPTS[head];"
        f"[tail][head]acrossfade=d={cf_fade}:c1=tri:c2=tri[trans];"
        f"[body][trans]concat=n=2:v=0:a=1[out]"
    )
    subprocess.run(
        ["ffmpeg", "-y",
         "-i", audio_path, "-i", audio_path, "-i", audio_path,
         "-filter_complex", flt, "-map", "[out]",
         "-ar", "44100", "-ac", "2", seamless_wav],
        capture_output=True, check=True)

    upd({"status": "generating", "progress": 60})

    _r2 = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", seamless_wav],
        capture_output=True, text=True)
    loop_dur = float(_r2.stdout.strip())

    # Concat enough copies; no crossfade needed — the loop IS already seamless.
    n = math.ceil(duration / loop_dur) + 1
    n = min(n, 40)

    cmd = ["ffmpeg", "-y"]
    for _ in range(n):
        cmd += ["-i", seamless_wav]

    concat_flt = "".join(f"[{i}:a]" for i in range(n)) + f"concat=n={n}:v=0:a=1[out]"
    cmd += [
        "-filter_complex", concat_flt,
        "-map", "[out]",
        "-t", str(duration),
        "-ar", "44100", "-ac", "2",
        "-c:a", "libmp3lame", "-b:a", "192k",
        out_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    os.unlink(seamless_wav)

    upd({"status": "done", "progress": 100, "file": out_path})
    print(f"OK: {out_path}")
    sys.exit(0)

# ── Normal text2music path ────────────────────────────────────────────────────
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
    guidance_scale=guidance_scale,
    scheduler_type="euler",
    cfg_type=cfg_type,
    omega_scale=10.0,
    manual_seeds=seed,
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
