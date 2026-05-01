"""
Generuje muzykę dla gry Wisp używając MusicGen (Meta AudioCraft).
Uruchamiany przez server.py jako subprocess w środowisku wisp-music.
Argumenty: job_id prompt duration_sec output_path
"""
import sys, json, time, warnings, threading
warnings.filterwarnings("ignore")

job_id   = sys.argv[1]
prompt   = sys.argv[2]
duration = float(sys.argv[3])
top_k    = int(sys.argv[4])
out_path = sys.argv[5]
status_f = sys.argv[6]  # plik statusu JSON

def update_status(s):
    with open(status_f, 'w') as f:
        json.dump(s, f)

def progress_ticker(status_label, pct_start, pct_end, interval, stop_event):
    """Płynnie przesuwa pasek od pct_start do pct_end co interval sekund."""
    pct = pct_start
    step = max(1, (pct_end - pct_start) // 20)
    while not stop_event.is_set() and pct < pct_end:
        time.sleep(interval)
        if stop_event.is_set():
            break
        pct = min(pct_end, pct + step)
        update_status({"status": status_label, "progress": pct})

update_status({"status": "loading", "progress": 5})

import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

update_status({"status": "loading", "progress": 15})

# Wątek postępu podczas ładowania modelu (15% → 38%, co 2s)
_stop_load = threading.Event()
_t_load = threading.Thread(target=progress_ticker,
    args=("loading", 15, 38, 2, _stop_load), daemon=True)
_t_load.start()

model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=duration, top_k=top_k)

_stop_load.set()

update_status({"status": "generating", "progress": 40})

# Wątek postępu podczas generacji audio (40% → 85%, co 1.5s)
_stop_gen = threading.Event()
_t_gen = threading.Thread(target=progress_ticker,
    args=("generating", 40, 85, 1.5, _stop_gen), daemon=True)
_t_gen.start()

wav = model.generate([prompt])

_stop_gen.set()

update_status({"status": "saving", "progress": 90})

# Zapisz jako WAV (audiocraft), potem skonwertuj na MP3 przez ffmpeg
import subprocess, os
tmp = out_path.replace('.mp3', '_tmp')
audio_write(tmp, wav[0].cpu(), model.sample_rate, strategy="loudness")
tmp_wav = tmp + '.wav'
subprocess.run(['ffmpeg', '-y', '-i', tmp_wav, '-b:a', '192k', out_path],
               capture_output=True)
os.unlink(tmp_wav)

update_status({"status": "done", "progress": 100, "file": out_path})
print(f"OK: {out_path}")
