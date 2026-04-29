"""
Generuje tekstury ostrych skał/kolców dla 4 światów gry Wisp.
Cel: rząd spiczastych skał jak zęby góry, wyraźne czubki skierowane ku górze.
"""
import gc, sys, time
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT = Path("/opt/gry/Wisp - NOWA wersja w budowie/public/assets/spikes")
OUT.mkdir(parents=True, exist_ok=True)

SPIKES = [
    ("w1_spikes",
     "side view of a row of sharp pointed grey stone spikes protruding upward, "
     "individual jagged rock teeth like mountain peaks, dark mossy granite, "
     "each spike clearly separated with sharp tip pointing up, "
     "dangerous stone stalagmites, fantasy game hazard, 2D hand painted",
     "flat, smooth, shelf, platform, no spikes, round, blurry, text, watermark, sky, background, person"),

    ("w2_spikes",
     "side view of a row of sharp pointed crystal spikes protruding upward, "
     "individual glowing purple crystal teeth, each spike clearly separated with sharp tip, "
     "magical gemstone stalagmites, vivid violet and pink, dangerous crystal hazard, "
     "2D hand painted game art, fantasy",
     "flat, smooth, shelf, round, blurry, text, watermark, sky, background, person, grey, brown"),

    ("w3_spikes",
     "side view of a row of sharp pointed dark stone spikes protruding upward, "
     "individual jagged brown rocky teeth like autumn mountain peaks, "
     "each spike clearly separated with sharp tip pointing up, "
     "rusty amber stone stalagmites, warm tones, 2D hand painted game art",
     "flat, smooth, shelf, round, blurry, text, watermark, sky, background, person, cold, blue"),

    ("w4_spikes",
     "side view of a row of sharp pointed icicle spikes protruding upward, "
     "individual frozen crystal teeth clearly separated, each icicle sharp tip pointing up, "
     "blue white ice stalagmites, frost crystal edges, inner glow, "
     "winter hazard, 2D hand painted game art, fantasy",
     "flat, smooth, shelf, round, blurry, text, watermark, sky, background, person, warm, green"),
]

BASE_STYLE = (
    ", isolated on pure black background, silhouette clearly visible, "
    "sharp pointy tips at top, wide horizontal row filling the frame, "
    "high contrast, crisp edges, game sprite asset, no ground texture below"
)

print("=" * 60)
print(f"Wisp — generowanie {len(SPIKES)} tekstur ostrych skał")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

t0 = time.time()
print("\n⏳ Ładowanie SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"✓ Model ({time.time()-t0:.1f}s)")

SEEDS = [1337, 2222, 4444]

for i, (name, prompt, neg) in enumerate(SPIKES):
    best = None

    for attempt, seed in enumerate(SEEDS):
        t1 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(seed + i * 1000)
        img = pipe(
            prompt=prompt + BASE_STYLE,
            negative_prompt=neg,
            width=768, height=512,
            num_inference_steps=40,
            guidance_scale=9.5,
            generator=gen,
        ).images[0]

        # Bierzemy TYLKO górną 1/3 (tam są czubki skał)
        w, h = img.size
        strip_h = h // 3          # 170px — same czubki
        spike_strip = img.crop((0, 0, w, strip_h))

        vname = f"{name}_v{attempt+1}"
        img.save(str(OUT / f"{vname}_full.png"))
        spike_strip.save(str(OUT / f"{vname}.png"))
        elapsed = time.time() - t1
        print(f"  [{i+1}/{len(SPIKES)}] {vname} seed={seed+i*1000} — {elapsed:.1f}s  →  {w}×{strip_h}px")
        sys.stdout.flush()

        if attempt == 0:
            best = spike_strip

    best.save(str(OUT / f"{name}.png"))

del pipe; gc.collect(); torch.cuda.empty_cache()
print(f"\n✓ Gotowe! {len(SPIKES)} × 3 wariantów w {(time.time()-t0)/60:.1f} min")
print(f"  → {OUT}")
