"""
Regeneruje 3 odrzucone tła z surowymi promptami sezonowymi.
"""
import gc, sys, time
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline

OUT_DIR = Path("/opt/gry/Wisp/asset-server/assets/pending")
COPY_BASE = Path("/opt/gry/Wisp - NOWA wersja w budowie/Claude - tla dla swiatow")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 576, 1024
STEPS = 32

BASE_STYLE = (
    "2d game background art, hand painted illustration, vibrant colors, "
    "magical atmosphere, children game art, soft beautiful lighting, "
    "high quality, detailed, whimsical, fairytale"
)

REPLACEMENTS = [
    (
        20, "w2", "swiat2_flower_meadow",
        # Grand flower meadow finale — NOT a mountain collage
        "magical paradise flower meadow, grand finale, enormous explosion of colorful flowers everywhere, "
        "roses tulips daisies sunflowers all blooming together, rainbow arching over meadow, "
        "golden magical light, butterflies and flower petals floating, lush green grass, "
        "single beautiful scene panoramic meadow, joyful celebration",
        "mountain, mountains, rocks, cliffs, collage, panels, grid, split image, dark, scary, "
        "winter, snow, ice, bare trees, autumn leaves, orange red leaves, "
        "human, person, text, watermark, blurry, photorealistic"
    ),
    (
        34, "w4", "swiat4_winter_wonder",
        # Frozen lake — STRICTLY winter, bare trees only
        "frozen lake in deep winter enchanted forest, perfect mirror-like ice surface reflecting sky, "
        "snow-covered bare trees with no leaves, white birch trees in snow, "
        "crisp blue and white winter palette, snowflakes gently falling, "
        "icicles hanging from bare branches, magical cold blue light, peaceful winter scene",
        "autumn leaves, orange leaves, red leaves, yellow leaves, colorful foliage, "
        "green leaves, any leaves on trees, warm colors, orange, red, brown, "
        "flowers, spring, summer, dark, scary, human, person, text, watermark, blurry, photorealistic"
    ),
    (
        36, "w4", "swiat4_winter_wonder",
        # Ice crystal forest — STRICTLY winter, no flowering trees
        "magical ice crystal forest, bare trees completely encased in sparkling ice crystals, "
        "crystal clear ice formations on every branch, rainbow refractions in pure white snow, "
        "brilliant sparkling magical winter, deep blue and white and silver palette, "
        "no foliage anywhere only ice and snow and bare branches, ethereal winter magic",
        "flowers, pink flowers, red flowers, any flowers, blossoms, cherry blossom, "
        "autumn leaves, orange leaves, red leaves, yellow leaves, colorful foliage, green leaves, "
        "warm colors, orange, red, brown, pink, spring, summer, "
        "human, person, text, watermark, blurry, photorealistic"
    ),
]

print("=" * 60)
print("Wisp — regeneracja 3 odrzuconych teł")
print(f"GPU: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

print("\n⏳ Ładowanie SDXL...")
t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"✓ Model załadowany ({time.time()-t0:.1f}s)")

for i, (level, world, world_dir, desc, neg) in enumerate(REPLACEMENTS):
    t1 = time.time()
    fname = f"level{level:02d}_{world}.png"
    out_path = OUT_DIR / fname

    prompt = f"{desc}, {BASE_STYLE}"
    # Użyj innego seed niż oryginał (42 + level*17)
    seed = 1000 + level * 31

    print(f"\n[{i+1}/3] Generuję Level {level} ({world}) — seed={seed}")
    print(f"  Prompt: {desc[:80]}...")

    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=neg,
        width=W, height=H,
        num_inference_steps=STEPS,
        guidance_scale=8.0,
        generator=gen,
    ).images[0]
    img.save(str(out_path))

    dest_dir = COPY_BASE / world_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    img.save(str(dest_dir / fname))

    elapsed = time.time() - t1
    print(f"  ✓ Zapisano: {fname} ({elapsed:.1f}s)")
    sys.stdout.flush()

del pipe
gc.collect()
torch.cuda.empty_cache()

total = time.time() - t0
print(f"\n✓ Gotowe! 3 grafiki wygenerowano w {total/60:.1f} minut")
print(f"  Pliki: {OUT_DIR}")
print(f"  Kopia: {COPY_BASE}")
