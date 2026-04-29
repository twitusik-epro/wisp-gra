"""
Generuje 4 tła ekranu "Poziom ukończony" dla gry Wisp — po jednym na świat.
Format: 768×320 PNG, bajkowy styl, dzieci, bez tekstu.
"""
import gc, sys, time
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT = Path("/opt/gry/Wisp - NOWA wersja w budowie/public/assets/ui")
OUT.mkdir(parents=True, exist_ok=True)

IMGS = [
    ("win_w1",
     "beautiful magical enchanted forest clearing, celebration scene, "
     "golden sunbeams through tall ancient trees, glowing fireflies, "
     "colorful butterflies, sparkling magical dust, lush green ferns, "
     "soft warm light, fairy tale children illustration, wide horizontal banner, "
     "no text, no characters, no people",
     "dark, scary, text, letters, watermark, person, human, ugly, blurry"),

    ("win_w2",
     "magical fairy tale mushroom garden, giant glowing colorful mushrooms, "
     "vibrant flowers in bloom, sparkling fairy dust, butterflies, "
     "rainbow pastel colors, magical meadow celebration, wide horizontal banner, "
     "children illustration style, no text, no characters",
     "dark, scary, text, letters, watermark, person, human, ugly, blurry"),

    ("win_w3",
     "beautiful golden autumn forest celebration, warm orange amber sunlight, "
     "falling maple leaves in the air, glowing mushrooms, cozy magical atmosphere, "
     "rich warm colors, sparkling light through trees, wide horizontal banner, "
     "children illustration style, no text, no characters",
     "cold, winter, snow, dark, scary, text, watermark, person, human, blurry"),

    ("win_w4",
     "magical sparkling winter wonderland, beautiful ice crystal formations, "
     "soft blue glowing snow, snowflakes floating, aurora borealis colors, "
     "peaceful enchanted ice forest, wide horizontal banner, "
     "children illustration style, no text, no characters",
     "dark, scary, text, letters, watermark, person, human, ugly, blurry, warm, autumn"),
]

BASE = ", game UI background art, celebration, wide cinematic, vibrant, high quality, 2D illustration"

print("=" * 60)
print(f"Wisp — ekrany 'Poziom ukończony' ({len(IMGS)} obrazki)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"✓ Model ({time.time()-t0:.1f}s)\n")

SEEDS = [3141, 2718, 1618]

for name, prompt, neg in IMGS:
    for attempt, seed in enumerate(SEEDS):
        t1 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        img = pipe(
            prompt=prompt + BASE,
            negative_prompt=neg,
            width=768, height=320,
            num_inference_steps=38,
            guidance_scale=8.5,
            generator=gen,
        ).images[0]
        vname = f"{name}_v{attempt+1}"
        img.save(str(OUT / f"{vname}.png"))
        print(f"  {vname} seed={seed} — {time.time()-t1:.1f}s")
        sys.stdout.flush()
    # domyślny = v1
    img_def = Image.open(str(OUT / f"{name}_v1.png"))
    img_def.save(str(OUT / f"{name}.png"))

del pipe; gc.collect(); torch.cuda.empty_cache()
print(f"\n✓ Gotowe w {(time.time()-t0)/60:.1f} min  →  {OUT}")
