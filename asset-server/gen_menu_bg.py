"""
Generuje tło ekranu głównego menu gry Wisp.
Format: 390×844 PNG (portrait mobile), bajkowy zmierzch leśny, dzieci.
"""
import gc, sys, time
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT = Path("/opt/gry/Wisp - NOWA wersja w budowie/public/assets/ui")
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
  ("menu_bg_v1",
   "magical enchanted forest at golden twilight dusk, warm golden pink purple sky visible through tall ancient tree silhouettes, "
   "dozens of glowing green yellow fireflies floating upward, soft volumetric light rays through forest canopy, "
   "first stars appearing in upper sky, lush glowing mushrooms on forest floor, sparkling magical dust particles, "
   "warm dreamy cozy atmosphere, children fairy tale illustration, vibrant colors, NOT dark, bright and colorful, "
   "vertical portrait orientation, no text, no characters, no people",
   "dark scary night, text, watermark, person, human, ugly, blurry, black, horror, gloomy"),

  ("menu_bg_v2",
   "beautiful fairy tale forest clearing at sunset dusk, sky gradient from warm golden orange to soft violet purple, "
   "glowing fireflies everywhere, ancient magical trees with bioluminescent moss, colorful wildflowers glowing softly, "
   "whimsical sparkling light, cozy warm magical glow from below, dreamy pastel colors, "
   "children game background art, bright and cheerful NOT dark, portrait vertical, no text, no characters",
   "dark night, scary, text, watermark, person, human, ugly, blurry, horror, black sky"),

  ("menu_bg_v3",
   "enchanted forest twilight scene, warm amber golden hour light, magical glowing will-o-wisps floating, "
   "tall tree trunks with warm light from behind, colorful glowing plants and flowers, "
   "soft pink purple blue gradient sky, sparkling magical particles, fairy tale storybook illustration, "
   "bright warm cheerful colors, children's book style, portrait format, no text, no characters",
   "full dark night, horror, scary, text, watermark, person, blurry, ugly, gloomy"),
]

print("=" * 60)
print(f"Wisp — tło menu ({len(PROMPTS)} warianty)")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("=" * 60)

t0 = time.time()
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"✓ Model ({time.time()-t0:.1f}s)\n")

SEEDS = [42, 777, 1234]

for name, prompt, neg in PROMPTS:
    seed = SEEDS[int(name[-1])-1]
    t1 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=neg,
        width=640, height=896,
        num_inference_steps=40,
        guidance_scale=9.0,
        generator=gen,
    ).images[0]
    img.save(str(OUT / f"{name}.png"))
    print(f"  {name} seed={seed} — {time.time()-t1:.1f}s")
    sys.stdout.flush()

del pipe; gc.collect(); torch.cuda.empty_cache()
print(f"\n✓ Gotowe w {(time.time()-t0)/60:.1f} min → {OUT}")
