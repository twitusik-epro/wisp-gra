"""
Generuje tekstury platform dla 4 światów gry Wisp.
Format: 768×128 PNG (tileable poziomo)
"""
import gc, sys, time
from pathlib import Path
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT = Path("/opt/gry/Wisp - NOWA wersja w budowie/public/assets/platforms")
OUT.mkdir(parents=True, exist_ok=True)

PLATFORMS = [
    ("w1_log",
     "seamless tileable game platform texture, ancient mossy fallen log cross-section, "
     "thick bark with green soft moss, small mushrooms on sides, glowing magical spores, "
     "warm golden sunlight on wood, top surface flat and walkable, "
     "lush forest fantasy game art, hand painted 2d, side view horizontal",
     "dark, scary, text, watermark, human, person, vertical, portrait, ugly, blurry"),

    ("w1_stone",
     "seamless tileable game platform, mossy enchanted forest stone slab, "
     "flat top covered in bright green moss and tiny flowers, glowing runes on sides, "
     "ancient magical rock, warm dappled sunlight, fantasy children game art, "
     "hand painted 2d illustration, side view horizontal platform",
     "dark, scary, text, watermark, human, person, blurry, ugly"),

    ("w2_mushroom",
     "seamless tileable game platform, giant magical mushroom cap top, "
     "flat wide red and white spotted mushroom top surface, glowing spots, "
     "soft sparkles, whimsical fairy tale, vivid colors, "
     "children fantasy game art, hand painted 2d, side view horizontal",
     "dark, scary, text, watermark, human, person, blurry, ugly, brown"),

    ("w2_flower",
     "seamless tileable game platform, thick magical vine and flower bridge, "
     "lush colorful flowers on top pink yellow purple, green leaves and vines on sides, "
     "sparkling fairy dust, rainbow pastel colors, whimsical fantasy, "
     "children game art, hand painted 2d, horizontal side view",
     "dark, scary, text, watermark, human, person, blurry, ugly"),

    ("w3_wood",
     "seamless tileable game platform, old wooden log covered in autumn leaves, "
     "warm orange red yellow maple leaves piled on top, rich brown bark, "
     "golden amber glow, cozy autumn forest, magical warm light, "
     "children fantasy game art, hand painted 2d, side view horizontal",
     "dark, scary, text, watermark, human, person, blurry, ugly, cold, winter"),

    ("w3_rock",
     "seamless tileable game platform, autumn stone ledge with fallen leaves, "
     "flat mossy rock top covered in orange leaves, warm golden sunlight, "
     "amber glow, cozy autumn atmosphere, magical forest, "
     "children game art, hand painted 2d, side view horizontal",
     "dark, scary, text, watermark, human, person, blurry, ugly, cold"),

    ("w4_ice",
     "seamless tileable game platform, magical thick ice crystal shelf, "
     "flat smooth frozen top surface, blue white transparent ice, "
     "frost crystal formations on edges, inner blue glow, snowflakes, "
     "winter wonderland fantasy, children game art, hand painted 2d, side view horizontal",
     "dark, scary, text, watermark, human, person, blurry, ugly, warm, orange, autumn"),

    ("w4_snow",
     "seamless tileable game platform, thick snow and ice platform, "
     "fluffy white snow piled on top, icicles hanging below, "
     "soft blue magical glow, sparkling ice crystals, peaceful winter, "
     "children fantasy game art, hand painted 2d, side view horizontal",
     "dark, scary, text, watermark, human, person, blurry, ugly, warm, green, flowers"),
]

BASE_STYLE = (
    ", game asset, isolated on dark background, wide horizontal platform, "
    "no background clutter, clear edges, vibrant, beautiful, high quality"
)

print("=" * 60)
print(f"Wisp — generowanie {len(PLATFORMS)} tekstur platform")
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

for i, (name, prompt, neg) in enumerate(PLATFORMS):
    t1 = time.time()
    seed = 7777 + i * 313

    # Generuj 768×512 (dobra jakość), potem przytnij do paska platformy
    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(
        prompt=prompt + BASE_STYLE,
        negative_prompt=neg,
        width=768, height=512,
        num_inference_steps=35,
        guidance_scale=8.5,
        generator=gen,
    ).images[0]

    # Przytnij środkowy pas (platforma to górna 1/3 obrazka)
    w, h = img.size
    strip_h = h // 3
    platform = img.crop((0, 0, w, strip_h))

    # Zapisz pełny obraz + wycięty pasek
    img.save(str(OUT / f"{name}_full.png"))
    platform.save(str(OUT / f"{name}.png"))

    elapsed = time.time() - t1
    print(f"  [{i+1}/{len(PLATFORMS)}] {name} — {elapsed:.1f}s  →  {w}×{strip_h}px")
    sys.stdout.flush()

del pipe; gc.collect(); torch.cuda.empty_cache()
print(f"\n✓ Gotowe! {len(PLATFORMS)} tekstur w {(time.time()-t0)/60:.1f} min")
print(f"  → {OUT}")
