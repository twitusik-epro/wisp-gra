import torch, time, gc, sys
from pathlib import Path
from diffusers import StableDiffusionXLPipeline

OUT = Path("/opt/gry/Wisp/asset-server/assets/pending")
COPY = Path("/opt/gry/Wisp - NOWA wersja w budowie/Claude - tla dla swiatow/swiat4_winter_wonder")
COPY.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("Ladowanie SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"Model gotowy ({time.time()-t0:.1f}s)")

variants = [
    (4242, "enchanted fairy tale ice crystal forest, trees made of glowing magical ice crystals, "
           "soft rainbow light refractions sparkling everywhere, gentle magical snowflakes floating, "
           "pastel blue and white and silver with soft pink and violet crystal glow, "
           "whimsical fairy lights inside ice crystals, magical warm glow in cold winter, "
           "dreamlike beautiful fantasy, cozy magical winter wonderland"),
    (5555, "magical crystal winter forest, giant sparkling ice crystal trees glowing with inner light, "
           "soft blue purple and silver palette, tiny magical fairies and glowing orbs among crystals, "
           "rainbow prisms dancing on snow, fairy tale enchanted winter, soft warm magical light, "
           "beautiful gentle winter scene, children fairy tale illustration"),
    (6101, "whimsical ice forest, translucent crystal trees glowing pink blue and silver, "
           "magical sparkles and glowing snowflakes, pastel winter colors, "
           "soft ethereal light, charming fairy tale mood, cozy and magical not scary, "
           "enchanted winter paradise, warm magical atmosphere despite snow and ice"),
]

neg = (
    "dark, scary, horror, ugly, gloomy, harsh, cold atmosphere, "
    "human, person, text, watermark, blurry, photorealistic, photograph, 3d render, "
    "autumn leaves, orange, brown, warm earth tones, flowers, spring"
)
style = (", 2d game background art, hand painted illustration, vibrant colors, "
         "magical atmosphere, children game art, soft beautiful lighting, "
         "high quality, detailed, whimsical, fairytale")

for seed, desc in variants:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(prompt=desc + style, negative_prompt=neg,
               width=576, height=1024, num_inference_steps=35,
               guidance_scale=7.5, generator=gen).images[0]
    fname = f"level36_w4_fairy{seed}.png"
    img.save(str(OUT / fname))
    img.save(str(COPY / fname))
    print(f"Zapisano: {fname}")
    sys.stdout.flush()

del pipe; gc.collect(); torch.cuda.empty_cache()
print(f"Gotowe ({time.time()-t0:.1f}s)")
