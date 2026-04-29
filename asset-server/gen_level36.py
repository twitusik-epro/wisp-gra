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
    (2501, "magical winter forest, every bare tree branch completely covered in thick white ice crystals and frost, "
           "deep blue winter sky, fresh white snow on ground, sparkling diamond-like ice refractions, "
           "cold crisp pure winter, no leaves no flowers no foliage, only ice and snow on bare branches"),
    (3777, "enchanted ice crystal forest in deep winter, bare skeletal trees encrusted in sparkling ice formations, "
           "blue white silver color palette only, snowflakes swirling, winter moonlight on pristine snow, "
           "ethereal frozen wonderland, every branch bare and coated in crystalline ice"),
]

neg = (
    "flowers, blossoms, petals, cherry blossom, pink flowers, red flowers, any flowers, "
    "green leaves, autumn leaves, orange leaves, red leaves, yellow leaves, any leaves, foliage, "
    "warm colors, orange, red, brown, pink, magenta, spring, summer, autumn, "
    "dark, scary, human, person, text, watermark, blurry, photorealistic"
)
style = (", 2d game background art, hand painted illustration, magical atmosphere, "
         "children game art, high quality, detailed, whimsical, fairytale")

for seed, desc in variants:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(prompt=desc + style, negative_prompt=neg,
               width=576, height=1024, num_inference_steps=35,
               guidance_scale=9.0, generator=gen).images[0]
    fname = f"level36_w4_v{seed}.png"
    img.save(str(OUT / fname))
    img.save(str(COPY / fname))
    print(f"Zapisano: {fname}")
    sys.stdout.flush()

del pipe; gc.collect(); torch.cuda.empty_cache()
print(f"Gotowe ({time.time()-t0:.1f}s)")
