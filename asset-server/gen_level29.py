import torch, time, gc
from pathlib import Path
from diffusers import StableDiffusionXLPipeline

OUT = Path("/opt/gry/Wisp/asset-server/assets/pending")
COPY = Path("/opt/gry/Wisp - NOWA wersja w budowie/Claude - tla dla swiatow/swiat3_golden_sunrise")
COPY.mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("Ladowanie SDXL...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16",
).to("cuda")
pipe.set_progress_bar_config(disable=True)
print(f"Model gotowy ({time.time()-t0:.1f}s)")

prompt = (
    "purple twilight magical forest, last colors of warm sunset fading to deep night, "
    "rich purple violet and indigo sky glowing at horizon, silhouetted tree trunks against sky, "
    "first stars twinkling in purple sky, fireflies beginning to glow gold, "
    "warm amber horizon blending into cool purple night, magical dreamy twilight, "
    "2d game background art, hand painted illustration, vibrant colors, "
    "magical atmosphere, children game art, soft beautiful lighting, "
    "high quality, detailed, whimsical, fairytale"
)
neg = (
    "dark, scary, horror, ugly, violent, text, watermark, blurry, noisy, "
    "photorealistic, photograph, 3d render, deformed, human, person, face, "
    "low quality, grainy, oversaturated, winter, snow, ice, monochrome"
)

gen = torch.Generator(device="cpu").manual_seed(1999)
img = pipe(prompt=prompt, negative_prompt=neg,
           width=576, height=1024, num_inference_steps=32,
           guidance_scale=8.0, generator=gen).images[0]

fname = "level29_w3.png"
img.save(str(OUT / fname))
img.save(str(COPY / fname))
print(f"Zapisano: {fname} ({time.time()-t0:.1f}s)")

del pipe; gc.collect(); torch.cuda.empty_cache()
