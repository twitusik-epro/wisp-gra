"""
Generuje 40 unikalnych teł dla gry Wisp (1 na poziom).
Model SDXL ładowany raz, trzymany w VRAM przez całe generowanie.
"""
import gc, json, sys, time
from pathlib import Path
from datetime import datetime

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

OUT_DIR = Path("/opt/gry/Wisp/asset-server/assets/pending")
META_FILE = Path("/opt/gry/Wisp/asset-server/assets/meta.json")
COPY_DIR = Path("/opt/gry/Wisp - NOWA wersja w budowie/Claude - tla dla swiatow")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 576, 1024
STEPS = 28

BASE_NEG = (
    "dark, scary, horror, ugly, violent, text, watermark, blurry, noisy, "
    "photorealistic, photograph, 3d render, deformed, human, person, face, "
    "low quality, grainy, oversaturated"
)

BASE_STYLE = (
    "2d game background art, hand painted illustration, vibrant colors, "
    "magical atmosphere, children game art, soft beautiful lighting, "
    "high quality, detailed, whimsical, fairytale"
)

LEVELS = [
    # ── Świat 1: Sunny Forest (1-10) ────────────────────────────────────────
    (1,  "w1", "enchanted forest at early dawn, first golden rays of sunlight piercing through ancient trees, soft morning mist, dewy grass, glowing green leaves"),
    (2,  "w1", "sunny forest path lined with giant glowing ferns, sunbeams creating light columns, butterflies, vibrant emerald green"),
    (3,  "w1", "magical forest clearing, ancient oak trees with luminous golden leaves, floating light orbs, wildflowers on the ground"),
    (4,  "w1", "bright forest with crystal clear stream, sunlight sparkling on water, mossy rocks, colorful birds, lush vegetation"),
    (5,  "w1", "sun-drenched forest meadow, tall magical trees, rainbow light through canopy, blooming flowers, cheerful warm atmosphere"),
    (6,  "w1", "ancient magical forest, trees with glowing runes, golden afternoon light, fairy mushrooms, floating sparkles"),
    (7,  "w1", "enchanted forest waterfall hidden among giant trees, mist catching sunlight, rainbow, vibrant tropical plants"),
    (8,  "w1", "forest of giant luminous mushrooms and ancient trees, warm golden glow, fireflies appearing, magical dusk"),
    (9,  "w1", "treetop canopy view from above, golden sunset light, birds flying, fluffy clouds below, magical sky bridge"),
    (10, "w1", "grand enchanted forest finale, ancient magical tree of life glowing, rainbow light, celebration, ultimate magical forest"),

    # ── Świat 2: Flower Meadow (11-20) ──────────────────────────────────────
    (11, "w2", "magical cherry blossom forest entrance, pink petals falling like snow, soft pink and white tones, spring magic"),
    (12, "w2", "giant colorful mushroom garden, oversized blue purple red mushrooms, glowing spots, whimsical fairy atmosphere"),
    (13, "w2", "rainbow flower meadow, flowers of every color, rainbow arching overhead, butterflies everywhere, joyful magical"),
    (14, "w2", "butterfly valley, thousands of colorful butterflies among flowers, golden light, dreamlike soft colors"),
    (15, "w2", "magical rose garden, giant roses red pink white, sparkles, petals floating in breeze, romantic fairy tale"),
    (16, "w2", "lavender field at sunset, purple lavender stretching to horizon, warm golden sky, magical fireflies"),
    (17, "w2", "sunflower meadow with magical twist, giant glowing sunflowers, bees, warm amber and gold, happy summer"),
    (18, "w2", "secret fairy garden, tiny glowing fairy lights among flowers, moonflowers, pastel magical night garden"),
    (19, "w2", "crystal flower garden, flowers made of crystals and gems, rainbow reflections, magical glittering light"),
    (20, "w2", "grand flower meadow finale, explosion of colorful flowers and light, rainbow, magical celebration, paradise garden"),

    # ── Świat 3: Golden Sunrise / Autumn (21-30) ────────────────────────────
    (21, "w3", "early autumn forest, first golden leaves appearing, warm morning light, misty and cozy, amber and green mix"),
    (22, "w3", "misty autumn morning forest, golden light through orange maple trees, fallen leaves on ground, magical fog"),
    (23, "w3", "warm amber autumn forest, rich orange and red maple trees, sunlight from behind, cinematic warm glow"),
    (24, "w3", "maple grove in full autumn glory, deep red orange canopy, golden light shafts, magical warm atmosphere"),
    (25, "w3", "autumn forest at golden hour sunset, dramatic orange sky, silhouetted trees, warm magical atmosphere"),
    (26, "w3", "river through autumn forest, golden reflections on water, colorful leaves floating, warm misty light"),
    (27, "w3", "ancient oak forest at dusk, golden last light, acorns, warm amber glow, magical ancient atmosphere"),
    (28, "w3", "evening forest with fireflies emerging, last golden light, firefly glow beginning, magical transition to night"),
    (29, "w3", "purple twilight magical forest, last colors of sunset, deep purple blue sky, silhouetted trees, magical stars appearing"),
    (30, "w3", "grand autumn finale, magical tree shedding golden leaves like stars, warm glowing light, breathtaking beauty"),

    # ── Świat 4: Winter Wonderland (31-40) ──────────────────────────────────
    (31, "w4", "first snow in enchanted forest, gentle snowflakes falling, soft white coating on pine trees, peaceful blue white"),
    (32, "w4", "snowy pine forest, heavy snow on branches, soft blue shadows, cozy winter atmosphere, snowflakes in air"),
    (33, "w4", "crystal ice cave entrance surrounded by snow, icicles gleaming, blue crystal light, magical winter cave"),
    (34, "w4", "frozen lake in winter forest, mirror-like ice surface, snow-covered trees reflecting, crisp blue white"),
    (35, "w4", "snowy mountain forest, dramatic peaks in background, pine trees covered in snow, crisp cold clear day"),
    (36, "w4", "magical ice crystal forest, trees made of ice crystals, rainbow refractions, sparkling magical winter"),
    (37, "w4", "aurora borealis over snowy forest, green purple aurora dancing in sky, snow sparkling below, magical night"),
    (38, "w4", "deep winter night forest, bright stars and moon, snow glittering, blue purple night, magical peaceful"),
    (39, "w4", "magical ice palace in winter forest, crystalline towers, snow swirling, magical blue white glow, fantasy"),
    (40, "w4", "grand winter finale, Wisp home magical bright cozy winter cottage in enchanted forest, warm light, snow, magical celebration, beautiful ending"),
]

def load_meta():
    if META_FILE.exists():
        return json.loads(META_FILE.read_text())
    return {}

def save_meta(meta):
    META_FILE.write_text(json.dumps(meta, indent=2))

print("=" * 60)
print(f"Wisp — generowanie 40 teł dla poziomów")
print(f"Urządzenie: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
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

meta = load_meta()
job_id = f"batch40_{datetime.now().strftime('%H%M%S')}"
meta[job_id] = {"job_id": job_id, "status": "generating", "files": [], "created_at": datetime.now().isoformat()}
save_meta(meta)

results = []
for i, (level, world, desc) in enumerate(LEVELS):
    t1 = time.time()
    fname = f"level{level:02d}_{world}.png"
    out_path = OUT_DIR / fname

    prompt = f"{desc}, {BASE_STYLE}"
    seed = 42 + level * 17

    gen = torch.Generator(device="cpu").manual_seed(seed)
    img = pipe(
        prompt=prompt,
        negative_prompt=BASE_NEG,
        width=W, height=H,
        num_inference_steps=STEPS,
        guidance_scale=7.5,
        generator=gen,
    ).images[0]
    img.save(str(out_path))

    elapsed = time.time() - t1
    total_elapsed = time.time() - t0
    remaining = (elapsed * (40 - i - 1))
    print(f"  [{i+1:2d}/40] Level {level:2d} ({world}) — {elapsed:.1f}s | pozostało ~{remaining/60:.1f} min")
    sys.stdout.flush()

    results.append(fname)

    # Kopiuj do katalogu użytkownika
    world_names = {"w1": "swiat1_sunny_forest", "w2": "swiat2_flower_meadow",
                   "w3": "swiat3_golden_sunrise", "w4": "swiat4_winter_wonder"}
    dest_dir = COPY_DIR / world_names[world]
    dest_dir.mkdir(parents=True, exist_ok=True)
    img.save(str(dest_dir / fname))

meta[job_id]["status"] = "pending"
meta[job_id]["files"] = results
meta[job_id]["finished_at"] = datetime.now().isoformat()
meta[job_id]["label"] = "all_40_levels"
meta[job_id]["prompt"] = "40 unique level backgrounds"
meta[job_id]["world"] = 0
meta[job_id]["layer"] = "full"
save_meta(meta)

# Zwolnij VRAM
del pipe
gc.collect()
torch.cuda.empty_cache()

total = time.time() - t0
print(f"\n✓ Gotowe! 40 grafik wygenerowano w {total/60:.1f} minut")
print(f"  Pliki: {OUT_DIR}")
print(f"  Kopia: {COPY_DIR}")
