"""
Generuje tekstury ostrych skał/kolców dla 4 światów gry Wisp — rysowanie kodem.
Efekt: ostre górzaste zęby z teksturą kamienia, jak w górach.
"""
import random, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

OUT = Path("/opt/gry/Wisp - NOWA wersja w budowie/public/assets/spikes")
OUT.mkdir(parents=True, exist_ok=True)

W, H = 768, 170

WORLDS = {
    'w1': dict(
        tip   = (185, 185, 185),  # jasny popielaty
        mid   = (110, 110, 112),  # średni szary
        base  = ( 50,  50,  52),  # ciemny szary
        crack = ( 28,  28,  30),
        glow  = None,
        bg    = (0, 0, 0),
        n_spikes = 24, seed = 101,
    ),
    'w2': dict(
        tip   = (185, 185, 185),
        mid   = (110, 110, 112),
        base  = ( 50,  50,  52),
        crack = ( 28,  28,  30),
        glow  = None,
        bg    = (0, 0, 0),
        n_spikes = 26, seed = 202,
    ),
    'w3': dict(
        tip   = (185, 185, 185),
        mid   = (110, 110, 112),
        base  = ( 50,  50,  52),
        crack = ( 28,  28,  30),
        glow  = None,
        bg    = (0, 0, 0),
        n_spikes = 22, seed = 303,
    ),
    'w4': dict(
        tip   = (185, 185, 185),
        mid   = (110, 110, 112),
        base  = ( 50,  50,  52),
        crack = ( 28,  28,  30),
        glow  = None,
        bg    = (0, 0, 0),
        n_spikes = 28, seed = 404,
    ),
}


def lerp3(a, b, t):
    t = float(np.clip(t, 0, 1))
    return (
        int(a[0] + (b[0]-a[0])*t),
        int(a[1] + (b[1]-a[1])*t),
        int(a[2] + (b[2]-a[2])*t),
    )


def make_spike_image(world, cfg):
    rng = random.Random(cfg['seed'])
    np.random.seed(cfg['seed'])

    img = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    pixels = np.zeros((H, W, 4), dtype=np.uint8)

    # ── 1. Profil czubków skał ───────────────────────────────────────
    profile = np.full(W, float(H))   # y-coord góry skały (0=top screen)

    n = cfg['n_spikes']
    # Rozkładamy szpice dość gęsto — trochę się nakładają żeby nie było luk
    xs = sorted(rng.sample(range(10, W - 10), n))

    spike_data = []
    for sx in xs:
        # Zmienna wysokość szpiców — najwyższe ok 90% H, najniższe 35% H
        sh = rng.uniform(0.35, 0.92) * H
        # Szerokość podstawy — wąskie szpice jak stalaktyty
        bw = rng.randint(10, 32)
        spike_data.append((sx, sh, bw))

    # Wektorowo oblicz profil
    X_idx = np.arange(W, dtype=float)
    for sx, sh, bw in spike_data:
        dist = np.abs(X_idx - sx) / bw          # 0 w centrum, 1 na krawędzi
        # Potęga 1.3 = ostry czubek, szerokie boki
        t_shape = np.clip(dist, 0, 1) ** 1.3
        spike_top = H - sh * (1.0 - t_shape)
        spike_top = np.clip(spike_top, 0, H)
        profile = np.minimum(profile, spike_top)

    # Lekkie wygładzenie żeby krawędź nie była schodkowa
    from scipy.ndimage import uniform_filter1d
    profile = uniform_filter1d(profile, size=3)

    # ── 2. Wypełnij kolor skały ──────────────────────────────────────
    Y_idx, X_idx2 = np.mgrid[0:H, 0:W]
    prof2d    = profile[np.newaxis, :].repeat(H, axis=0)
    rock_mask = Y_idx >= prof2d                          # True = wnętrze skały
    spike_h2d = (H - prof2d).clip(min=1)
    t_val     = ((Y_idx - prof2d) / spike_h2d).clip(0, 1)  # 0=czubek 1=podstawa

    # Perlin-like noise — losowy szum dla tekstury kamienia
    noise_raw = np.random.randint(-18, 18, size=(H, W))

    # Kolory przez interpolację t
    tip, mid, base_ = np.array(cfg['tip']), np.array(cfg['mid']), np.array(cfg['base'])
    # t<0.25 → tip→mid, t>=0.25 → mid→base
    t1 = (t_val / 0.25).clip(0, 1)
    t2 = ((t_val - 0.25) / 0.75).clip(0, 1)
    color_tm = tip[np.newaxis, np.newaxis, :] + (mid - tip)[np.newaxis, np.newaxis, :] * t1[:, :, np.newaxis]
    color_mb = mid[np.newaxis, np.newaxis, :] + (base_ - mid)[np.newaxis, np.newaxis, :] * t2[:, :, np.newaxis]
    use_mb = (t_val >= 0.25)[:, :, np.newaxis]
    rock_color = np.where(use_mb, color_mb, color_tm).astype(float)

    # Dodaj szum tekstury kamienia
    rock_color += noise_raw[:, :, np.newaxis]
    rock_color = rock_color.clip(0, 255).astype(np.uint8)

    # Glow na krawędzi czubka (jasność wzdłuż profilu)
    edge_dist = (Y_idx - prof2d).clip(0)
    edge_glow = np.exp(-edge_dist / 4.0)   # silny blask przy czubku
    glow_col  = np.array(cfg['glow'] if cfg['glow'] else cfg['tip'], dtype=float)
    rock_color_f = rock_color.astype(float)
    rock_color_f += edge_glow[:, :, np.newaxis] * (glow_col - rock_color_f) * 0.55
    rock_color_f = rock_color_f.clip(0, 255).astype(np.uint8)

    # ── 3. Pionowe pęknięcia (tekstura kamienia) ────────────────────
    crack_col = np.array(cfg['crack'])
    rng2 = random.Random(cfg['seed'] + 77)
    n_cracks = rng2.randint(18, 30)
    for _ in range(n_cracks):
        cx  = rng2.randint(0, W - 1)
        # Pęknięcie zaczyna się w okolicach czubka skały w tej kolumnie
        cy0 = int(profile[cx]) + rng2.randint(1, 5)
        cy1 = min(H, cy0 + rng2.randint(8, 35))
        width_px = rng2.randint(1, 2)
        for cy in range(cy0, cy1):
            # Lekko wężykujące — przesunięcie X o 0 lub ±1
            ox = rng2.randint(-1, 1)
            rx = max(0, min(W-1, cx + ox))
            if rock_mask[cy, rx]:
                rock_color_f[cy, rx] = crack_col.astype(float)

    # ── 4. Złóż obraz ───────────────────────────────────────────────
    pixels[:, :, :3] = rock_color_f.clip(0, 255).astype(np.uint8)
    pixels[:, :, 3]  = np.where(rock_mask, 255, 0)   # przezroczyste tło

    result = Image.fromarray(pixels, 'RGBA')

    # Lekkie zaostrzenie konturów
    result = result.filter(ImageFilter.SHARPEN)
    return result


for world, cfg in WORLDS.items():
    img = make_spike_image(world, cfg)
    out_path = OUT / f"{world}_spikes.png"
    img.save(str(out_path))
    print(f"✓ {world}_spikes.png  ({W}×{H}px)")

print(f"\nGotowe → {OUT}")
