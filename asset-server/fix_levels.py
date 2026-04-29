"""
Analizuje i naprawia dane poziomów gry Wisp:
1. Kolce na ruchomych platformach (lewitują gdy platforma odpływa)
2. Kolce bez żadnej platformy pod spodem
3. Kolec przy wyjściu na poziomie 1 (diament niedostępny)
"""
import re, json, copy

GAME_HTML = "/opt/gry/Wisp - NOWA wersja w budowie/public/game.html"

# ── Parsowanie danych poziomów z game.html ───────────────────────────────────

with open(GAME_HTML, encoding='utf-8') as f:
    src = f.read()

# Znajdź blok configs = [ ... ];
m = re.search(r'const configs = \[(.*?)\];\s*\n\s*const cfg', src, re.DOTALL)
raw = m.group(1)

# Parsuj każdy poziom: szukaj kolejnych obiektów {}
levels_raw = re.findall(r'\{plat:\[(.*?)\]\s*,\s*spk:\[(.*?)\].*?gem:\[(.*?)\].*?goal:\[(.*?)\]', raw, re.DOTALL)

def parse_arr(s):
    """Parsuj ciąg tablic [[a,b,c],[d,e,f],...] → list of lists"""
    s = re.sub(r'//.*', '', s)  # usuń komentarze
    items = re.findall(r'\[([^\[\]]+)\]', s)
    result = []
    for it in items:
        vals = []
        for x in it.split(','):
            x = x.strip().strip("'\"")
            if not x: continue
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(x)  # 'g' lub inne stringi
        result.append(vals)
    return result

def is_moving(plat):
    return len(plat) >= 5 and plat[4] == 1

def platform_x_range(plat):
    """Zwraca (xmin, xmax) uwzględniając ruch poziomy"""
    x, y, w = plat[0], plat[1], plat[2]
    if is_moving(plat):
        mx = plat[5] if len(plat) > 5 else 0
        return (x - mx, x + w + mx)
    return (x, x + w)

def spike_x_range(spk):
    return (spk[0], spk[0] + spk[2])

def platforms_below_spike(spk, platforms, tolerance=3):
    """
    Zwraca listę (platforma, czy_statyczna) które mają wierzch 16px pod spodem kolca.
    """
    sy = spk[1]
    sx0, sx1 = spike_x_range(spk)
    results = []
    for p in platforms:
        py = p[1]
        if abs(py - (sy + 16)) > tolerance:
            continue
        px0, px1 = p[0], p[0] + p[2]
        # sprawdź nakładanie x
        overlap = min(sx1, px1) - max(sx0, px0)
        if overlap >= 4:
            results.append((p, not is_moving(p), overlap))
    return results

def find_static_platform_for_height(target_y, platforms, min_width=30):
    """Znajdź statyczną platformę przy danym y=target_y (wierzch)"""
    candidates = []
    for p in platforms:
        if is_moving(p): continue
        if abs(p[1] - target_y) > 3: continue
        if p[2] >= min_width:
            candidates.append(p)
    return candidates

def gems_near(spk, gems, margin=30):
    """Zwraca gemy w zasięgu kolca"""
    sx0, sx1 = spike_x_range(spk)
    sy = spk[1]
    cx = (sx0 + sx1) / 2
    near = []
    for g in gems:
        gx, gy = g[0], g[1]
        if abs(gx - cx) < spk[2]/2 + margin and abs(gy - sy) < 50:
            near.append(g)
    return near

# ── Parsowanie poziomów ────────────────────────────────────────────────────
levels = []
# Ręcznie parsujemy bo regex jest złożony; wyciągamy dane inaczej
# Szukamy configs[lvl] bezpośrednio

# Znajdź wszystkie {plat:[...], spk:[...], gem:[...], goal:[...], ...}
level_blocks = re.findall(
    r'\{plat:\[([\s\S]*?)\]\s*,\s*\n?\s*spk:\[([\s\S]*?)\]\s*,\s*\n?\s*gem:\[([\s\S]*?)\]\s*,\s*\n?\s*goal:\[([^\]]+)\]',
    raw
)

for i, (plat_s, spk_s, gem_s, goal_s) in enumerate(level_blocks):
    plats = parse_arr(plat_s)
    spks  = parse_arr(spk_s)
    gems  = parse_arr(gem_s)
    goal  = [float(x.strip()) for x in goal_s.split(',')]
    levels.append({'plat': plats, 'spk': spks, 'gem': gems, 'goal': goal})

print(f"Zparsowano {len(levels)} poziomów\n")

# ── Analiza i naprawa ───────────────────────────────────────────────────────
FIXES = {}

for lvl_i, lvl in enumerate(levels):
    lvl_n = lvl_i + 1
    plats = lvl['plat']
    spks  = lvl['spk']
    gems  = lvl['gem']
    goal  = lvl['goal']

    new_spks = []
    changes = []

    for spk in spks:
        sx0, sx1 = spike_x_range(spk)
        sx_c = (sx0 + sx1) / 2
        sy = spk[1]
        sw = spk[2]

        below = platforms_below_spike(spk, plats)

        if not below:
            # BRAK platformy pod spodem — kolec lewituje
            changes.append(f"  ⚠ spk {spk} → BEZ PLATFORMY, usuwam")
            # Nie dodajemy go do new_spks
            continue

        # Sprawdź czy jest przynajmniej 1 statyczna platforma
        static_below = [(p, ov) for (p, is_st, ov) in below if is_st]
        moving_below = [(p, ov) for (p, is_st, ov) in below if not is_st]

        if static_below:
            # OK — jest na statycznej platformie
            new_spks.append(spk)
            continue

        # Tylko ruchome platformy → kolec lewituje gdy platforma odpływa
        # Szukamy statycznej platformy w pobliżu i przenosimy
        # Sprawdzamy platformy na różnych wysokościach w okolicy
        best = None
        best_dist = 999
        for p in plats:
            if is_moving(p): continue
            if p[4] == 'g' if len(p) > 4 and isinstance(p[4], str) else False:
                continue  # ziemia – pomijamy (może być dziurawa)
            px0, px1 = p[0], p[0] + p[2]
            overlap = min(sx1, px1) - max(sx0, px0)
            if overlap < 8: continue  # za małe nakładanie
            dist = abs(p[1] - (sy + 16))
            if dist < best_dist and dist <= 60:
                best_dist = dist
                best = p

        if best:
            new_sy = int(best[1] - 16)
            new_spk = [int(spk[0]), new_sy, int(spk[2])]
            changes.append(f"  → spk {spk} przenoszę na static plat y={best[1]}: spk y={new_sy}")
            new_spks.append(new_spk)
        else:
            changes.append(f"  ✗ spk {spk} → BRAK statycznej platformy w pobliżu, usuwam")

    # Popraw kolce które nakładają się na gemy
    final_spks = []
    for spk in new_spks:
        sx0, sx1 = spike_x_range(spk)
        sy = spk[1]
        conflict = False
        for g in gems:
            gx, gy = g[0], g[1]
            # Kolec jest od sy-7 (czubek) do sy+19 (podstawa) wizualnie
            # Gem zbierany w promieniu 18
            if sx0 - 18 < gx < sx1 + 18 and sy - 40 < gy < sy + 30:
                changes.append(f"  ⚡ spk {spk} nakłada się z gem {g} → usuwam kolec")
                conflict = True
                break
        if not conflict:
            final_spks.append(spk)

    if changes or final_spks != spks:
        FIXES[lvl_n] = {'old_spk': spks, 'new_spk': final_spks, 'changes': changes}
        print(f"Poziom {lvl_n:2d}: {len(spks)} kolców → {len(final_spks)}")
        for c in changes:
            print(c)

# ── Poziom 1: gem przy wyjściu ──────────────────────────────────────────────
print("\n─── Sprawdzam poziom 1: gem przy wyjściu ───")
lvl1 = levels[0]
goal = lvl1['goal']
goal_cx = goal[0] + 16
goal_cy = goal[1] + 20
print(f"Goal center: ({goal_cx}, {goal_cy})")
for g in lvl1['gem']:
    dist = ((g[0]-goal_cx)**2 + (g[1]-goal_cy)**2)**0.5
    if dist < 45:
        print(f"  ⚠ gem {g} za blisko goal! dist={dist:.1f}px")

# ── Podsumowanie ────────────────────────────────────────────────────────────
print(f"\n═══ Podsumowanie: {len(FIXES)} poziomów wymaga poprawek ═══")
for ln, fix in FIXES.items():
    old = [list(map(int,s)) for s in fix['old_spk']]
    new = [list(map(int,s)) for s in fix['new_spk']]
    print(f"  Lvl {ln:2d}: {old} → {new}")
