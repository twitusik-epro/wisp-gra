"""
Naprawia kolce we wszystkich 40 poziomach:
- Usuwa lewitujące (brak gruntu pod spodem)
- Usuwa ze ruchomych platform
- Usuwa nakładające się z diamentami
- Dodaje nowe na statycznych platformach (bezpieczne pozycje)
- Naprawia diament przy wyjściu na poz.1
"""
import re, copy

GAME_HTML = "/opt/gry/Wisp - NOWA wersja w budowie/public/game.html"

# ── helpers ────────────────────────────────────────────────────────────────
def parse_arr(s):
    s = re.sub(r'//.*', '', s)
    items = re.findall(r'\[([^\[\]]+)\]', s)
    result = []
    for it in items:
        vals = []
        for x in it.split(','):
            x = x.strip().strip("'\"")
            if not x: continue
            try: vals.append(float(x))
            except ValueError: vals.append(x)
        result.append(vals)
    return result

def is_moving(p):
    return len(p) >= 5 and p[4] == 1

def is_ground(p):
    return len(p) >= 5 and p[4] == 'g'

def gem_safe(cx, sy, gems, margin=38):
    """True jeśli pozycja (cx, sy) jest bezpieczna od wszystkich gemów"""
    for g in gems:
        if abs(g[0] - cx) < margin and abs(g[1] - sy) < 55:
            return False
    return True

def goal_safe(cx, sy, goal, margin=45):
    """True jeśli pozycja jest bezpieczna od wyjścia"""
    gcx = goal[0] + 16
    gcy = goal[1] + 20
    return abs(cx - gcx) > margin or abs(sy - gcy) > 55

def find_spike_positions(plat, gems, goal, spike_w=20, min_plat_w=30):
    """
    Dla danej statycznej platformy znajdź do 2 bezpiecznych pozycji na kolce.
    Zwraca listę [x, y_plat-16, spike_w]
    """
    if is_moving(plat) or is_ground(plat): return []
    px, py, pw = plat[0], plat[1], plat[2]
    if pw < min_plat_w: return []

    sy = py - 16  # y kolca = 16px powyżej wierzchu platformy
    positions = []

    # Spróbuj 3 pozycje: lewa, środek, prawa krawędź (z marginesem)
    candidates = []
    margin = 8
    if pw >= 40:
        candidates.append(px + margin)                   # lewa część
    if pw >= 50:
        candidates.append(px + pw//2 - spike_w//2)      # środek
    if pw >= 40:
        candidates.append(px + pw - spike_w - margin)   # prawa część

    for cx_left in candidates:
        cx = cx_left + spike_w // 2
        if gem_safe(cx, sy, gems) and goal_safe(cx, sy, goal):
            positions.append([int(cx_left), int(sy), spike_w])
            if len(positions) >= 1:
                break  # max 1 kolec na platformę

    return positions

# ── wczytaj game.html ──────────────────────────────────────────────────────
with open(GAME_HTML, encoding='utf-8') as f:
    src = f.read()

raw_match = re.search(r'const configs = \[(.*?)\];\s*\n\s*const cfg', src, re.DOTALL)
raw = raw_match.group(1)

level_blocks = re.findall(
    r'\{plat:\[([\s\S]*?)\]\s*,\s*\n?\s*spk:\[([\s\S]*?)\]\s*,\s*\n?\s*gem:\[([\s\S]*?)\]\s*,\s*\n?\s*goal:\[([^\]]+)\]',
    raw
)

levels = []
for plat_s, spk_s, gem_s, goal_s in level_blocks:
    plats = parse_arr(plat_s)
    spks  = parse_arr(spk_s)
    gems  = parse_arr(gem_s)
    goal  = [float(x.strip()) for x in goal_s.split(',')]
    levels.append({'plat': plats, 'spk': spks, 'gem': gems, 'goal': goal})

print(f"Wczytano {len(levels)} poziomów\n")

# ── generuj poprawione spk per level ──────────────────────────────────────
new_spks_per_level = {}

for lvl_i, lvl in enumerate(levels):
    lvl_n = lvl_i + 1
    plats = lvl['plat']
    gems  = lvl['gem']
    goal  = lvl['goal']

    new_spks = []

    # 1. Kolce na ziemi — tylko tam gdzie jest faktyczna platforma ziemna
    ground_plats = [p for p in plats if is_ground(p)]
    for gp in ground_plats:
        gx, gy, gw = gp[0], gp[1], gp[2]
        sy = gy - 16
        # Podziel na segmenty i dodaj kolce z zachowaniem odległości od gemów
        # Jeden kolec na segment ziemi, 40% długości, z prawej strony
        if gw >= 80:
            # 2 kolce: w 1/4 i 3/4 segmentu
            positions = [
                int(gx + gw * 0.18),
                int(gx + gw * 0.65),
            ]
            for xp in positions:
                cx = xp + 15
                if gem_safe(cx, sy, gems, 35) and goal_safe(cx, sy, goal, 40):
                    new_spks.append([xp, int(sy), 30])
        elif gw >= 40:
            xp = int(gx + gw * 0.3)
            cx = xp + 10
            if gem_safe(cx, sy, gems, 35) and goal_safe(cx, sy, goal, 40):
                new_spks.append([xp, int(sy), 20])

    # 2. Kolce na statycznych platformach (nie ziemia, nie ruchome)
    static_plats = [p for p in plats if not is_moving(p) and not is_ground(p)]

    # Wybierz co drugą/trzecią platformę dla kolców (od dołu, żeby pierwsza platforma była bez kolca)
    # Sortuj od dołu (największy y = najniżej)
    static_plats_sorted = sorted(static_plats, key=lambda p: -p[1])

    # Dla poziomów 1-10: co 3cia platforma; 11-20: co 2ga; 21+: co 2ga
    step = 3 if lvl_n <= 10 else 2
    for idx, p in enumerate(static_plats_sorted):
        if idx % step != 1: continue  # skip 0th, 2nd, 4th...
        positions = find_spike_positions(p, gems, goal)
        new_spks.extend(positions)

    # 3. Upewnij się że nie ma żadnych kolców nakładających się na gemy
    safe_spks = []
    for spk in new_spks:
        sx0, sx1 = spk[0], spk[0] + spk[2]
        cx = (sx0 + sx1) / 2
        sy = spk[1]
        if gem_safe(cx, sy, gems, 35) and goal_safe(cx, sy, goal, 40):
            safe_spks.append(spk)

    # Usuń duplikaty (bardzo blisko siebie)
    deduped = []
    for spk in safe_spks:
        too_close = False
        for prev in deduped:
            if abs(spk[0] - prev[0]) < 25 and abs(spk[1] - prev[1]) < 25:
                too_close = True
                break
        if not too_close:
            deduped.append(spk)

    new_spks_per_level[lvl_n] = deduped
    old_count = len(lvl['spk'])
    print(f"Lvl {lvl_n:2d}: {old_count} → {len(deduped)} kolców: {deduped}")

# ── Specjalna poprawka: poziom 1, diament przy wyjściu ────────────────────
# gem [300,70] jest za blisko goal [290,50] → przesunąć gem na bezpieczne miejsce
# Przesuniemy go na platformę [80,150,100,16] → gem nad centrum tej platformy
print("\n─── Poprawka diamentu lvl 1 ───")
print("  gem [300,70] za blisko goal → przesuwa się na [130,130]")

# ── Zastosuj poprawki w game.html ────────────────────────────────────────
new_src = src

# 1. Zamień spk:[] dla każdego poziomu
def format_spk(spk_list):
    if not spk_list:
        return "[]"
    items = ",".join(f"[{s[0]},{s[1]},{s[2]}]" for s in spk_list)
    return f"[{items}]"

# Znajdź każdy blok poziomu i zamień spk
level_pattern = re.compile(
    r'(\{plat:\[[\s\S]*?\]\s*,\s*\n?\s*)spk:\[([\s\S]*?)\](\s*,\s*\n?\s*gem:)',
)

matches = list(level_pattern.finditer(new_src))
print(f"\nZnaleziono {len(matches)} bloków spk do zamiany")

# Zamieniamy od tyłu żeby nie przesuwać indeksów
for i, m in enumerate(reversed(matches)):
    lvl_n = len(matches) - i  # 1..40
    new_spk_str = format_spk(new_spks_per_level.get(lvl_n, []))
    replacement = m.group(1) + f"spk:{new_spk_str}" + m.group(3)
    new_src = new_src[:m.start()] + replacement + new_src[m.end():]

# 2. Napraw diament przy wyjściu lvl 1
# gem:[[100,540],[270,480],[130,400],[310,340],[90,270],[260,200],[110,130],[300,70]]
# Zmień [300,70] na [130,130]  ← nad platformą [80,150] jest już gem [110,130]... hmm
# Znajdźmy inną pozycję: [295,130] - na platformie [260,90,120,16] ale dalej od goal
new_src = new_src.replace(
    "gem:[[100,540],[270,480],[130,400],[310,340],[90,270],[260,200],[110,130],[300,70]]",
    "gem:[[100,540],[270,480],[130,400],[310,340],[90,270],[260,200],[110,130],[270,130]]"
)

with open(GAME_HTML, 'w', encoding='utf-8') as f:
    f.write(new_src)

print("\n✓ Zapisano game.html z poprawionymi kolcami i diamentem lvl 1")
