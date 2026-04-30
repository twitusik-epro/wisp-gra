# Wisp — Duch Lasu · Dev Log

Plik historii zmian projektu. Aktualizowany po każdej sesji roboczej.
Repo: https://github.com/twitusik-epro/wisp-gra

---

## 2026-04-30
- **Nowy landing page** (`public/index.html`) — pełny redesign:
  - Hero fullscreen z tłem (`hero-bg-0.jpg`), animacja świetlików (canvas fireflies)
  - Dwie karty portali (Wisp + Forest Cards) z `clip-path: polygon(16%)` — ścięte narożniki 45°
  - i18n 5 języków (PL/EN/DE/FR/ES), blog sekcja, footer
  - `wisp-menu-icon.png` — usunięto białe tło PNG (Pillow → transparent)
- **Paddle wyłączone na desktopie** (`public/game.html`):
  - `showShopButtons()`: jeśli `!IS_TWA` → tekst o instalacji na telefon zamiast przycisków
  - Powód: Paddle usunęło grę z platformy płatności
  - Na TWA (Android) — Google Play Billing bez zmian
  - Tłumaczenia w 5 językach (`LANG.shopMobile`)
- **GSC** — przeanalizowano 5 problemów, żaden nie wymaga zmian w kodzie; do kliknięcia "Sprawdź poprawkę" w GSC dla www. i http://

## 2026-04-29
- **Fajerwerki po ukończeniu poziomu 40** — canvas animation identyczna jak w Forest Cards
- **Blokada Continue po lvl40** — po ukończeniu gry czyszczony zapis lokalny i serwerowy
- **Fix stompu muchy** — okno kolizji 22px (było 5.5px), próg `P.vy > 0`
- **Kolor "Usuń zapis"** → `#ffe080` (jak przycisk Continue)
- **Popstate fix** — przycisk Wstecz (Android) zamyka panele zamiast wychodzić z gry
- **TWA v1.31 (versionCode 32)** — splash.png bez białych narożników (Pillow recolor → `#060E06`)
- **SW cache** bumped do `wisp-v7`
- **Wisp Asset Studio** (`asset-server/`) — panel generowania assetów AI:
  - Muzyka AI: MusicGen-small, progress ticker, MP3 export
  - Wideo AI: CogVideoX-5b, sequential CPU offload, upscaling FHD/2K/4K, port 3004

## 2026-04-22 — TWA v1.30 (versionCode 31)
- Bump wersji po wdrożeniu muzyki/tła/edu na web

## 2026-04-21 — Muzyka, tła, edu fakty
- Muzyka tła MP3 — `world1-4.mp3` (`assets/music/`)
  - `_bgmPlay` używa `new window.Audio()` (NIE `new Audio()` — IIFE scope shadow!)
- Tła poziomów AI — `level01-40.png` (`assets/bg/`), SDXL, 4 światy wizualnie
- Ekrany wygranej — `win_w1-4.png` (`assets/ui/`), SDXL, per świat
- Tło menu — `menu_bg_v3.png` (nocny bajkowy las)
- Edu fakty — `EDU_FACTS_ALL` 5 języków, 40 faktów/język

## 2026-04-20 — Nowy sprite duszka i splash
- `wisp-ghost.png` (2048×2048 RGBA) — nowe logo zamiast canvas-drawn oval
- `wisp-splash.jpg` — splash screen 1200ms + fade 650ms
- TWA v1.28 (versionCode 29): nowy sprite w grze + splash
- TWA v1.29 (versionCode 30): fix splash (piękna scena zamiast białych narożników ikony)

## 2026-04-19 — Tekstury platform
- `assets/platforms/w1_log.png … w4_snow.png` (8 PNG, 768×170px)
- Absolutna ścieżka `/assets/platforms/${name}.png?v=2` (fix mobile TWA)
- Load state: `_platLoaded[name] = true` w `img.onload` (NIE `img.complete` — zawodne na WebView)
- Renderowanie: `drawImage` tiling (NIE `createPattern` — null na niektórych WebView)
- TWA v1.25 (versionCode 26): Android 15 edge-to-edge

## 2026-04-15 — Google Play Billing
- **Fix root cause**: `com.google.androidbrowserhelper:billing:1.1.0` (zawiera `PaymentActivity`)
- Cena/waluta ze sklepu Play: `updatePlayPrices()` → `getDigitalGoodsService` → `getDetails`
- Backend: POST `/api/gplay/verify` — Google Play Developer API (androidpublisher v3)
- Kluczowa lekcja: `PaymentActivity` jest w osobnym artefakcie `billing`, NIE w `androidbrowserhelper`

## 2026-04-01 — SEO i blog
- Blog: 5 języków (PL/EN/DE/FR/ES), 30+ artykułów, hreflang
- `game.html`: noindex + usunięta z sitemap.xml
- `sitemap.xml`: dodane `/blog/en/`, `/blog/de/`, `/blog/fr/`, `/blog/es/`
- www.wispplay.com → 301 redirect na https://wispplay.com (nginx)

## 2026-03-15 — Pierwsze wdrożenie
- Gra HTML5 na wispplay.com, port 3001, PM2 process `wisp`
- Google OAuth, SQLite, Paddle billing (web), JWT auth
- 40 poziomów w 4 światach, cross-device save sync
- Fixed timestep game loop (60Hz i 120Hz)
