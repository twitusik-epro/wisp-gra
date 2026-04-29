// Service Worker — Wisp: Duch Lasu
const CACHE = 'wisp-v7';
const STATIC = ['./manifest.json'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(STATIC)));
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  // HTML — zawsze z sieci
  if (url.pathname.endsWith('.html') || url.pathname === '/' || url.pathname === '') {
    e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
    return;
  }
  // API i auth — zawsze z sieci (dane dynamiczne, nigdy z cache!)
  if (url.pathname.startsWith('/api/') || url.pathname.startsWith('/auth/') || url.pathname.startsWith('/webhook/')) {
    e.respondWith(fetch(e.request));
    return;
  }
  // Pozostałe zasoby statyczne — cache first
  e.respondWith(
    caches.match(e.request).then(r => r || fetch(e.request).then(res => {
      if (res.ok) {
        const clone = res.clone();
        caches.open(CACHE).then(c => c.put(e.request, clone));
      }
      return res;
    }))
  );
});
