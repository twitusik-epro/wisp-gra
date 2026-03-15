'use strict';
require('dotenv').config();

const express    = require('express');
const path       = require('path');
const cors       = require('cors');
const passport   = require('passport');
const { Strategy: GoogleStrategy } = require('passport-google-oauth20');
const jwt        = require('jsonwebtoken');
const { Paddle, Environment } = require('@paddle/paddle-node-sdk');
const Database   = require('better-sqlite3');

// ─── Config ────────────────────────────────────────────────────────────────
const PORT              = process.env.PORT              || 3001;
const BASE_URL          = process.env.BASE_URL          || `http://localhost:${PORT}`;
const JWT_SECRET        = process.env.JWT_SECRET;
const GOOGLE_CLIENT_ID  = process.env.GOOGLE_CLIENT_ID;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const PADDLE_API_KEY       = process.env.PADDLE_API_KEY;
const PADDLE_WEBHOOK_SECRET = process.env.PADDLE_WEBHOOK_SECRET;
const PADDLE_ENV           = process.env.PADDLE_ENV || 'sandbox'; // 'sandbox' lub 'production'
const NODE_ENV          = process.env.NODE_ENV || 'development';
const ADMIN_USER        = process.env.ADMIN_USER || 'admin';
const ADMIN_PASS        = process.env.ADMIN_PASS || 'changeme';

if (!JWT_SECRET || JWT_SECRET.length < 32) {
  console.error('BŁĄD: JWT_SECRET nie ustawiony lub za krótki (min 32 znaki)!');
  process.exit(1);
}

// ─── Database ──────────────────────────────────────────────────────────────
const db = new Database(path.join(__dirname, 'database', 'wisp.db'));
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

db.exec(`
  CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    google_id   TEXT UNIQUE,
    email       TEXT UNIQUE,
    nick        TEXT,
    avatar_url  TEXT,
    lives       INTEGER DEFAULT 3,
    score       INTEGER DEFAULT 0,
    level       INTEGER DEFAULT 1,
    difficulty  TEXT DEFAULT 'medium',
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login  DATETIME
  );

  CREATE TABLE IF NOT EXISTS scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
    score       INTEGER,
    level       INTEGER,
    difficulty  TEXT,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS purchases (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER REFERENCES users(id) ON DELETE CASCADE,
    paddle_txn_id     TEXT UNIQUE,
    package_id        TEXT,
    lives             INTEGER,
    amount_eur_ct     INTEGER,
    status            TEXT DEFAULT 'pending',
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);

// Dodaj kolumnę progress_ts jeśli jeszcze nie istnieje (migracja)
try { db.exec('ALTER TABLE users ADD COLUMN progress_ts INTEGER DEFAULT 0'); } catch {}

// Prepared statements
const stmts = {
  upsertUser: db.prepare(`
    INSERT INTO users (google_id, email, nick, avatar_url, last_login)
    VALUES (@google_id, @email, @nick, @avatar_url, CURRENT_TIMESTAMP)
    ON CONFLICT(google_id) DO UPDATE SET
      email      = excluded.email,
      nick       = COALESCE(users.nick, excluded.nick),
      avatar_url = excluded.avatar_url,
      last_login = CURRENT_TIMESTAMP
    RETURNING *
  `),
  getUserById:    db.prepare('SELECT * FROM users WHERE id = ?'),
  getUserByEmail: db.prepare('SELECT * FROM users WHERE email = ?'),
  addLives:       db.prepare('UPDATE users SET lives = lives + ? WHERE id = ?'),
  setScore:       db.prepare('UPDATE users SET score = ?, level = ?, difficulty = ? WHERE id = ?'),
  insertScore:    db.prepare('INSERT INTO scores (user_id, score, level, difficulty) VALUES (?, ?, ?, ?)'),
  topScores:      db.prepare(`
    SELECT u.nick, u.avatar_url, s.score, s.level, s.difficulty, s.created_at
    FROM scores s
    JOIN users u ON u.id = s.user_id
    ORDER BY s.score DESC
    LIMIT 20
  `),
  insertPurchase: db.prepare(`
    INSERT OR IGNORE INTO purchases (user_id, paddle_txn_id, package_id, lives, amount_eur_ct, status)
    VALUES (@user_id, @paddle_txn_id, @package_id, @lives, @amount_eur_ct, 'completed')
  `),
  saveProgress: db.prepare('UPDATE users SET lives = ?, level = ?, score = ?, difficulty = ?, progress_ts = ? WHERE id = ?'),

  // Admin
  listUsers:       db.prepare(`SELECT id, nick, email, lives, score, level, difficulty, created_at, last_login FROM users ORDER BY last_login DESC LIMIT 200`),
  searchUsers:     db.prepare(`SELECT id, nick, email, lives, score, level, difficulty, created_at, last_login FROM users WHERE nick LIKE @q OR email LIKE @q ORDER BY last_login DESC LIMIT 50`),
  deleteUser:      db.prepare('DELETE FROM users WHERE id = ?'),
  setLives:        db.prepare('UPDATE users SET lives = ? WHERE id = ?'),
  statsUsers:      db.prepare(`SELECT COUNT(*) as total, COUNT(CASE WHEN last_login > datetime('now','-7 days') THEN 1 END) as active_7d, COUNT(CASE WHEN last_login > datetime('now','-30 days') THEN 1 END) as active_30d, COUNT(CASE WHEN created_at > datetime('now','-7 days') THEN 1 END) as new_7d FROM users`),
  statsPurchases:  db.prepare(`SELECT COUNT(*) as total, COALESCE(SUM(CASE WHEN status='completed' THEN amount_eur_ct ELSE 0 END),0) as revenue_gr, COUNT(CASE WHEN status='completed' THEN 1 END) as completed FROM purchases`),
  recentPurchases: db.prepare(`SELECT p.id, p.user_id, p.package_id, p.lives, p.amount_pln_gr, p.status, p.created_at, u.nick, u.email FROM purchases p LEFT JOIN users u ON u.id = p.user_id ORDER BY p.created_at DESC LIMIT 100`),
  purgeOldUsers:   db.prepare(`DELETE FROM users WHERE last_login < datetime('now','-3 years') AND last_login IS NOT NULL`),
};

// ─── Paddle ────────────────────────────────────────────────────────────────
const paddle = PADDLE_API_KEY ? new Paddle(PADDLE_API_KEY, {
  environment: PADDLE_ENV === 'production' ? Environment.Production : Environment.Sandbox,
}) : null;

const PACKAGES = {
  pack_10: { lives: 10, amount:  99, price_id: 'pri_01kkrrj131kge5bgkxrg1dvybb', price_display: '0,99 €' },
  pack_25: { lives: 25, amount: 199, price_id: 'pri_01kkrrm69dvc1rn4ke4gm22062', price_display: '1,99 €' },
  pack_50: { lives: 50, amount: 399, price_id: 'pri_01kkrrnv2cv70zvrk0c781sdkg', price_display: '3,99 €' },
};

// ─── Express App ───────────────────────────────────────────────────────────
const app = express();

// Paddle webhook musi dostać raw body — PRZED express.json()
app.post('/webhook/paddle', express.raw({ type: 'application/json' }), handlePaddleWebhook);

app.use(cors({
  origin: BASE_URL,
  credentials: true,
}));
app.use(express.json());
app.use(passport.initialize());

// ─── Admin Routes ───────────────────────────────────────────────────────────
app.get('/admin', adminAuth, (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'admin.html'));
});
app.get('/admin.html', adminAuth, (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'admin.html'));
});

app.get('/api/admin/stats', adminAuth, (_req, res) => {
  const users     = stmts.statsUsers.get();
  const purchases = stmts.statsPurchases.get();
  res.json({ users, purchases });
});

app.get('/api/admin/users', adminAuth, (req, res) => {
  const q = req.query.q ? `%${req.query.q}%` : null;
  const rows = q ? stmts.searchUsers.all({ q }) : stmts.listUsers.all();
  res.json(rows);
});

app.post('/api/admin/users/:id/lives', adminAuth, (req, res) => {
  const id    = parseInt(req.params.id);
  const user  = stmts.getUserById.get(id);
  if (!user) return res.status(404).json({ error: 'Użytkownik nie istnieje' });
  const { lives, delta } = req.body;
  let newLives;
  if (typeof delta === 'number') newLives = Math.max(0, user.lives + delta);
  else if (typeof lives === 'number') newLives = Math.max(0, lives);
  else return res.status(400).json({ error: 'Podaj lives lub delta' });
  stmts.setLives.run(newLives, id);
  res.json({ ok: true, lives: newLives });
});

app.delete('/api/admin/users/:id', adminAuth, (req, res) => {
  const id = parseInt(req.params.id);
  stmts.deleteUser.run(id);
  res.json({ ok: true });
});

app.get('/api/admin/purchases', adminAuth, (_req, res) => {
  const rows = stmts.recentPurchases.all();
  res.json(rows);
});

// Static files — gra HTML5
app.use(express.static(path.join(__dirname, 'public'), {
  setHeaders(res, filePath) {
    if (filePath.endsWith('.html')) {
      res.setHeader('Cache-Control', 'no-cache, must-revalidate');
    } else {
      res.setHeader('Cache-Control', 'public, max-age=3600');
    }
  }
}));

// ─── Passport Google OAuth ─────────────────────────────────────────────────
if (GOOGLE_CLIENT_ID && GOOGLE_CLIENT_SECRET) {
  passport.use(new GoogleStrategy(
    {
      clientID:     GOOGLE_CLIENT_ID,
      clientSecret: GOOGLE_CLIENT_SECRET,
      callbackURL:  `${BASE_URL}/auth/google/callback`,
    },
    (_accessToken, _refreshToken, profile, done) => {
      const email     = profile.emails?.[0]?.value || null;
      const nick      = profile.displayName || email?.split('@')[0] || 'Gracz';
      const avatarUrl = profile.photos?.[0]?.value || null;

      try {
        const user = stmts.upsertUser.get({
          google_id:  profile.id,
          email,
          nick,
          avatar_url: avatarUrl,
        });
        done(null, user);
      } catch (err) {
        done(err);
      }
    }
  ));
}

// ─── Admin Auth Middleware ──────────────────────────────────────────────────
function adminAuth(req, res, next) {
  const header = req.headers.authorization || '';
  if (!header.startsWith('Basic ')) {
    res.set('WWW-Authenticate', 'Basic realm="Wisp Admin"');
    return res.status(401).send('Unauthorized');
  }
  const decoded = Buffer.from(header.slice(6), 'base64').toString();
  const colon   = decoded.indexOf(':');
  const user    = decoded.slice(0, colon);
  const pass    = decoded.slice(colon + 1);
  if (user === ADMIN_USER && pass === ADMIN_PASS) return next();
  res.set('WWW-Authenticate', 'Basic realm="Wisp Admin"');
  res.status(401).send('Unauthorized');
}

// ─── Auth Middleware ────────────────────────────────────────────────────────
function requireAuth(req, res, next) {
  const header = req.headers.authorization || '';
  const token  = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Brak tokenu autoryzacji' });

  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Token nieprawidłowy lub wygasły' });
  }
}

// ─── Auth Routes ───────────────────────────────────────────────────────────
app.get('/auth/google',
  passport.authenticate('google', { scope: ['profile', 'email'], session: false })
);

app.get('/auth/google/callback',
  passport.authenticate('google', { session: false, failureRedirect: '/game.html?auth=error' }),
  (req, res) => {
    const token = jwt.sign(
      { id: req.user.id, email: req.user.email, nick: req.user.nick },
      JWT_SECRET,
      { expiresIn: '30d' }
    );
    // Przekieruj do gry z tokenem w URL (gra odbierze i zapisze w localStorage)
    res.redirect(`/game.html?token=${token}`);
  }
);

app.get('/auth/me', requireAuth, (req, res) => {
  const user = stmts.getUserById.get(req.user.id);
  if (!user) return res.status(404).json({ error: 'Użytkownik nie istnieje' });
  const { google_id, ...safe } = user; // nie zwracamy google_id
  res.json(safe);
});

app.post('/auth/logout', requireAuth, (_req, res) => {
  // JWT jest bezstanowy — wylogowanie odbywa się po stronie klienta (usunięcie tokenu)
  res.json({ ok: true });
});

// ─── Game State Routes ──────────────────────────────────────────────────────
// Zapis wyniku na serwer
app.post('/api/score', requireAuth, (req, res) => {
  const { score, level, difficulty } = req.body;
  if (typeof score !== 'number' || typeof level !== 'number') {
    return res.status(400).json({ error: 'Nieprawidłowe dane' });
  }

  const userId = req.user.id;
  const user   = stmts.getUserById.get(userId);
  if (!user) return res.status(404).json({ error: 'Użytkownik nie istnieje' });

  stmts.insertScore.run(userId, score, level, difficulty || 'medium');

  // Zaktualizuj rekord użytkownika jeśli lepszy wynik
  if (score > (user.score || 0)) {
    stmts.setScore.run(score, level, difficulty || 'medium', userId);
  }

  res.json({ ok: true, score });
});

// Globalny ranking Top 20
app.get('/api/leaderboard', (_req, res) => {
  const rows = stmts.topScores.all();
  res.json(rows);
});

// Synchronizacja życ z serwerem (po zalogowaniu)
app.get('/api/lives', requireAuth, (req, res) => {
  const user = stmts.getUserById.get(req.user.id);
  if (!user) return res.status(404).json({ error: 'Użytkownik nie istnieje' });
  res.json({ lives: user.lives });
});

// Synchronizacja postępu gry między urządzeniami
app.get('/api/progress', requireAuth, (req, res) => {
  const user = stmts.getUserById.get(req.user.id);
  if (!user) return res.status(404).json({ error: 'Użytkownik nie istnieje' });
  res.json({ level: user.level, score: user.score, lives: user.lives, difficulty: user.difficulty, progress_ts: user.progress_ts || 0 });
});

app.post('/api/progress', requireAuth, (req, res) => {
  const { level, score, lives, difficulty, ts } = req.body;
  if (typeof level !== 'number' || typeof score !== 'number') {
    return res.status(400).json({ error: 'Nieprawidłowe dane' });
  }
  // ts od klienta jest w ms (Date.now()), konwertujemy do sekund Unix
  const progressTs = typeof ts === 'number' ? Math.floor(ts / 1000) : Math.floor(Date.now() / 1000);
  stmts.saveProgress.run(
    Math.max(1, lives || 3),
    Math.max(1, level),
    Math.max(0, score),
    difficulty || 'medium',
    progressTs,
    req.user.id
  );
  res.json({ ok: true });
});

// ─── Paddle Routes ──────────────────────────────────────────────────────────
app.post('/api/buy', requireAuth, (req, res) => {
  if (!paddle) return res.status(503).json({ error: 'Płatności tymczasowo niedostępne' });

  const { package_id } = req.body;
  const pkg = PACKAGES[package_id];
  if (!pkg) return res.status(400).json({ error: 'Nieznany pakiet' });

  // Paddle Checkout otwierany po stronie klienta (Paddle.js overlay)
  // Zwracamy price_id + metadane — frontend otwiera checkout
  res.json({
    price_id:    pkg.price_id,
    user_id:     req.user.id,
    package_id,
    lives:       pkg.lives,
  });
});

// ─── Paddle Webhook ──────────────────────────────────────────────────────────
async function handlePaddleWebhook(req, res) {
  console.log('📦 Paddle webhook');
  if (!PADDLE_WEBHOOK_SECRET) {
    return res.status(503).json({ error: 'Webhook nie skonfigurowany' });
  }

  // Weryfikacja podpisu HMAC-SHA256 bez SDK
  const sig = req.headers['paddle-signature'];
  if (!sig) return res.status(400).json({ error: 'Brak podpisu' });
  try {
    const [tsPart, h1Part] = sig.split(';');
    const ts = tsPart.split('=')[1];
    const h1 = h1Part.split('=')[1];
    const body = req.body.toString();
    const computed = require('crypto').createHmac('sha256', PADDLE_WEBHOOK_SECRET)
      .update(`${ts}:${body}`).digest('hex');
    if (computed !== h1) {
      console.error('Paddle webhook: nieprawidłowy podpis');
      return res.status(400).json({ error: 'Invalid signature' });
    }
  } catch (err) {
    console.error('Paddle webhook signature error:', err.message);
    return res.status(400).json({ error: err.message });
  }

  let event;
  try {
    event = JSON.parse(req.body.toString());
  } catch (err) {
    return res.status(400).json({ error: 'Invalid JSON' });
  }

  if (event.event_type === 'transaction.completed') {
    const txn = event.data;
    const customData = txn.custom_data || {};
    try {
      const userId  = parseInt(customData.user_id);
      const pkgId   = customData.package_id;
      const pkg     = PACKAGES[pkgId];

      if (!userId || !pkg) {
        console.error('Paddle webhook: brak user_id lub package_id w customData', customData);
        return res.json({ received: true });
      }

      const result = stmts.insertPurchase.run({
        user_id:      userId,
        paddle_txn_id: txn.id,
        package_id:   pkgId,
        lives:        pkg.lives,
        amount_eur_ct: pkg.amount,
      });

      if (result.changes > 0) {
        stmts.addLives.run(pkg.lives, userId);
        console.log(`✅ Dodano ${pkg.lives} żyć użytkownikowi ${userId} (txn: ${txn.id})`);
      } else {
        console.log(`ℹ️  Duplikat transakcji zignorowany: ${txn.id}`);
      }
    } catch (err) {
      console.error('DB error przy Paddle webhook:', err.message);
      return res.status(500).json({ error: 'DB error' });
    }
  }

  res.json({ received: true });
}

// ─── Health Check ───────────────────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    version: '1.0.0',
    game: 'Wisp v19',
    env: NODE_ENV,
  });
});

// ─── Auto-cleanup nieaktywnych kont (>3 lata) ───────────────────────────────
function purgeStaleAccounts() {
  const r = stmts.purgeOldUsers.run();
  if (r.changes > 0) console.log(`🗑️  Usunięto ${r.changes} nieaktywnych kont (>3 lata bez logowania)`);
}
purgeStaleAccounts(); // przy starcie
setInterval(purgeStaleAccounts, 24 * 60 * 60 * 1000); // co dobę

// ─── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, '127.0.0.1', () => {
  console.log(`🎮 Wisp serwer uruchomiony na http://127.0.0.1:${PORT}`);
  console.log(`   BASE_URL: ${BASE_URL}`);
  console.log(`   Paddle: ${paddle ? `✅ aktywny (${PADDLE_ENV})` : '⚠️  brak klucza (tryb testowy)'}`);
  console.log(`   Google OAuth: ${GOOGLE_CLIENT_ID ? '✅ aktywny' : '⚠️  brak CLIENT_ID'}`);
});
