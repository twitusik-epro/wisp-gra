'use strict';
require('dotenv').config();

const express    = require('express');
const path       = require('path');
const cors       = require('cors');
const passport   = require('passport');
const { Strategy: GoogleStrategy } = require('passport-google-oauth20');
const jwt        = require('jsonwebtoken');
const Stripe     = require('stripe');
const Database   = require('better-sqlite3');

// ─── Config ────────────────────────────────────────────────────────────────
const PORT              = process.env.PORT              || 3001;
const BASE_URL          = process.env.BASE_URL          || `http://localhost:${PORT}`;
const JWT_SECRET        = process.env.JWT_SECRET;
const GOOGLE_CLIENT_ID  = process.env.GOOGLE_CLIENT_ID;
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET;
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;
const STRIPE_WEBHOOK_SECRET = process.env.STRIPE_WEBHOOK_SECRET;
const NODE_ENV          = process.env.NODE_ENV || 'development';

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
    stripe_session_id TEXT UNIQUE,
    package_id        TEXT,
    lives             INTEGER,
    amount_pln_gr     INTEGER,
    status            TEXT DEFAULT 'pending',
    created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
  );
`);

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
    INSERT INTO purchases (user_id, stripe_session_id, package_id, lives, amount_pln_gr, status)
    VALUES (@user_id, @stripe_session_id, @package_id, @lives, @amount_pln_gr, 'pending')
  `),
  completePurchase: db.prepare(`
    UPDATE purchases SET status = 'completed' WHERE stripe_session_id = ? AND status = 'pending'
    RETURNING user_id, lives
  `),
  getPurchaseBySession: db.prepare('SELECT * FROM purchases WHERE stripe_session_id = ?'),
  saveProgress: db.prepare('UPDATE users SET lives = ?, level = ?, score = ?, difficulty = ? WHERE id = ?'),
};

// ─── Stripe ────────────────────────────────────────────────────────────────
const stripe = STRIPE_SECRET_KEY ? new Stripe(STRIPE_SECRET_KEY) : null;

const PACKAGES = {
  pack_10: { lives: 10, amount: 100,  label: '10 żyć',  price_pln: '1,00 zł' },
  pack_25: { lives: 25, amount: 250,  label: '25 żyć',  price_pln: '2,50 zł' },
  pack_50: { lives: 50, amount: 500,  label: '50 żyć',  price_pln: '5,00 zł' },
};

// ─── Express App ───────────────────────────────────────────────────────────
const app = express();

// Stripe webhook musi dostać raw body — PRZED express.json()
app.post('/webhook/stripe', express.raw({ type: 'application/json' }), handleStripeWebhook);

app.use(cors({
  origin: BASE_URL,
  credentials: true,
}));
app.use(express.json());
app.use(passport.initialize());

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
  res.json({ level: user.level, score: user.score, lives: user.lives, difficulty: user.difficulty });
});

app.post('/api/progress', requireAuth, (req, res) => {
  const { level, score, lives, difficulty } = req.body;
  if (typeof level !== 'number' || typeof score !== 'number') {
    return res.status(400).json({ error: 'Nieprawidłowe dane' });
  }
  stmts.saveProgress.run(
    Math.max(1, lives || 3),
    Math.max(1, level),
    Math.max(0, score),
    difficulty || 'medium',
    req.user.id
  );
  res.json({ ok: true });
});

// ─── Stripe Routes ──────────────────────────────────────────────────────────
app.post('/api/buy', requireAuth, async (req, res) => {
  if (!stripe) return res.status(503).json({ error: 'Płatności tymczasowo niedostępne' });

  const { package_id } = req.body;
  const pkg = PACKAGES[package_id];
  if (!pkg) return res.status(400).json({ error: 'Nieznany pakiet' });

  const userId = req.user.id;

  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card', 'blik', 'p24'],
      line_items: [{
        price_data: {
          currency: 'pln',
          product_data: {
            name:        `Wisp — ${pkg.label}`,
            description: `Dodaj ${pkg.lives} żyć do swojego konta w grze Wisp — Duch Lasu`,
          },
          unit_amount: pkg.amount, // w groszach
        },
        quantity: 1,
      }],
      mode:        'payment',
      success_url: `${BASE_URL}/game.html?purchase=success`,
      cancel_url:  `${BASE_URL}/game.html?purchase=cancel`,
      metadata: {
        user_id:    String(userId),
        package_id,
        lives:      String(pkg.lives),
      },
      client_reference_id: String(userId),
    });

    // Zapisz pending purchase
    stmts.insertPurchase.run({
      user_id:          userId,
      stripe_session_id: session.id,
      package_id,
      lives:            pkg.lives,
      amount_pln_gr:    pkg.amount,
    });

    res.json({ url: session.url });
  } catch (err) {
    console.error('Stripe error:', err.message);
    res.status(500).json({ error: 'Błąd podczas tworzenia sesji płatności' });
  }
});

// ─── Stripe Webhook ─────────────────────────────────────────────────────────
async function handleStripeWebhook(req, res) {
  if (!stripe || !STRIPE_WEBHOOK_SECRET) {
    return res.status(503).json({ error: 'Webhook nie skonfigurowany' });
  }

  let event;
  try {
    event = stripe.webhooks.constructEvent(
      req.body,
      req.headers['stripe-signature'],
      STRIPE_WEBHOOK_SECRET
    );
  } catch (err) {
    console.error('Webhook signature error:', err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  if (event.type === 'checkout.session.completed') {
    const session = event.data.object;
    try {
      const rows = stmts.completePurchase.all(session.id);
      if (rows.length > 0) {
        const { user_id, lives } = rows[0];
        stmts.addLives.run(lives, user_id);
        console.log(`✅ Dodano ${lives} żyć użytkownikowi ${user_id} (session: ${session.id})`);
      }
    } catch (err) {
      console.error('DB error przy webhook:', err.message);
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

// ─── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, '127.0.0.1', () => {
  console.log(`🎮 Wisp serwer uruchomiony na http://127.0.0.1:${PORT}`);
  console.log(`   BASE_URL: ${BASE_URL}`);
  console.log(`   Stripe: ${stripe ? '✅ aktywny' : '⚠️  brak klucza (tryb testowy)'}`);
  console.log(`   Google OAuth: ${GOOGLE_CLIENT_ID ? '✅ aktywny' : '⚠️  brak CLIENT_ID'}`);
});
