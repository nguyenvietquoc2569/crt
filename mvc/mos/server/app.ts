// HTTP layer of the listening-test server: the rater API, the admin API, the stimuli and the static pages.
// Binds to 127.0.0.1 only; the public address is a tunnel in front of it. The server reads $MOS_DATA_DIR/package and
// never $MOS_DATA_DIR/private (the key, items.json, the audio cache): no route can reach them, and the server does
// not start while the package folder holds a key file (mvc/LISTENING_ADDENDUM_mvc_v1.md §11.4).
import { chmodSync, existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync,
         writeFileSync } from 'node:fs';
import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';
import { join } from 'node:path';
import type { Server } from 'bun';
import { AUDIO_TYPES, BLOCKS, type Block, browserFamily, C, codeHmac, csvRows, CSV_COLUMNS, extOf, type Half,
         type Headphone, hpFailed, intIn, ipKey, isTestData, LANGS, loadSampleData, type LtData, makeSession, nextTrial,
         normalizeCode, parseLtFile, planFor, PLAY_ROLES, PREFIX_RE, PUBLIC_DIR, randomCode, RANGE, resultsFor,
         sampleConsentHtml, sampleWav, sanitizeLtData, type Session, type SessionRow, sessionStimuli, sha256, SID_RE,
         TID_RE } from './core';
import { scrubCopy, type SessionMeta, Store } from './db';

export type Options = {
  dataDir: string;
  hostname?: string;
  port?: number;
  sample?: boolean;          // no package imported: serve the journal's sample data (TEST DATA)
  loginLimit?: number;       // failed logins per IP and window (successful ones do not count; default 100)
  loginTotalLimit?: number;  // all logins and registrations per IP and window (default 10 x loginLimit)
  registerLimit?: number;    // open registrations per window from all clients together (default 30)
  loginWindowMs?: number;
  trustProxy?: boolean;      // take the client IP of the rate limit from CF-Connecting-IP (cloudflared)
  hosts?: string[];          // host names served besides 127.0.0.1, localhost and [::1] (e.g. the tunnel's)
  cookieSecure?: boolean;
  backup?: boolean;          // the daily copy of the database in $MOS_DATA_DIR/backup (default on)
  log?: (msg: string) => void;
};

const BODY_LIMIT = 16 * 1024;
const MAX_MS = 864e5;   // rt_ms and t_ms are stored capped at 24 h
const CONSENT_LIMIT = 256 * 1024;
const COOKIE = 'mos_sid';
const SESSION_HOURS = 24;   // server-side lifetime of a sign-in; the cookie itself ends with the browser session
const CONSENT_FILE = 'consent_v1.html';   // the filled mvc/mos/consent_template_v1.html (both languages)
// The placeholders of mvc/mos/consent_template_v1.html and the draft fields of benchmark/listening/consent_v1.html.
export const CONSENT_PLACEHOLDERS = ['[CONTACT_NAME]', '[CONTACT_EMAIL]', '[ETHICS_NOTE]', '[RETENTION_DATE]',
                                     '[CONTACT]', '[COMPENSATION', '[ETHICS', '[THÙ LAO', '[ĐẠO ĐỨC'];
const RATER_ID_RE = /^R\d{3,}$/;
export const MAX_CODES = 40;   // the default cap on the codes issued by both routes (addendum §7.1)
const LOOPBACK = ['127.0.0.1', 'localhost', '[::1]'];
const STATIC: Record<string, [string, string]> = {
  '/': ['index.html', 'text/html; charset=utf-8'],
  '/index.html': ['index.html', 'text/html; charset=utf-8'],
  '/admin': ['admin.html', 'text/html; charset=utf-8'],
  '/style.css': ['style.css', 'text/css; charset=utf-8'],
  '/i18n.js': ['i18n.js', 'text/javascript; charset=utf-8'],
  '/lt_core.js': ['lt_core.js', 'text/javascript; charset=utf-8'],
  '/lt_app.js': ['lt_app.js', 'text/javascript; charset=utf-8'],
  '/admin.js': ['admin.js', 'text/javascript; charset=utf-8'],
};
const SECURITY_HEADERS: Record<string, string> = {
  'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; " +
    "media-src 'self'; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; " +
    "frame-ancestors 'none'",
  'X-Content-Type-Options': 'nosniff',
  'Referrer-Policy': 'no-referrer',
  'X-Frame-Options': 'DENY',
  'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'Permissions-Policy': 'camera=(), microphone=(), geolocation=(), payment=()',
};

// A tag's text with character references decoded and whitespace and control characters dropped, as a browser reads
// an attribute value ("java&#115;cript:", "jav&#x09;ascript:" and "java&Tab;script:" all become "javascript:").
function decodedTag(tag: string): string {
  const named: Record<string, string> = {colon: ':', tab: '\t', newline: '\n', sol: '/', lpar: '(', rpar: ')'};
  return tag.replace(/&#x([0-9a-f]+);?/gi, (_, h) => String.fromCodePoint(Math.min(parseInt(h, 16), 0x10ffff)))
    .replace(/&#(\d+);?/g, (_, d) => String.fromCodePoint(Math.min(Number(d), 0x10ffff)))
    .replace(/&(colon|tab|newline|sol|lpar|rpar);/gi, (_, n) => named[n.toLowerCase()])
    .replace(/[\x00-\x20]+/g, '');
}

// Why a consent text cannot be shown: scripts, event handlers, frames, forms or script URLs, and placeholders left
// unfilled. Handlers and URLs are looked for in every tag, also after "/" or a quote ("<svg/onload=...>") and behind
// character references. The page's content security policy blocks all of these as well.
export function consentProblems(html: string): string[] {
  const out: string[] = [];
  const tags = html.match(/<[^>]*>?/g) ?? [];
  const bad = tags.some((tag) => /^<\s*\/?\s*(script|iframe|frame|frameset|object|embed|applet|base|meta|link|form|portal)\b/i.test(tag) ||
                                 /[\s/"']on[a-z]+\s*=/i.test(tag) ||
                                 /(javascript|vbscript|livescript):/i.test(decodedTag(tag)));
  if (bad) out.push('the consent text holds scripts, event handlers, frames or forms');
  const left = CONSENT_PLACEHOLDERS.filter((m) => html.includes(m));
  if (left.length) out.push(`placeholders left: ${left.join(', ')}`);
  return out;
}

// Files named key*.json anywhere in the package folder (two levels deep).
function keyFiles(dir: string, depth = 0): string[] {
  if (!existsSync(dir)) return [];
  const out: string[] = [];
  for (const e of readdirSync(dir, {withFileTypes: true})) {
    const p = join(dir, e.name);
    if (/^key.*\.json$/i.test(e.name)) out.push(p);
    else if (e.isDirectory() && depth < 2) out.push(...keyFiles(p, depth + 1));
  }
  return out;
}

// A secret of 32 random bytes in a file of the data directory, created once with mode 600; never printed.
function secretFile(path: string, log: (m: string) => void, what: string): string {
  if (!existsSync(path)) {
    writeFileSync(path, randomBytes(32).toString('base64url') + '\n', {mode: 0o600, flag: 'wx'});
    log(`${what} created in ${path} (mode 600)`);
  }
  if ((statSync(path).mode & 0o077) !== 0) chmodSync(path, 0o600);
  const value = readFileSync(path, 'utf8').trim();
  if (value.length < 32) throw new Error(`${path} holds fewer than 32 characters; use a longer value`);
  return value;
}

class HttpError extends Error {
  retryAfter?: number;
  constructor(public status: number, public code: string, message: string, public details?: string[]) {
    super(message);
  }
}

const nowIso = () => new Date().toISOString();
const isObj = (x: unknown): x is Record<string, any> => typeof x === 'object' && x !== null && !Array.isArray(x);

function json(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {status, headers: {'Content-Type': 'application/json; charset=utf-8',
                                                               'Cache-Control': 'no-store', ...headers}});
}

function errorResponse(e: HttpError): Response {
  const error: Record<string, unknown> = {code: e.code, message: e.message};
  if (e.details) error.details = e.details;
  return json(e.status, {error}, e.retryAfter ? {'Retry-After': String(e.retryAfter)} : {});
}

// Fixed-window counter per key, in memory only. Only a hit adds an entry; expired entries are dropped at most once a
// second (or once a window, if shorter), so that requests from many addresses cost no time each.
export class RateLimiter {
  private hits = new Map<string, {n: number; reset: number}>();
  private pruned = 0;
  constructor(private limit: number, private windowMs: number) {}

  get size() { return this.hits.size; }

  private live(key: string, now: number) {
    const h = this.hits.get(key);
    return h && h.reset > now ? h : null;
  }

  // Seconds until the key is under its limit again; 0 when it is under it now.
  wait(key: string): number {
    const now = Date.now();
    const h = this.live(key, now);
    return !h || h.n < this.limit ? 0 : Math.max(1, Math.ceil((h.reset - now) / 1000));
  }

  hit(key: string) {
    const now = Date.now();
    if (this.hits.size > 10000 && now - this.pruned > Math.min(1000, this.windowMs)) {
      this.pruned = now;
      for (const [k, v] of this.hits) if (v.reset <= now) this.hits.delete(k);
    }
    let h = this.live(key, now);
    if (!h) this.hits.set(key, (h = {n: 0, reset: now + this.windowMs}));
    h.n++;
  }
}

function cookieValue(req: Request, name: string): string | null {
  for (const part of (req.headers.get('cookie') || '').split(';')) {
    const i = part.indexOf('=');
    if (i > 0 && part.slice(0, i).trim() === name) return part.slice(i + 1).trim();
  }
  return null;
}

// One "bytes=a-b", "bytes=a-" or "bytes=-n" range of a Blob; anything else gets the whole file (RFC 9110 lets a
// server ignore Range).
function rangeResponse(req: Request, blob: Blob, type: string, cache: string): Response {
  const size = blob.size;
  const headers: Record<string, string> = {'Content-Type': type, 'Accept-Ranges': 'bytes', 'Cache-Control': cache};
  const head = req.method === 'HEAD';
  const m = /^bytes=(\d*)-(\d*)$/.exec((req.headers.get('range') || '').trim());
  if (m && (m[1] !== '' || m[2] !== '')) {
    let start: number, end: number;
    if (m[1] === '') {
      start = Math.max(0, size - Number(m[2]));
      end = size - 1;
      if (Number(m[2]) === 0) start = size;
    } else {
      start = Number(m[1]);
      end = m[2] === '' ? size - 1 : Math.min(Number(m[2]), size - 1);
    }
    if (start >= size || start > end) {
      return new Response(null, {status: 416, headers: {...headers, 'Content-Range': `bytes */${size}`}});
    }
    return new Response(head ? null : blob.slice(start, end + 1), {status: 206, headers: {
      ...headers, 'Content-Range': `bytes ${start}-${end}/${size}`, 'Content-Length': String(end - start + 1)}});
  }
  return new Response(head ? null : blob, {status: 200, headers: {...headers, 'Content-Length': String(size)}});
}

export function createApp(opts: Options) {
  const dataDir = opts.dataDir;
  const log = opts.log ?? ((m: string) => console.log(m));
  const cookieSecure = opts.cookieSecure !== false;
  const trustProxy = opts.trustProxy !== false;
  const loginLimit = opts.loginLimit ?? 100, loginWindow = opts.loginWindowMs ?? 10 * 60 * 1000;
  const failedLogins = new RateLimiter(loginLimit, loginWindow);
  const allLogins = new RateLimiter(opts.loginTotalLimit ?? 10 * loginLimit, loginWindow);
  const registrations = new RateLimiter(opts.registerLimit ?? 30, loginWindow);   // all clients together
  const hosts = new Set([...LOOPBACK, ...(opts.hosts ?? [])].map((h) => h.trim().toLowerCase()).filter(Boolean));
  mkdirSync(dataDir, {recursive: true, mode: 0o700});
  chmodSync(dataDir, 0o700);   // also when the builder created it first
  const pkgDir = join(dataDir, 'package');
  const keys = keyFiles(pkgDir);
  if (keys.length) {
    throw new Error(`the package folder holds ${keys.join(', ')}: the key belongs in ${join(dataDir, 'private')}, ` +
                    'never next to the stimuli; the server does not start');
  }

  // ── the admin token and the server secret of the code HMAC: in the data directory, mode 600, never printed ──
  const tokenPath = join(dataDir, 'admin_token');
  const adminDigest = createHash('sha256').update(secretFile(tokenPath, log, 'admin token')).digest();
  const secret = secretFile(join(dataDir, 'server_secret'), log, 'server secret');
  const hmacOf = (code: string) => codeHmac(secret, code);
  const store = new Store(join(dataDir, 'mos.sqlite'));

  // ── the current package (or the sample) ──
  const sampleAudio = new Map<string, Uint8Array>();
  let current: {id: string; sample: boolean; imported_at: string; data: LtData} | null = null;
  function loadCurrent() {
    const id = store.get('current_package');
    current = id ? store.getPackage(id) : null;
    if (current && current.sample && !opts.sample) {
      store.set('current_package', null);
      current = null;
    }
    if (!current && opts.sample) {
      const data = loadSampleData();
      const sid = 'sample-' + sha256(JSON.stringify(data)).slice(0, 12);
      store.addPackage(sid, data, true, nowIso());
      store.set('current_package', sid);
      current = store.getPackage(sid);
    }
  }
  loadCurrent();

  // ── the daily copy of the database (addendum §11.5) ──
  const backupDir = join(dataDir, 'backup');
  function dailyBackup() {
    try {
      const f = join(backupDir, `mos-${nowIso().slice(0, 10).replace(/-/g, '')}.sqlite`);
      if (existsSync(f)) return;
      mkdirSync(backupDir, {recursive: true, mode: 0o700});
      rmSync(f + '.tmp', {force: true});
      store.backupTo(f + '.tmp');
      chmodSync(f + '.tmp', 0o600);
      renameSync(f + '.tmp', f);
      log(`database copied to ${f}`);
    } catch (e) {
      log(`backup failed: ${(e as Error).message}`);
    }
  }
  let backupTimer: ReturnType<typeof setInterval> | null = null;
  if (opts.backup !== false) {
    dailyBackup();
    backupTimer = setInterval(dailyBackup, 36e5);
    backupTimer.unref?.();
  }

  // A rater's rating session in the current package: stored from the consent on, null before it. Nothing is built
  // or kept in memory for a rater who has not consented (the consent builds the session and stores it).
  function sessionOf(raterId: string) {
    const pkg = requirePackage();
    return {pkg, meta: store.session(raterId, pkg.id)};
  }

  const consentPath = join(dataDir, CONSENT_FILE);
  // The consent text raters see: $MOS_DATA_DIR/consent_v1.html, unless it holds a placeholder or a script (then the
  // test is not open); the journal's sample text in sample mode.
  function consentText(): {html: string; sha256: string} | null {
    if (existsSync(consentPath)) {
      const html = readFileSync(consentPath, 'utf8');
      return consentProblems(html).length ? null : {html, sha256: sha256(html)};
    }
    if (!current?.sample) return null;
    const html = sampleConsentHtml();
    return {html, sha256: sha256(html)};
  }
  const isOpen = () => !!current && consentText() !== null;
  const openRegistration = () => store.get('open_registration') !== '0';   // on unless switched off
  // The cap on the codes issued by both routes together, every code ever issued counting (addendum §7.1).
  const maxCodes = () => Number(store.get('max_codes') ?? MAX_CODES);
  const codesLeft = () => Math.max(0, maxCodes() - store.codesIssued());
  const registrationOpen = () => openRegistration() && codesLeft() > 0;

  // ── request helpers ──
  function clientIp(req: Request, server?: Server): string {
    const cf = trustProxy ? req.headers.get('cf-connecting-ip') : null;
    return ipKey(cf ? cf.slice(0, 100) : server?.requestIP(req)?.address ?? 'local');
  }

  // The body is read as a stream and dropped as soon as it passes the limit: Bun's maxRequestBodySize checks only
  // Content-Length, so a chunked body would otherwise be read whole into memory.
  async function body(req: Request, limit = BODY_LIMIT): Promise<Record<string, any>> {
    const tooLarge = () => new HttpError(413, 'too_large', `the body is larger than ${limit} bytes`);
    if (Number(req.headers.get('content-length') || 0) > limit) throw tooLarge();
    if (!(req.headers.get('content-type') || '').toLowerCase().startsWith('application/json')) {
      throw new HttpError(415, 'unsupported_media_type', 'the body must be application/json');
    }
    const chunks: Uint8Array[] = [];
    if (req.body) {
      const reader = req.body.getReader();
      let n = 0;
      for (;;) {
        let r: ReadableStreamReadResult<Uint8Array>;
        try {
          r = await reader.read();
        } catch {
          throw new HttpError(400, 'bad_body', 'the body could not be read');
        }
        if (r.done) break;
        n += r.value.byteLength;
        if (n > limit) {
          reader.cancel().catch(() => {});
          throw tooLarge();
        }
        chunks.push(r.value);
      }
    }
    let x: unknown;
    try {
      x = JSON.parse(Buffer.concat(chunks).toString('utf8') || 'null');
    } catch {
      throw new HttpError(400, 'bad_json', 'the body is not valid JSON');
    }
    if (!isObj(x)) throw new HttpError(400, 'bad_request', 'the body must be a JSON object');
    return x;
  }

  function onlyKeys(b: Record<string, any>, allowed: string[]) {
    const extra = Object.keys(b).filter((k) => !allowed.includes(k));
    if (extra.length) throw new HttpError(400, 'bad_request', `unknown fields: ${extra.join(', ')}`);
  }

  // Cross-site requests: a POST from another origin is refused (the cookie is SameSite=Strict as well).
  function checkOrigin(req: Request) {
    const origin = req.headers.get('origin');
    if (!origin) return;
    let host = '';
    try { host = new URL(origin).host; } catch { /* "null" or garbage */ }
    if (!host || host !== req.headers.get('host')) throw new HttpError(403, 'bad_origin', 'cross-origin request refused');
  }

  // A browser-session cookie (no Max-Age), so that a closed browser on a shared computer does not stay signed in;
  // the code resumes the test on any device.
  function setCookie(token: string): string {
    return `${COOKIE}=${token}; Path=/; HttpOnly; SameSite=Strict${cookieSecure ? '; Secure' : ''}`;
  }

  function clearCookie(): string {
    return `${COOKIE}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0${cookieSecure ? '; Secure' : ''}`;
  }

  // 429 while the IP has too many failed logins, or too many logins and registrations in all; for a registration
  // also while all clients together have registered too often.
  function checkLoginLimit(ip: string, register = false) {
    const wait = Math.max(failedLogins.wait(ip), allLogins.wait(ip), register ? registrations.wait('all') : 0);
    if (!wait) return;
    const e = new HttpError(429, 'rate_limited', 'too many attempts; try again in a few minutes');
    e.retryAfter = wait;
    throw e;
  }

  // extra: the new code of an open registration, the one time it is shown.
  function login(raterId: string, extra: Record<string, unknown> = {}): Response {
    const token = randomBytes(32).toString('base64url');
    const now = nowIso();
    store.addAuth(sha256(token), raterId, now, new Date(Date.now() + SESSION_HOURS * 36e5).toISOString());
    store.touch(raterId, now);
    return json(200, {...stateFor(raterId), ...extra}, {'Set-Cookie': setCookie(token)});
  }

  function raterOf(req: Request): string | null {
    const token = cookieValue(req, COOKIE);
    if (!token || token.length > 100) return null;
    const id = store.authRater(sha256(token), nowIso());
    if (id) store.touch(id, nowIso());
    return id;
  }

  function requireRater(req: Request): string {
    const id = raterOf(req);
    if (!id) throw new HttpError(401, 'unauthorized', 'not signed in');
    return id;
  }

  function requirePackage() {
    if (!current) throw new HttpError(503, 'not_open', 'the listening test is not open');
    return current;
  }

  // The rater's results; before the consent an empty skeleton with the rater's half (no trials), of which nothing
  // is stored.
  function rowOf(raterId: string): {row: SessionRow; meta: SessionMeta | null} {
    const {pkg, meta} = sessionOf(raterId);
    if (meta) return {row: meta, meta};
    const r = store.rater(raterId)!;
    const cfg = pkg.data.config;
    const session: Session = {raterId, parity: r.half === 'A' ? 0 : 1, half: cfg.halves ? r.half : null,
                              config_title: cfg.title, practice: [], mos: [], smos: [], cmos: []};
    return {meta: null, row: {rater_id: raterId, route: r.route, package_id: pkg.id, session, browser: null,
                              consent_at: null, consent_lang: null, consent_sha: null,
                              headphone: {attempts: [], passed: false}, background: null, finished: null,
                              completed_at: null}};
  }

  function stateFor(raterId: string) {
    if (!current) return {auth: true, package_id: null, open: false};
    const {row, meta} = rowOf(raterId);
    return {auth: true, package_id: current.id, sample: isTestData(current), consented: !!meta,
            completed: !!meta?.completed_at, results: resultsFor(row, meta ? store.answers(raterId, current.id) : [])};
  }

  function requireAdmin(req: Request) {
    const m = /^Bearer (\S{1,200})$/.exec(req.headers.get('authorization') || '');
    const given = createHash('sha256').update(m ? m[1] : '').digest();
    if (!m || !timingSafeEqual(given, adminDigest)) throw new HttpError(401, 'unauthorized', 'admin token required');
  }

  const hpConfig = () => requirePackage().data.config.headphone_check;

  // ── rater API ──
  async function raterApi(req: Request, path: string, server?: Server): Promise<Response> {
    const get = req.method === 'GET';
    if (get && path === '/api/info') {
      return json(200, {open: isOpen(), sample: !!current && isTestData(current), open_registration: registrationOpen(),
                        contact: store.get('contact') || null, languages: LANGS});
    }
    if (get && path === '/api/consent') {
      const c = current ? consentText() : null;
      if (!c) throw new HttpError(404, 'no_consent', 'no consent text is set');
      return json(200, c);
    }
    if (get && path === '/api/state') {
      const id = raterOf(req);
      return json(200, id ? stateFor(id) : {auth: false});
    }
    if (get && path === '/api/plan') {
      const id = requireRater(req);
      const {pkg, meta} = sessionOf(id);
      if (!meta) throw new HttpError(409, 'consent_required', 'consent is not recorded');
      return json(200, planFor(pkg, meta.session));
    }
    if (req.method !== 'POST') throw new HttpError(404, 'not_found', 'no such endpoint');
    checkOrigin(req);
    if (path === '/api/login') {
      const b = await body(req);
      const ip = clientIp(req, server);
      checkLoginLimit(ip);
      onlyKeys(b, ['code']);
      allLogins.hit(ip);
      const code = normalizeCode(b.code);
      const r = code ? store.raterByHmac(hmacOf(code)) : null;
      if (!r || r.disabled) {
        failedLogins.hit(ip);
        throw new HttpError(401, 'invalid_code', 'this code is not valid');
      }
      return login(r.rater_id);
    }
    if (path === '/api/register') {
      onlyKeys(await body(req), []);
      if (!openRegistration()) throw new HttpError(403, 'closed', 'registration is closed');
      const ip = clientIp(req, server);
      checkLoginLimit(ip, true);
      if (!isOpen()) throw new HttpError(503, 'not_open', 'the listening test is not open');
      if (!codesLeft()) throw new HttpError(403, 'closed', 'registration is closed: every code has been issued');
      allLogins.hit(ip);
      registrations.hit('all');
      const [x] = createRaters(1, '', 'open', null);
      return login(x.rater_id, {code: x.code});
    }
    if (path === '/api/logout') {
      onlyKeys(await body(req), []);
      const token = cookieValue(req, COOKIE);
      if (token) store.dropAuth(sha256(token));
      return json(200, {ok: true}, {'Set-Cookie': clearCookie()});
    }
    const id = requireRater(req);
    const b = await body(req);
    const {pkg, meta} = sessionOf(id);
    if (path === '/api/consent') {
      onlyKeys(b, ['lang', 'sha256']);
      if (!LANGS.includes(b.lang)) throw new HttpError(400, 'bad_request', 'lang must be vi or en');
      const c = consentText();
      if (!c) throw new HttpError(404, 'no_consent', 'no consent text is set');
      if (b.sha256 !== c.sha256) throw new HttpError(409, 'consent_changed', 'the consent text has changed; reload');
      // the rating session starts here: before it only the use of the code is stored
      if (!meta) {
        store.addSession(id, pkg.id, makeSession(pkg.data, id, store.rater(id)!.half),
                         browserFamily(req.headers.get('user-agent')), b.lang, b.sha256, nowIso());
      }
      return json(200, {ok: true, duplicate: !!meta});
    }
    if (!meta) throw new HttpError(409, 'consent_required', 'consent is not recorded');
    const session = meta.session;
    if (path === '/api/headphone') return headphone(id, pkg.id, meta, b);
    if (path === '/api/answer') return answer(id, pkg.id, meta, b);
    if (path === '/api/background') {
      onlyKeys(b, C.BACKGROUND.map((q: any) => q.key));
      const missing = C.checkBackground(b);
      if (missing.length) throw new HttpError(400, 'bad_request', `missing or unknown answers: ${missing.join(', ')}`);
      if (nextTrial(session, store.answerCounts(id, pkg.id))) {
        throw new HttpError(409, 'not_finished', 'the background questions come after the last rating');
      }
      if (!meta.background) store.setBackground(id, pkg.id, Object.fromEntries(C.BACKGROUND.map((q: any) => [q.key, b[q.key]])));
      return json(200, {ok: true, duplicate: !!meta.background});
    }
    if (path === '/api/finish') {
      onlyKeys(b, []);
      const failed = hpFailed(meta.headphone, hpConfig().attempts);
      const done = !nextTrial(session, store.answerCounts(id, pkg.id)) && !!meta.background;
      if (!failed && !done) throw new HttpError(409, 'not_finished', 'the session is not finished');
      store.setCompleted(id, pkg.id, nowIso());
      store.dropAuthOf(id);   // signed out everywhere: the next person at a shared computer starts afresh
      return json(200, {ok: true}, {'Set-Cookie': clearCookie()});
    }
    throw new HttpError(404, 'not_found', 'no such endpoint');
  }

  // One headphone-check trial: {attempt, trial, choice}. The tones of each trial follow from the rater ID (lt_core),
  // so the server records the order and the right answer itself; a repeat of a recorded trial changes nothing.
  function headphone(id: string, pkgId: string, s: SessionMeta, b: Record<string, any>): Response {
    onlyKeys(b, ['attempt', 'trial', 'choice']);
    const hpc = hpConfig();
    if (!intIn(b.attempt, 0, hpc.attempts - 1) || !intIn(b.trial, 0, hpc.trials - 1) || !intIn(b.choice, 0, 2)) {
      throw new HttpError(400, 'bad_request', 'attempt, trial and choice must be integers in range');
    }
    const hp: Headphone = s.headphone;
    const a = hp.attempts[b.attempt];
    if (a && b.trial < a.trials.length) return json(200, {ok: true, duplicate: true, headphone: hp});
    const last = hp.attempts[hp.attempts.length - 1];
    const opening = b.attempt === hp.attempts.length && !hp.passed && (!last || last.correct !== null) && b.trial === 0;
    const continuing = a && a.correct === null && b.trial === a.trials.length;
    if (!opening && !continuing) throw new HttpError(409, 'out_of_order', 'not the next headphone trial');
    const att = a ?? {trials: [], correct: null};
    if (!a) hp.attempts.push(att);
    const t = C.headphoneTrials(hpc.trials, C.rng(s.session.raterId + ':headphone:' + b.attempt))[b.trial];
    att.trials.push({order: t.order, answer: t.answer, choice: b.choice});
    if (att.trials.length >= hpc.trials) {
      att.correct = att.trials.filter((x) => x.choice === x.answer).length;
      hp.passed = att.correct >= hpc.pass;
    }
    store.setHeadphone(id, pkgId, hp);
    return json(200, {ok: true, headphone: hp});
  }

  // One rating: {tid, block, value, lang, plays, played_s, rt_ms, t_ms}. Only the rater's next trial is taken; a
  // repeat of an answered trial is acknowledged and changes nothing (the first answer stays). lang is the page
  // language at the rating (addendum §5.4). When the package asks for full plays, every clip must have been played.
  function answer(id: string, pkgId: string, s: SessionMeta, b: Record<string, any>): Response {
    onlyKeys(b, ['tid', 'block', 'value', 'lang', 'plays', 'played_s', 'rt_ms', 't_ms']);
    if (!(BLOCKS as readonly string[]).includes(b.block)) throw new HttpError(400, 'bad_request', 'unknown block');
    const block = b.block as Block;
    if (typeof b.tid !== 'string' || !TID_RE.test(b.tid)) throw new HttpError(400, 'bad_request', 'bad tid');
    const pos = s.session[block].findIndex((t) => t.tid === b.tid);
    if (pos < 0) throw new HttpError(400, 'unknown_tid', `tid ${b.tid} is not a ${block} trial of this rater`);
    if (!intIn(b.value, ...RANGE[block])) {
      throw new HttpError(400, 'bad_request', `value must be an integer in ${RANGE[block].join('..')}`);
    }
    if (!LANGS.includes(b.lang)) throw new HttpError(400, 'bad_request', 'lang must be vi or en');
    const roles = PLAY_ROLES[block];
    const okPlays = (x: unknown) => intIn(x, 0, 1000);
    let plays: unknown = b.plays;
    if (roles) {
      if (!isObj(plays) || Object.keys(plays).sort().join() !== roles.slice().sort().join() ||
          !roles.every((r) => okPlays((plays as any)[r]))) {
        throw new HttpError(400, 'bad_request', `plays must be {${roles.join(', ')}} with integers 0-1000`);
      }
      plays = Object.fromEntries(roles.map((r) => [r, (b.plays as any)[r]]));
    } else if (!okPlays(plays)) throw new HttpError(400, 'bad_request', 'plays must be an integer 0-1000');
    if (requirePackage().data.config.require_full_play && (roles ? roles.some((r) => (plays as any)[r] < 1) : plays === 0)) {
      throw new HttpError(400, 'not_played', 'every clip of the trial must be played before it is rated');
    }
    const num = (x: unknown, hi: number) => x === undefined || x === null || (typeof x === 'number' && x >= 0 && x <= hi);
    if (!num(b.played_s, 36000)) throw new HttpError(400, 'bad_request', 'played_s must be 0-36000 seconds');
    for (const k of ['rt_ms', 't_ms']) {
      if (!(b[k] === undefined || b[k] === null || intIn(b[k], 0, Number.MAX_SAFE_INTEGER))) {
        throw new HttpError(400, 'bad_request', `${k} must be a non-negative integer`);
      }
    }
    const capMs = (x: number | null | undefined) => (x == null ? null : Math.min(x, MAX_MS));   // a trial left open for days
    const prev = store.answer(id, pkgId, b.tid);
    if (prev) return json(200, {ok: true, duplicate: true});
    if (!s.headphone.passed) throw new HttpError(409, 'headphone_required', 'the sound check is not passed');
    const next = nextTrial(s.session, store.answerCounts(id, pkgId));
    if (!next || next.block !== block || next.index !== pos) {
      throw new HttpError(409, 'out_of_order', 'not the next trial of this rater');
    }
    const counts = store.answerCounts(id, pkgId);
    counts[block] = (counts[block] ?? 0) + 1;
    const last = nextTrial(s.session, counts) === null;
    store.addAnswer(id, pkgId, {block, pos, tid: b.tid, rating: b.value, lang: b.lang, plays,
                                played_s: b.played_s == null ? null : Math.round(b.played_s * 100) / 100,
                                rt_ms: capMs(b.rt_ms), t_ms: capMs(b.t_ms), t: nowIso()}, last);
    return json(200, {ok: true, finished: last});
  }

  // ── admin ──
  // New raters with codes. Each gets the half with fewer consented raters who can still complete in the current
  // package (disabled raters and raters who failed the sound check not counted), counting the codes of this call as
  // if they had consented; on a tie, the half with fewer codes issued (not disabled) (addendum §6.2, §7.1). Decided
  // by those counts only, never by a rating. The codes are returned once and kept only as their HMAC. The caller
  // checks the cap on the codes.
  function createRaters(n: number, prefix: string, route: 'invite' | 'open', note: string | null) {
    const {issued, sessions} = store.halfCounts(current?.id ?? null);
    const consented = {A: 0, B: 0};
    const attempts = current?.data.config.headphone_check.attempts ?? 0;
    for (const s of sessions) if (!hpFailed(s.headphone, attempts)) consented[s.half]++;
    const out: {code: string; rater_id: string; half: Half}[] = [];
    while (out.length < n) {
      const half: Half = consented.A !== consented.B ? (consented.A < consented.B ? 'A' : 'B')
        : issued.A <= issued.B ? 'A' : 'B';
      const code = randomCode(prefix);
      const id = store.addRater(hmacOf(code), route, half, note, nowIso());
      if (!id) continue;
      consented[half]++;
      issued[half]++;
      out.push({code, rater_id: id, half});
    }
    return out;
  }

  // The rater of {rater_id} or {code} in an admin request; codes only ever travel in POST bodies.
  function raterFrom(b: Record<string, any>) {
    const byCode = b.code !== undefined;
    if (byCode === (b.rater_id !== undefined)) throw new HttpError(400, 'bad_request', 'give rater_id or code');
    if (!byCode && (typeof b.rater_id !== 'string' || !RATER_ID_RE.test(b.rater_id))) {
      throw new HttpError(400, 'bad_request', 'rater_id: R001, R002, ...');
    }
    if (byCode && typeof b.code !== 'string') throw new HttpError(400, 'bad_request', 'code: the access code');
    const code = byCode ? normalizeCode(b.code) : null;
    const r = byCode ? (code ? store.raterByHmac(hmacOf(code)) : null) : store.rater(b.rater_id);
    if (!r) throw new HttpError(404, 'not_found', 'no such rater');
    return r;
  }

  // A withdrawn rater leaves the daily copies of the database too. A copy that cannot be cleaned is deleted.
  function pruneBackups(raterId: string): number {
    if (!existsSync(backupDir)) return 0;
    let n = 0;
    for (const f of readdirSync(backupDir).filter((x) => x.endsWith('.sqlite')).map((x) => join(backupDir, x))) {
      try {
        if (scrubCopy(f, raterId)) n++;
      } catch (e) {
        rmSync(f, {force: true});
        log(`backup ${f} could not be cleaned and was deleted: ${(e as Error).message}`);
        n++;
      }
    }
    return n;
  }

  // A withdrawn rater leaves the export folders too: their results file and their rows of answers.csv.
  function pruneExports(raterId: string): number {
    const root = join(dataDir, 'export');
    if (!existsSync(root)) return 0;
    let n = 0;
    for (const d of readdirSync(root, {withFileTypes: true})) {
      if (!d.isDirectory()) continue;
      const f = join(root, d.name, 'results', `listening_${raterId}.json`);
      if (existsSync(f)) {
        rmSync(f);
        n++;
      }
      const csv = join(root, d.name, 'answers.csv');
      if (!existsSync(csv)) continue;
      const lines = readFileSync(csv, 'utf8').split('\n');
      const kept = lines.filter((l, i) => i === 0 || l.split(',', 3)[1] !== raterId);
      if (kept.length === lines.length) continue;
      writeFileSync(csv + '.tmp', kept.join('\n'), {mode: 0o600});
      renameSync(csv + '.tmp', csv);
      n++;
    }
    return n;
  }

  function importPackage(force: boolean) {
    const jsonPath = join(pkgDir, 'lt_data.json'), jsPath = join(pkgDir, 'lt_data.js');
    const path = existsSync(jsonPath) ? jsonPath : existsSync(jsPath) ? jsPath : null;
    if (!path) throw new HttpError(400, 'no_package', `no lt_data.json or lt_data.js in ${pkgDir}`);
    const keyLike = [...readdirSync(pkgDir).filter((f) => /key/i.test(f)).map((f) => join(pkgDir, f)), ...keyFiles(pkgDir)];
    if (keyLike.length) {
      throw new HttpError(400, 'key_in_package', `the package folder holds ${[...new Set(keyLike)].join(', ')}: the ` +
                          `key belongs in ${join(dataDir, 'private')}, never next to the stimuli`);
    }
    let raw: unknown;
    try {
      raw = parseLtFile(readFileSync(path, 'utf8'), path === jsPath);
    } catch (e) {
      throw new HttpError(400, 'bad_package', `${path} cannot be read: ${(e as Error).message}`);
    }
    const {data, errors} = sanitizeLtData(raw);
    if (!data) throw new HttpError(400, 'bad_package', `${errors.length} problems in ${path}`, errors.slice(0, 30));
    const hashes: string[] = [];
    const problems: string[] = [];
    for (const sid of Object.keys(data.stimuli).sort()) {
      const f = join(pkgDir, 'stimuli', `${sid}.${extOf(data, sid)}`);
      let st;
      try { st = lstatSync(f); } catch { problems.push(`missing ${f}`); continue; }
      if (!st.isFile() || st.size === 0) { problems.push(`${f} is not a non-empty regular file`); continue; }
      hashes.push(sid + ':' + sha256(readFileSync(f)));
    }
    if (problems.length) throw new HttpError(400, 'bad_package', `${problems.length} stimulus files`, problems.slice(0, 30));
    const id = sha256(JSON.stringify(data) + '\n' + hashes.join('\n')).slice(0, 16);
    if (current && current.id === id) return {ok: true, package_id: id, unchanged: true};
    if (current && !current.sample && store.answerCount(current.id) > 0 && !force) {
      throw new HttpError(409, 'answers_exist', `package ${current.id} already has answers; import with force to ` +
                          'replace it (its answers stay in the database and can still be exported)');
    }
    store.addPackage(id, data, false, nowIso());
    store.set('current_package', id);
    loadCurrent();
    const counts = Object.fromEntries(BLOCKS.map((b) => [b, data[b].length]));
    log(`package ${id} imported (${Object.keys(data.stimuli).length} stimuli)`);
    return {ok: true, package_id: id, stimuli: Object.keys(data.stimuli).length, trials: counts};
  }

  function writeConsent(html: unknown) {
    if (typeof html !== 'string' || !html.trim()) throw new HttpError(400, 'bad_request', 'the consent text is empty');
    const problems = consentProblems(html);
    if (problems.length) throw new HttpError(400, 'bad_consent', problems.join('; '));
    writeFileSync(consentPath + '.tmp', html.trim() + '\n', {mode: 0o600});
    renameSync(consentPath + '.tmp', consentPath);
    return sha256(readFileSync(consentPath, 'utf8'));
  }

  function packageFor(req: Request) {
    const id = new URL(req.url).searchParams.get('package');
    const pkg = id ? store.getPackage(id) : current;
    if (!pkg) throw new HttpError(404, 'no_package', 'no such package');
    return pkg;
  }

  function exportedSessions(pkgId: string, completedOnly: boolean) {
    return store.sessions(pkgId).filter((s) => !completedOnly || s.completed_at);
  }

  function progress(s: SessionMeta | null) {
    if (!s) return null;
    const counts = store.answerCounts(s.rater_id, s.package_id);
    return Object.fromEntries(BLOCKS.map((b) => [b, {answered: counts[b] ?? 0, total: s.session[b].length}]));
  }

  async function adminApi(req: Request, path: string): Promise<Response> {
    requireAdmin(req);
    const get = req.method === 'GET';
    if (!get) checkOrigin(req);
    if (get && path === '/api/admin/status') {
      const raters = store.raters();
      const html = existsSync(consentPath) ? readFileSync(consentPath, 'utf8') : null;
      return json(200, {
        data_dir: dataDir, open: isOpen(), open_registration: openRegistration(), contact: store.get('contact') || '',
        codes: {issued: store.codesIssued(), max: maxCodes(), left: codesLeft()},
        consent: {file: consentPath, set: html !== null, sha256: html === null ? null : sha256(html),
                  problems: html === null ? [] : consentProblems(html)},
        raters: {total: raters.length, disabled: raters.filter((r) => r.disabled).length,
                 invite: raters.filter((r) => r.route === 'invite').length,
                 open: raters.filter((r) => r.route === 'open').length},
        package: current ? {id: current.id, sample: current.sample, dry_run: current.data.dry_run === true,
                            imported_at: current.imported_at,
                            halves: current.data.config.halves, stimuli: Object.keys(current.data.stimuli).length,
                            trials: Object.fromEntries(BLOCKS.map((b) => [b, current!.data[b].length]))} : null,
      });
    }
    if (get && path === '/api/admin/raters') {
      const pkg = new URL(req.url).searchParams.get('package') ? packageFor(req) : current;
      const failedHp = (hp: Headphone) => hpFailed(hp, pkg!.data.config.headphone_check.attempts);
      const sessions = new Map((pkg ? store.sessions(pkg.id) : []).map((s) => [s.rater_id, s]));
      const out = store.raters().map((r) => {
        const s = sessions.get(r.rater_id) ?? null;
        const hp = s?.headphone;
        return {rater_id: r.rater_id, route: r.route, half: r.half, note: r.note, created_at: r.created_at,
                disabled: !!r.disabled, first_used: r.first_used, last_seen: r.last_seen,
                consent_at: s?.consent_at ?? null, consent_lang: s?.consent_lang ?? null, browser: s?.browser ?? null,
                headphone: hp ? {attempts: hp.attempts.length, passed: hp.passed, failed: failedHp(hp)} : null,
                progress: progress(s), finished: s?.finished ?? null, completed_at: s?.completed_at ?? null};
      });
      // progress per half: codes issued (not disabled), consented, every rating given, completed
      const halves = Object.fromEntries(['A', 'B'].map((h) => {
        const mine = out.filter((x) => x.half === h && !x.disabled);
        return [h, {issued: mine.length, consented: mine.filter((x) => x.consent_at).length,
                    finished: mine.filter((x) => x.finished).length,
                    completed: mine.filter((x) => x.completed_at && x.finished).length,
                    headphone_failed: mine.filter((x) => x.headphone?.failed).length}];
      }));
      return json(200, {package_id: pkg?.id ?? null, halves, raters: out});
    }
    if (get && path === '/api/admin/consent') {
      const html = existsSync(consentPath) ? readFileSync(consentPath, 'utf8') : null;
      return json(200, {html, sha256: html === null ? null : sha256(html)});
    }
    if (get && path === '/api/admin/export/rater') {
      const pkg = packageFor(req);
      const rid = new URL(req.url).searchParams.get('rater') || '';
      const s = RATER_ID_RE.test(rid) ? store.session(rid, pkg.id) : null;
      if (!s) throw new HttpError(404, 'not_found', 'no session of this rater in the package');
      return json(200, resultsFor(s, store.answers(s.rater_id, pkg.id)),
                  {'Content-Disposition': `attachment; filename="listening_${s.rater_id}.json"`});
    }
    if (get && path === '/api/admin/export/answers.csv') {
      const pkg = packageFor(req);
      const only = new URL(req.url).searchParams.get('completed_only') === '1';
      const lines = [CSV_COLUMNS.join(',')];
      for (const s of exportedSessions(pkg.id, only)) lines.push(...csvRows(pkg.id, s, store.answers(s.rater_id, pkg.id)));
      return new Response(lines.join('\n') + '\n', {headers: {
        'Content-Type': 'text/csv; charset=utf-8', 'Cache-Control': 'no-store',
        'Content-Disposition': `attachment; filename="answers_${pkg.id}.csv"`}});
    }
    if (get) throw new HttpError(404, 'not_found', 'no such endpoint');
    const b = await body(req, path === '/api/admin/consent' ? CONSENT_LIMIT : BODY_LIMIT);
    if (path === '/api/admin/codes') {
      onlyKeys(b, ['n', 'prefix', 'note']);
      if (!intIn(b.n, 1, 500)) throw new HttpError(400, 'bad_request', 'n must be 1-500');
      const prefix = typeof b.prefix === 'string' ? b.prefix.trim().toUpperCase() : '';
      if (prefix && !PREFIX_RE.test(prefix)) throw new HttpError(400, 'bad_request', 'prefix: 1-8 letters or digits');
      const note = typeof b.note === 'string' && b.note.trim() ? b.note.trim().slice(0, 200) : null;
      if (b.n > codesLeft()) {
        throw new HttpError(409, 'code_cap', `the cap of ${maxCodes()} codes (both routes) allows ${codesLeft()} more; ` +
                            'raise it in the settings to issue more');
      }
      return json(200, {halves: !!current?.data.config.halves, codes: createRaters(b.n, prefix, 'invite', note)});
    }
    if (path === '/api/admin/raters/disable') {
      onlyKeys(b, ['rater_id', 'disabled']);
      if (typeof b.disabled !== 'boolean') throw new HttpError(400, 'bad_request', 'disabled: boolean');
      const r = raterFrom({rater_id: b.rater_id});
      store.setDisabled(r.rater_id, b.disabled);
      return json(200, {ok: true, rater_id: r.rater_id, disabled: b.disabled});
    }
    if (path === '/api/admin/raters/find') {
      onlyKeys(b, ['code']);
      const r = raterFrom({code: b.code ?? 0});
      return json(200, {rater_id: r.rater_id, route: r.route, half: r.half, disabled: !!r.disabled});
    }
    if (path === '/api/admin/raters/delete') {
      onlyKeys(b, ['rater_id', 'code']);
      const r = raterFrom(b);
      const {walCleared} = store.deleteRater(r.rater_id);
      const files = pruneExports(r.rater_id);
      const backups = pruneBackups(r.rater_id);
      log(`rater ${r.rater_id} deleted (${files} export files and ${backups} backups changed` +
          `${walCleared ? '' : '; the write-ahead log could not be emptied yet'})`);
      return json(200, {ok: true, rater_id: r.rater_id, export_files: files, backup_files: backups, wal_cleared: walCleared});
    }
    if (path === '/api/admin/package/import') {
      onlyKeys(b, ['force']);
      return json(200, importPackage(b.force === true));
    }
    if (path === '/api/admin/consent') {
      onlyKeys(b, ['html']);
      return json(200, {ok: true, sha256: writeConsent(b.html)});
    }
    if (path === '/api/admin/settings') {
      onlyKeys(b, ['open_registration', 'contact', 'max_codes']);
      if (b.open_registration !== undefined && typeof b.open_registration !== 'boolean') {
        throw new HttpError(400, 'bad_request', 'open_registration: boolean');
      }
      if (b.contact !== undefined && (typeof b.contact !== 'string' || b.contact.length > 200)) {
        throw new HttpError(400, 'bad_request', 'contact: text up to 200 characters');
      }
      if (b.max_codes !== undefined && !intIn(b.max_codes, 0, 10000)) {
        throw new HttpError(400, 'bad_request', 'max_codes: an integer 0-10000');
      }
      if (b.open_registration !== undefined) store.set('open_registration', b.open_registration ? '1' : '0');
      if (b.contact !== undefined) store.set('contact', b.contact.trim() || null);
      if (b.max_codes !== undefined && b.max_codes !== maxCodes()) {
        log(`code cap changed from ${maxCodes()} to ${b.max_codes} (${store.codesIssued()} issued)`);
        store.set('max_codes', String(b.max_codes));
      }
      return json(200, {ok: true, open_registration: openRegistration(), contact: store.get('contact'),
                        max_codes: maxCodes()});
    }
    if (path === '/api/admin/export') {
      onlyKeys(b, ['package', 'completed_only']);
      const pkg = b.package ? store.getPackage(String(b.package)) : current;
      if (!pkg) throw new HttpError(404, 'no_package', 'no such package');
      const stamp = nowIso().replace(/[-:]/g, '').replace(/\..*/, '').replace('T', '-');
      const dir = join(dataDir, 'export', `${pkg.id}_${stamp}`);
      mkdirSync(join(dir, 'results'), {recursive: true, mode: 0o700});
      const lines = [CSV_COLUMNS.join(',')];
      const sessions = exportedSessions(pkg.id, b.completed_only === true);
      for (const s of sessions) {
        const answers = store.answers(s.rater_id, pkg.id);
        writeFileSync(join(dir, 'results', `listening_${s.rater_id}.json`), JSON.stringify(resultsFor(s, answers), null, 2) + '\n',
                      {mode: 0o600});
        lines.push(...csvRows(pkg.id, s, answers));
      }
      writeFileSync(join(dir, 'answers.csv'), lines.join('\n') + '\n', {mode: 0o600});
      return json(200, {ok: true, dir, raters: sessions.length, package_id: pkg.id, sample: pkg.sample});
    }
    throw new HttpError(404, 'not_found', 'no such endpoint');
  }

  // ── stimuli ──
  function stimulus(req: Request, sid: string, ext: string): Response {
    const id = requireRater(req);
    const pkg = requirePackage();
    const meta = pkg.data.stimuli[sid];
    if (!meta || extOf(pkg.data, sid) !== ext) throw new HttpError(404, 'not_found', 'no such stimulus');
    const s = sessionOf(id);
    if (!s.meta) throw new HttpError(409, 'consent_required', 'consent is not recorded');
    if (!sessionStimuli(s.meta.session).has(sid)) throw new HttpError(404, 'not_found', 'no such stimulus');
    const cache = 'private, max-age=31536000, immutable';
    if (pkg.sample) {
      let bytes = sampleAudio.get(sid);
      if (!bytes) sampleAudio.set(sid, (bytes = sampleWav(sid, meta.dur)));
      return rangeResponse(req, new Blob([bytes]), AUDIO_TYPES.wav, cache);
    }
    const f = join(pkgDir, 'stimuli', `${sid}.${ext}`);
    let st;
    try { st = lstatSync(f); } catch { throw new HttpError(404, 'not_found', 'no such stimulus'); }
    if (!st.isFile()) throw new HttpError(404, 'not_found', 'no such stimulus');
    return rangeResponse(req, Bun.file(f), AUDIO_TYPES[ext], cache);
  }

  // Only the configured host names and the loopback addresses are served: a page of another site whose name was
  // rebound to 127.0.0.1 (DNS rebinding) is refused, and the origin check can rely on the Host header.
  function checkHost(req: Request) {
    const m = /^(\[[0-9a-f:.]+\]|[^:\[\]]+)(?::\d{1,5})?$/i.exec((req.headers.get('host') || '').trim());
    if (!m || !hosts.has(m[1].toLowerCase())) throw new HttpError(421, 'bad_host', 'unknown host');
  }

  async function route(req: Request, server?: Server): Promise<Response> {
    checkHost(req);
    const url = new URL(req.url);
    const path = url.pathname;
    if (path.startsWith('/api/admin/')) return adminApi(req, path);
    if (path.startsWith('/api/')) return raterApi(req, path, server);
    if (req.method !== 'GET' && req.method !== 'HEAD') throw new HttpError(405, 'method_not_allowed', 'method not allowed');
    const m = /^\/s\/([^/]+)\.([a-z0-9]+)$/.exec(path);
    if (m) {
      if (!SID_RE.test(m[1]) || !AUDIO_TYPES[m[2]]) throw new HttpError(404, 'not_found', 'no such stimulus');
      return stimulus(req, m[1], m[2]);
    }
    const st = STATIC[path];
    if (!st) return new Response('Not found\n', {status: 404, headers: {'Content-Type': 'text/plain; charset=utf-8'}});
    const file = Bun.file(join(PUBLIC_DIR, st[0]));
    return new Response(req.method === 'HEAD' ? null : file, {headers: {'Content-Type': st[1], 'Cache-Control': 'no-cache'}});
  }

  async function fetch(req: Request, server?: Server): Promise<Response> {
    let res: Response;
    try {
      res = await route(req, server);
    } catch (e) {
      if (e instanceof HttpError) res = errorResponse(e);
      else {
        log(`error ${req.method} ${new URL(req.url).pathname}: ${(e as Error)?.stack || e}`);
        res = errorResponse(new HttpError(500, 'internal', 'internal error'));
      }
    }
    return withSecurityHeaders(res);
  }

  return {fetch, store, tokenPath, dataDir, close: () => {
    if (backupTimer) clearInterval(backupTimer);
    store.close();
  }};
}

export function withSecurityHeaders(res: Response): Response {
  for (const [k, v] of Object.entries(SECURITY_HEADERS)) res.headers.set(k, v);
  return res;
}

// Bun's last resort when a request fails outside the app's own error handling: a bare 500 with the same headers.
export const fallbackError = () =>
  withSecurityHeaders(new Response('internal error\n', {status: 500, headers: {'Content-Type': 'text/plain; charset=utf-8'}}));

export function startServer(opts: Options) {
  const app = createApp(opts);
  const server = Bun.serve({
    hostname: opts.hostname ?? '127.0.0.1',
    port: opts.port ?? 8790,
    maxRequestBodySize: CONSENT_LIMIT + 1024,
    fetch: (req, srv) => app.fetch(req, srv),
    error: fallbackError,
  });
  return {
    app, server,
    async stop() {
      await server.stop();
      app.close();
    },
  };
}
