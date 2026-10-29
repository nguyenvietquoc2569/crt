// Listening-test logic of the server: the stimulus package (LT_DATA of mvc/mos/build_stimuli.py), the per-rater
// session and plan, the journal page's results file, access codes and their HMAC, the browser family, the answers CSV
// and the synthetic sample audio. The seeded randomisation is public/lt_core.js (the journal page's, with the changes
// of mvc/LISTENING_ADDENDUM_mvc_v1.md §6), so the server builds the sessions and headphone trials the page shows.
import { createHash, createHmac, randomBytes } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { runInNewContext } from 'node:vm';

export const C = require('../public/lt_core.js');
export const PUBLIC_DIR = join(import.meta.dir, '..', 'public');

export const BLOCKS = ['practice', 'mos', 'smos', 'cmos'] as const;
export type Block = (typeof BLOCKS)[number];
export const RANGE: Record<Block, [number, number]> = {practice: [1, 5], mos: [1, 5], smos: [1, 5], cmos: [-3, 3]};
// The clips of a trial in LT_DATA, and in a rater's session (the CMOS x/y become a/b in the shown order).
const DATA_FIELDS: Record<Block, string[]> = {practice: ['stim'], mos: ['stim'], smos: ['stim', 'ref'], cmos: ['x', 'y']};
export const PLAN_FIELDS: Record<Block, string[]> = {practice: ['stim'], mos: ['stim'], smos: ['stim', 'ref'],
                                                     cmos: ['a', 'b']};
// The roles of the plays of a two-clip trial, as the page reports them.
export const PLAY_ROLES: Record<Block, string[] | null> = {practice: null, mos: null, smos: ['ref', 'stim'],
                                                           cmos: ['a', 'b']};
// Ids start with a letter or digit, so that no id is a spreadsheet formula ("-x", "=x") in answers.csv.
export const SID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{5,63}$/;
export const TID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
export const AUDIO_TYPES: Record<string, string> = {
  flac: 'audio/flac', wav: 'audio/wav', mp3: 'audio/mpeg', ogg: 'audio/ogg', opus: 'audio/ogg', m4a: 'audio/mp4',
};

export type Trial = {tid: string; [k: string]: unknown};
export const LANGS = ['vi', 'en'];
export type LtData = {
  version: 1;
  dry_run?: true;   // a build from synthetic inputs (mvc/mos/build_stimuli.py --dry_run): shown as TEST DATA
  config: {title: string | null; halves: boolean; require_full_play: boolean;
           headphone_check: {trials: number; pass: number; attempts: number}};
  stimuli: Record<string, {src: string; dur: number}>;
  practice: Trial[]; mos: Trial[]; smos: Trial[]; cmos: Trial[];
};
export type Session = {raterId: string; parity: number; half: string | null; config_title: string | null;
                       practice: Trial[]; mos: Trial[]; smos: Trial[]; cmos: Trial[]};
export type Half = 'A' | 'B';

const isObj = (x: unknown): x is Record<string, any> => typeof x === 'object' && x !== null && !Array.isArray(x);
export const intIn = (x: unknown, lo: number, hi: number): x is number =>
  Number.isInteger(x) && (x as number) >= lo && (x as number) <= hi;

export function sha256(data: string | Uint8Array): string {
  return createHash('sha256').update(data).digest('hex');
}

// ── stimulus package ─────────────────────────────────────────

const LT_PREFIX = 'window.LT_DATA = ';

// LT_DATA from lt_data.json, or from the journal builder's lt_data.js ("window.LT_DATA = {...};").
export function parseLtFile(text: string, js: boolean): unknown {
  let t = text.trim();
  if (js) {
    if (!t.startsWith(LT_PREFIX) || !t.endsWith(';')) throw new Error('lt_data.js does not start with "window.LT_DATA = "');
    t = t.slice(LT_PREFIX.length, -1);
  }
  return JSON.parse(t);
}

// The item of every main MOS and SMOS trial, for the rule that no item is on two consecutive main trials (addendum
// §6.1). An opaque `item` id given by the builder is used as it is. Otherwise the item follows from the builder's
// layout: the clips of one item share one SMOS reference, so an SMOS trial's item is its `ref`, and a MOS trial's
// item is the reference of the SMOS trial that plays the same clip. Returns the problems; the server never sends an
// item to a page.
function assignItems(data: any): string[] {
  const errors: string[] = [];
  const refOf = new Map<string, Set<string>>();
  for (const t of data.smos) if (!C.isAttention(t)) {
    if (t.item === undefined) t.item = t.ref;
    if (!refOf.has(t.stim)) refOf.set(t.stim, new Set());
    refOf.get(t.stim)!.add(t.item);
  }
  for (const t of data.mos) if (!C.isAttention(t) && t.item === undefined) {
    const refs = refOf.get(t.stim);
    if (refs?.size === 1) t.item = [...refs][0];
    else errors.push(`mos ${t.tid}: no item id, and no single SMOS trial plays its clip (give the trial an item)`);
  }
  // possible only when no item has more than half of the main trials of a block and half
  for (const block of ['mos', 'smos']) {
    for (const half of data.config.halves ? ['A', 'B'] : [null]) {
      const main = C.selectHalf(data[block].filter((t: any) => !C.isAttention(t)), half);
      const n = new Map<string, number>();
      for (const t of main) if (t.item !== undefined) n.set(t.item, (n.get(t.item) ?? 0) + 1);
      const most = Math.max(0, ...n.values());
      if (most > Math.ceil(main.length / 2)) {
        errors.push(`${block}${half ? ' half ' + half : ''}: an item has ${most} of ${main.length} main trials, so two ` +
                    'of its trials would be consecutive');
      }
    }
  }
  return errors;
}

// Keeps only the fields the page and the session need, so that nothing else in the builder's file can reach a rater.
// Durations are rounded to 0.1 s (addendum §4.2). items: false for the journal's sample data, which has no items.
export function sanitizeLtData(raw: unknown, {items = true} = {}): {data?: LtData; errors: string[]} {
  if (!isObj(raw)) return {errors: ['LT_DATA is missing or not an object']};
  const errors: string[] = [];
  if (raw.version !== 1) errors.push(`unsupported LT_DATA version ${JSON.stringify(raw.version)}`);
  const cfg = isObj(raw.config) ? raw.config : {};
  const hp = isObj(cfg.headphone_check) ? cfg.headphone_check : {};
  const hpc = {trials: hp.trials ?? 6, pass: hp.pass ?? 5, attempts: hp.attempts ?? 2};
  if (!intIn(hpc.trials, 1, 20) || !intIn(hpc.pass, 1, hpc.trials) || !intIn(hpc.attempts, 1, 5)) {
    errors.push('config.headphone_check needs trials 1-20, pass 1-trials, attempts 1-5');
  }
  const config = {title: typeof cfg.title === 'string' ? cfg.title.slice(0, 200) : null, halves: cfg.halves === true,
                  require_full_play: cfg.require_full_play !== false, headphone_check: hpc};
  const stimuli: LtData['stimuli'] = {};
  for (const [sid, s] of Object.entries(isObj(raw.stimuli) ? raw.stimuli : {})) {
    const m = isObj(s) && typeof s.src === 'string' ? /^stimuli\/([^/]+)\.([a-z0-9]+)$/.exec(s.src) : null;
    if (!SID_RE.test(sid)) {
      errors.push(`stimulus id ${JSON.stringify(sid).slice(0, 80)} is not 6-64 letters, digits, - or _ (starting ` +
                  'with a letter or digit)');
    }
    else if (!m || m[1] !== sid || !AUDIO_TYPES[m[2]]) {
      errors.push(`stimulus ${sid}: src must be stimuli/${sid}.<${Object.keys(AUDIO_TYPES).join('|')}>`);
    } else if (typeof s.dur !== 'number' || !(s.dur > 0 && s.dur <= 600)) errors.push(`stimulus ${sid}: dur must be 0-600 s`);
    else stimuli[sid] = {src: s.src, dur: Math.max(0.1, Math.round(s.dur * 10) / 10)};
  }
  const data: any = {version: 1, ...(raw.dry_run === true ? {dry_run: true} : {}), config, stimuli};
  const seen = new Map<string, string>();
  for (const block of BLOCKS) {
    const list = raw[block] ?? [];
    data[block] = [];
    if (!Array.isArray(list)) {
      errors.push(`${block} is not a list`);
      continue;
    }
    list.forEach((t: any, i: number) => {
      if (!isObj(t) || typeof t.tid !== 'string' || !TID_RE.test(t.tid)) {
        errors.push(`${block}[${i}]: tid must be 1-64 letters, digits, - or _ (starting with a letter or digit)`);
        return;
      }
      if (seen.has(t.tid) && seen.get(t.tid) !== block) errors.push(`tid ${t.tid} is in both ${seen.get(t.tid)} and ${block}`);
      seen.set(t.tid, block);
      const o: any = {tid: t.tid};
      for (const f of DATA_FIELDS[block]) o[f] = t[f];
      if (block !== 'practice' && t.attention !== undefined && t.attention !== null) {
        if (!intIn(t.attention, ...RANGE[block])) errors.push(`${block} ${t.tid}: attention must be in ${RANGE[block].join('..')}`);
        o.attention = t.attention;
        if (typeof t.instruction === 'string') o.instruction = t.instruction.slice(0, 500);
      }
      if ((block === 'mos' || block === 'smos') && t.half !== undefined && t.half !== null) {
        if (t.half !== 'A' && t.half !== 'B') errors.push(`${block} ${t.tid}: half must be A or B`);
        o.half = t.half;
      }
      if ((block === 'mos' || block === 'smos') && t.item !== undefined && t.item !== null) {
        if (typeof t.item !== 'string' || !TID_RE.test(t.item)) errors.push(`${block} ${t.tid}: item must be an opaque id`);
        o.item = t.item;
      }
      if (block === 'cmos' && t.group !== undefined && t.group !== null) {
        if (typeof t.group !== 'string' || !TID_RE.test(t.group)) errors.push(`cmos ${t.tid}: group must be an opaque id`);
        o.group = t.group;
      }
      data[block].push(o);
    });
  }
  errors.push(...C.checkData(data));
  if (!['mos', 'smos', 'cmos'].some((b) => data[b].length)) errors.push('no MOS, SMOS or CMOS trials');
  if (items && !errors.length) errors.push(...assignItems(data));
  return errors.length ? {errors} : {data, errors};
}

function sampleRaw(): any {
  const sandbox: any = {window: {}};
  runInNewContext(readFileSync(join(PUBLIC_DIR, 'sample_lt_data.js'), 'utf8'), sandbox);
  return JSON.parse(JSON.stringify(sandbox.window.LT_DATA));
}

// The journal page's sample data (synthetic), with WAV sources: the server makes their audio (sampleWav).
export function loadSampleData(): LtData {
  const raw = sampleRaw();
  for (const [sid, s] of Object.entries(raw.stimuli as Record<string, any>)) s.src = `stimuli/${sid}.wav`;
  const {data, errors} = sanitizeLtData(raw, {items: false});
  if (!data) throw new Error('sample_lt_data.js is invalid: ' + errors.join('; '));
  return data;
}

export function sampleConsentHtml(): string {
  return '<p><b>TEST DATA.</b> This is the sample consent text of the journal page.</p>' + sampleRaw().config.consent_html;
}

export const extOf = (data: LtData, sid: string): string => data.stimuli[sid].src.split('.').pop() as string;

// ── sessions, plans and results ──────────────────────────────

// The rater's session (lt_core.buildSession: the rater's stored half, the order seeded by the rater ID with no item
// on two consecutive main trials, attention placement, A/B order), reduced to what the page shows: no half, item,
// group or instruction, and the clips as opaque ids.
export function makeSession(data: LtData, raterId: string, half: Half): Session {
  const s = C.buildSession(data, raterId, half);
  const strip = (block: Block) => (t: any) => {
    const o: any = {tid: t.tid};
    for (const f of PLAN_FIELDS[block]) o[f] = t[f];
    if (C.isAttention(t)) o.attention = t.attention;
    return o;
  };
  return {raterId, parity: s.parity, half: s.half, config_title: s.config_title,
          practice: s.practice.map(strip('practice')), mos: s.mos.map(strip('mos')), smos: s.smos.map(strip('smos')),
          cmos: s.cmos.map(strip('cmos'))};
}

export function sessionStimuli(s: Session): Set<string> {
  const out = new Set<string>();
  for (const block of BLOCKS) for (const t of s[block]) for (const f of PLAN_FIELDS[block]) out.add(t[f] as string);
  return out;
}

// The journal sample and dry-run builds are shown to raters as TEST DATA.
export const isTestData = (pkg: {sample: boolean; data: LtData}) => pkg.sample || pkg.data.dry_run === true;

// What GET /api/plan sends: the rater's trials and only the stimuli they use. Never the key.
export function planFor(pkg: {id: string; sample: boolean; data: LtData}, s: Session) {
  const stimuli: Record<string, {src: string; dur: number}> = {};
  for (const sid of sessionStimuli(s)) {
    stimuli[sid] = {src: `/s/${sid}.${extOf(pkg.data, sid)}`, dur: pkg.data.stimuli[sid].dur};
  }
  return {package_id: pkg.id, sample: isTestData(pkg), config: pkg.data.config,
          session: {raterId: s.raterId, practice: s.practice, mos: s.mos, smos: s.smos, cmos: s.cmos}, stimuli};
}

export type Headphone = {attempts: {trials: {order: string[]; answer: number; choice: number}[]; correct: number | null}[];
                         passed: boolean};

// Every attempt of the sound check used and failed: the rater can never complete the session.
export const hpFailed = (hp: Headphone, attempts: number) =>
  !hp.passed && hp.attempts.length >= attempts && hp.attempts.every((a) => a.correct !== null);
// A rating session as the exports see it: consent fields are null while the rater has not agreed (nothing is stored
// then; addendum §11.5).
export type SessionRow = {rater_id: string; route: string; package_id: string; session: Session;
                          browser: string | null; consent_at: string | null; consent_lang: string | null;
                          consent_sha: string | null; headphone: Headphone; background: Record<string, string> | null;
                          finished: string | null; completed_at: string | null};
export type AnswerRow = {block: Block; pos: number; tid: string; rating: number; lang: string | null; plays: any;
                         played_s: number | null; rt_ms: number | null; t_ms: number | null; t: string};

// The results file of the journal page's "Download results" (benchmark/listening/page/README.md), which
// benchmark/listening_analysis.py reads, keyed by the rater ID, plus the fields of addendum §11.3: the access route,
// the consent (time, page language, sha256 of the text), the page language of every response and the end of the
// session. user_agent holds the browser family and major version only.
export function resultsFor(row: SessionRow, answers: AnswerRow[]) {
  const s = row.session;
  const responses: Record<Block, any[]> = {practice: [], mos: [], smos: [], cmos: []};
  const block_ms: Record<Block, number> = {practice: 0, mos: 0, smos: 0, cmos: 0};
  const sorted = answers.slice().sort((x, y) => BLOCKS.indexOf(x.block) - BLOCKS.indexOf(y.block) || x.pos - y.pos);
  for (const a of sorted) {
    const t = s[a.block][a.pos];
    let rec: any;
    if (a.block === 'smos') {
      rec = {tid: t.tid, stim: t.stim, ref: t.ref, rating: a.rating, plays_ref: a.plays.ref, plays_stim: a.plays.stim};
    } else if (a.block === 'cmos') {
      rec = {tid: t.tid, a: t.a, b: t.b, rating: a.rating, plays_a: a.plays.a, plays_b: a.plays.b};
    } else {
      rec = {tid: t.tid, stim: t.stim, rating: a.rating, plays: a.plays};
    }
    rec.rt_ms = a.rt_ms;
    rec.t = a.t;
    rec.lang = a.lang;
    if (C.isAttention(t)) rec.attention = t.attention;
    responses[a.block].push(rec);
    block_ms[a.block] += a.t_ms ?? 0;
  }
  return {
    version: 1, rater_id: row.rater_id, route: row.route, parity: s.parity, half: s.half, started: row.consent_at,
    finished: row.finished, completed: row.completed_at, user_agent: row.browser, config_title: s.config_title,
    consent: {at: row.consent_at, lang: row.consent_lang, sha256: row.consent_sha}, headphone: row.headphone,
    responses, block_ms, background: row.background,
  };
}

// The rater's next trial: the first unanswered one in block order (answers are taken in this order only).
export function nextTrial(s: Session, counts: Record<string, number>): {block: Block; index: number} | null {
  for (const block of BLOCKS) {
    if ((counts[block] ?? 0) < s[block].length) return {block, index: counts[block] ?? 0};
  }
  return null;
}

// ── access codes ─────────────────────────────────────────────

// Crockford base32: 12 characters = 60 bits (addendum §11.2). On input, lower case, spaces, O (for 0) and I / L (for
// 1) are accepted. Only 12-character bodies are codes: the 10-character (50-bit) codes of the first version of this
// server never sign in.
const CROCKFORD = '0123456789ABCDEFGHJKMNPQRSTVWXYZ';
const BODY_RE = /^[0-9A-HJKMNP-TV-Z]{12}$/;
export const CODE_CHARS = 12;
export const PREFIX_RE = /^[A-Z0-9]{1,8}$/;

export function randomCode(prefix = ''): string {
  let body = '';
  for (const b of randomBytes(CODE_CHARS)) body += CROCKFORD[b & 31];
  return prefix ? `${prefix}-${body}` : body;
}

export function normalizeCode(input: unknown): string | null {
  if (typeof input !== 'string' || input.length > 40) return null;
  const s = input.trim().toUpperCase().replace(/\s+/g, '');
  const i = s.lastIndexOf('-');
  const prefix = i >= 0 ? s.slice(0, i) : '';
  const body = (i >= 0 ? s.slice(i + 1) : s).replace(/O/g, '0').replace(/[IL]/g, '1');
  if (!BODY_RE.test(body) || (i >= 0 && !PREFIX_RE.test(prefix))) return null;
  return prefix ? `${prefix}-${body}` : body;
}

// What the database keeps of a code: HMAC-SHA-256 with the server secret (hex).
export function codeHmac(secret: string, code: string): string {
  return createHmac('sha256', secret).update(code).digest('hex');
}

export const raterIdOf = (n: number) => 'R' + String(n).padStart(3, '0');

// The key of the login and registration limits for a client address: an IPv4 address as it is, an IPv6 address by its
// /64 prefix (one subscriber usually holds a whole /64), an IPv4-mapped IPv6 address as IPv4. Kept in memory only.
export function ipKey(ip: string): string {
  const s = ip.trim().toLowerCase().replace(/^\[|\]$/g, '').replace(/%.*$/, '');
  const v4 = /^(?:::ffff:)?(\d{1,3}(?:\.\d{1,3}){3})$/.exec(s);
  if (v4) return v4[1];
  if (!s.includes(':')) return s.slice(0, 64);
  const [head, tail] = s.includes('::') ? s.split('::', 2) : [s, null];
  const h = head ? head.split(':') : [];
  const t = tail ? tail.split(':') : [];
  const width = (g: string[]) => g.reduce((n, x) => n + (x.includes('.') ? 2 : 1), 0);
  const groups = tail === null ? h : [...h, ...Array(Math.max(0, 8 - width(h) - width(t))).fill('0'), ...t];
  const top = groups.slice(0, 4);
  if (top.length < 4 || !top.every((g) => /^[0-9a-f]{1,4}$/.test(g))) return s.slice(0, 64);
  return top.map((g) => parseInt(g, 16).toString(16)).join(':') + '::/64';
}

// ── browser family ───────────────────────────────────────────

// The browser family and major version of a User-Agent header, e.g. "Chrome 130": all that is stored of it.
const BROWSERS: [RegExp, string][] = [
  [/\bEdg(?:e|A|iOS)?\/(\d+)/, 'Edge'], [/\bOPR\/(\d+)/, 'Opera'], [/\bcoc_coc_browser\/(\d+)/, 'Coc Coc'],
  [/\bSamsungBrowser\/(\d+)/, 'Samsung Internet'], [/\b(?:Firefox|FxiOS)\/(\d+)/, 'Firefox'],
  [/\b(?:Chrome|CriOS)\/(\d+)/, 'Chrome'],
  [/\bVersion\/(\d+)[\d.]*(?: Mobile\/\S+)? Safari\//, 'Safari'],
];

export function browserFamily(ua: string | null): string | null {
  if (!ua) return null;
  for (const [re, name] of BROWSERS) {
    const m = re.exec(ua);
    if (m) return `${name} ${Number(m[1])}`;
  }
  return 'Other';
}

// ── answers CSV ──────────────────────────────────────────────

export const CSV_COLUMNS = ['package_id', 'rater_id', 'route', 'parity', 'half', 'block', 'pos', 'tid', 'stim', 'ref',
                            'a', 'b', 'rating', 'attention', 'lang', 'plays', 'plays_ref', 'plays_stim', 'plays_a',
                            'plays_b', 'played_s', 'rt_ms', 't_ms', 't'];

function csvCell(v: unknown): string {
  if (v === null || v === undefined) return '';
  let s = String(v);
  if (typeof v === 'string' && /^[=+\-@\t\r]/.test(s) && !/^-?\d+(\.\d+)?$/.test(s)) s = "'" + s;  // no formulas
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
}

export function csvRows(packageId: string, row: SessionRow, answers: AnswerRow[]): string[] {
  const s = row.session;
  const sorted = answers.slice().sort((x, y) => BLOCKS.indexOf(x.block) - BLOCKS.indexOf(y.block) || x.pos - y.pos);
  return sorted.map((a) => {
    const t = s[a.block][a.pos];
    const two = PLAY_ROLES[a.block];
    const rec: Record<string, unknown> = {
      package_id: packageId, rater_id: row.rater_id, route: row.route, parity: s.parity, half: s.half, block: a.block,
      pos: a.pos, tid: a.tid, stim: t.stim, ref: t.ref, a: t.a, b: t.b, rating: a.rating,
      attention: C.isAttention(t) ? t.attention : null, lang: a.lang, plays: two ? null : a.plays,
      plays_ref: a.block === 'smos' ? a.plays.ref : null, plays_stim: a.block === 'smos' ? a.plays.stim : null,
      plays_a: a.block === 'cmos' ? a.plays.a : null, plays_b: a.block === 'cmos' ? a.plays.b : null,
      played_s: a.played_s, rt_ms: a.rt_ms, t_ms: a.t_ms, t: a.t,
    };
    return CSV_COLUMNS.map((c) => csvCell(rec[c])).join(',');
  });
}

// ── synthetic audio of the sample mode ───────────────────────

// A soft voiced-like tone (16 kHz, 16-bit mono WAV) whose pitch follows the stimulus id, so that the sample flow
// can be clicked through with sound. Capped at 6 s.
export function sampleWav(sid: string, dur: number): Uint8Array {
  const sr = 16000, n = Math.round(Math.min(dur, 6) * sr);
  const f0 = 100 + (C.hashString(sid) % 140);
  const buf = new ArrayBuffer(44 + 2 * n);
  const v = new DataView(buf);
  const str = (o: number, s: string) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + 2 * n, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, 2 * sr, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, 'data');
  v.setUint32(40, 2 * n, true);
  const ramp = Math.round(0.05 * sr);
  for (let i = 0; i < n; i++) {
    const t = i / sr;
    const env = Math.min(1, i / ramp, (n - 1 - i) / ramp) * (0.6 + 0.4 * Math.sin(2 * Math.PI * 3.5 * t) ** 2);
    let x = 0;
    for (let k = 1; k <= 5; k++) x += Math.sin(2 * Math.PI * f0 * k * t) / k;
    v.setInt16(44 + 2 * i, Math.round(0.12 * env * x * 32767), true);
  }
  return new Uint8Array(buf);
}
