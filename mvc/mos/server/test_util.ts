// Helpers of the server tests: a temporary data directory with a synthetic stimulus package (opaque ids, as the
// builder writes them) and its key in private/, a running server on a free port, and a small HTTP client with a
// cookie.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { connect } from 'node:net';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startServer, type Options } from './app';
import { C, sampleWav, sha256 } from './core';

export const CONDITIONS = ['nat', 'mvc_v1', 'crtal', 'seedvc', 'knnvc', 'src'];
export const KINDS = ['gold', 'gold_src', 'main', 'attention', 'practice'];
export const KEY_MARKER = 'KEY-MARKER-1f6e2d9a';
export const SOURCE_MARKER = '/datastore/ceai/crt/test-source';
const MOS_CONDITIONS = CONDITIONS.slice(0, 5);
const GENDERS = ['F2F', 'F2M', 'M2F', 'M2M'];
export const N_ITEMS = 8;   // 4 per half
const opaque = (...parts: unknown[]) => sha256('test:' + parts.join(':')).slice(0, 12);
export const pairOf = (item: number) => `vctk2k__p2${10 + item}_00${item}__p3${10 + item}`;

// A package shaped like the builder's (mvc/mos/build_stimuli.py): N_ITEMS items in halves A (even) and B (odd); MOS
// with nat (gold) and four systems; SMOS with the same and src (gold_src on cross-gender items), each with the item's
// reference; a clip of one item and condition shared by MOS and SMOS; 2 MOS and 2 SMOS attention trials on the natural
// clips of items 1 and 3 (new clip ids); 3 practice trials; unrounded durations. The kinds and conditions are in the
// key only. items: also give every main trial an explicit opaque item id.
export function makePackage(dataDir: string, {halves = true, js = false, dryRun = false, items = false} = {}) {
  const pkg = join(dataDir, 'package');
  mkdirSync(join(pkg, 'stimuli'), {recursive: true});
  const stimuli: Record<string, {src: string; dur: number}> = {};
  const clips: Record<string, unknown> = {}, trials: Record<string, unknown> = {};
  const clip = (item: number, cond: string, role: string) => {
    const sid = opaque('clip', item, cond, role);
    if (!stimuli[sid]) {
      const wav = sampleWav(sid, 0.05);
      writeFileSync(join(pkg, 'stimuli', sid + '.wav'), wav);
      stimuli[sid] = {src: `stimuli/${sid}.wav`, dur: 2 + 0.01234 * (item + 1) + 0.1 * CONDITIONS.indexOf(cond)};
      clips[sid] = {pair_id: pairOf(item), condition: cond, role, source_path: `${SOURCE_MARKER}/${item}.wav`,
                    marker: KEY_MARKER};
    }
    return sid;
  };
  const mos: any[] = [], smos: any[] = [], practice: any[] = [];
  for (let item = 0; item < N_ITEMS; item++) {
    const half = item % 2 ? 'B' : 'A', genders = GENDERS[item % 4];
    const id = items ? {item: opaque('item', item)} : {};
    const where = {pair_id: pairOf(item), genders, half};
    for (const cond of MOS_CONDITIONS) {
      const tid = opaque('mos', item, cond);
      mos.push({tid, stim: clip(item, cond, 'stim'), half, ...id});
      trials[tid] = {block: 'mos', kind: cond === 'nat' ? 'gold' : 'main', condition: cond, ...where};
    }
    for (const cond of CONDITIONS) {
      const tid = opaque('smos', item, cond);
      smos.push({tid, stim: clip(item, cond, 'stim'), ref: clip(item, 'ref', 'ref'), half, ...id});
      const kind = cond === 'src' && (genders === 'F2M' || genders === 'M2F') ? 'gold_src' : 'main';
      trials[tid] = {block: 'smos', kind, condition: cond, ...where};
    }
  }
  const attention: [string, number, number][] = [['mos', 1, 2], ['mos', 3, 4], ['smos', 1, 4], ['smos', 3, 2]];
  for (const [n, [block, item, r]] of attention.entries()) {
    const tid = opaque(block, 'attention', n);
    const entry: any = {tid, stim: clip(item, 'nat', 'att' + n), attention: r};
    if (block === 'smos') entry.ref = clip(item, 'ref', 'ref');
    (block === 'mos' ? mos : smos).push(entry);
    trials[tid] = {block, kind: 'attention', expected: r, condition: 'nat', pair_id: pairOf(item)};
  }
  for (let i = 0; i < 3; i++) {
    const tid = opaque('practice', i);
    practice.push({tid, stim: clip(10 + i, 'mvc_v1', 'practice')});
    trials[tid] = {block: 'practice', kind: 'practice', pair_id: pairOf(10 + i), condition: 'mvc_v1'};
  }
  const data = {version: 1, ...(dryRun ? {dry_run: true} : {}), config: {title: 'Listening test: MimiFlow-VC', halves, require_full_play: true,
                                     headphone_check: {trials: 6, pass: 5, attempts: 2},
                                     consent_html: '<p>not used</p>', contact: 'x', submit_url: null},
                stimuli, practice, mos, smos, cmos: []};
  const text = JSON.stringify(data, null, 1);
  if (js) writeFileSync(join(pkg, 'lt_data.js'), 'window.LT_DATA = ' + text + ';\n');
  else writeFileSync(join(pkg, 'lt_data.json'), text);
  mkdirSync(join(dataDir, 'private'), {recursive: true});
  const key = {version: 1, marker: KEY_MARKER, conditions: CONDITIONS, trials, clips};
  writeFileSync(join(dataDir, 'private', 'key.json'), JSON.stringify(key, null, 1));
  return {data, key};
}

// What no answer to a rater or to the admin may hold: anything of the key, a condition or kind as a value, a pair id,
// a source path, or a duration finer than 0.1 s. Returns what was found.
export function leaks(text: string): string[] {
  const out: string[] = [];
  for (const m of [KEY_MARKER, SOURCE_MARKER, 'vctk2k__', 'pair_id', 'condition']) if (text.includes(m)) out.push(m);
  if (/"dur":\s*-?\d+\.\d{2,}/.test(text)) out.push('unrounded duration');
  let data: unknown = null;
  try { data = JSON.parse(text); } catch { /* not JSON */ }
  const values: string[] = [];
  const walk = (x: unknown) => {
    if (typeof x === 'string') values.push(x);
    else if (Array.isArray(x)) x.forEach(walk);
    else if (x && typeof x === 'object') Object.values(x).forEach(walk);
  };
  walk(data);
  for (const v of values) for (const c of [...CONDITIONS, ...KINDS]) if (v === c) out.push(`value ${c}`);
  return out;
}

export function tempDir(): string {
  return mkdtempSync(join(tmpdir(), 'mos-test-'));
}

export async function boot(opts: Partial<Options> & {dataDir?: string} = {}) {
  const dataDir = opts.dataDir ?? tempDir();
  const logs: string[] = [];
  const {server, app, stop} = startServer({dataDir, port: 0, log: (m) => logs.push(m), ...opts});
  const base = `http://127.0.0.1:${server.port}`;
  const token = (await Bun.file(join(dataDir, 'admin_token')).text()).trim();
  return {
    base, app, dataDir, logs, token,
    admin: new Client(base, {Authorization: `Bearer ${token}`}),
    async stop(keep = false) {
      await stop();
      if (!keep) rmSync(dataDir, {recursive: true, force: true});
    },
  };
}

export class Client {
  cookie = '';
  constructor(public base: string, public headers: Record<string, string> = {}) {}

  async req(method: string, path: string, body?: unknown, headers: Record<string, string> = {}) {
    const h: Record<string, string> = {...this.headers, ...headers};
    if (this.cookie) h.Cookie = this.cookie;
    let payload: string | undefined;
    if (body !== undefined) {
      payload = typeof body === 'string' ? body : JSON.stringify(body);
      if (!('Content-Type' in h)) h['Content-Type'] = 'application/json';
    }
    const res = await fetch(this.base + path, {method, headers: h, body: payload});
    const set = res.headers.get('set-cookie');
    if (set) this.cookie = set.split(';')[0];
    const text = await res.text();
    let data: any = null;
    try { data = JSON.parse(text); } catch { /* not JSON */ }
    return {status: res.status, headers: res.headers, text, data};
  }

  get(path: string, headers?: Record<string, string>) { return this.req('GET', path, undefined, headers); }
  post(path: string, body: unknown = {}, headers?: Record<string, string>) { return this.req('POST', path, body, headers); }
}

export const CONSENT = '<p>Thông tin (vi)</p><hr><p>Information (en)</p>';

// The consent text and n invite codes of a ready test: returns the codes.
export async function setup(t: Awaited<ReturnType<typeof boot>>, n = 4) {
  expect200(await t.admin.post('/api/admin/consent', {html: CONSENT}));
  const r = await t.admin.post('/api/admin/codes', {n});
  expect200(r);
  return r.data.codes.map((c: any) => c.code) as string[];
}

function expect200(r: {status: number; text: string}) {
  if (r.status !== 200) throw new Error(`expected 200, got ${r.status}: ${r.text}`);
}

// The answers of a headphone-check attempt that pass it (or fail it: wrong).
export const hpAnswers = (raterId: string, attempt: number, wrong = false) =>
  C.headphoneTrials(6, C.rng(raterId + ':headphone:' + attempt)).map((x: any) => (x.answer + (wrong ? 1 : 0)) % 3);

// Logs in, agrees to the consent shown, and passes the sound check with the right answers. c.raterId: the rater ID.
export async function ready(base: string, code: string, ip = '10.0.0.1', lang = 'vi') {
  const c = new Client(base, {'CF-Connecting-IP': ip}) as Client & {raterId: string};
  const login = await c.post('/api/login', {code});
  if (login.status !== 200) throw new Error('login: ' + login.text);
  c.raterId = login.data.results.rater_id;
  const consent = await c.get('/api/consent');
  expect200(await c.post('/api/consent', {lang, sha256: consent.data.sha256}));
  const right = hpAnswers(c.raterId, 0);
  for (let i = 0; i < 6; i++) expect200(await c.post('/api/headphone', {attempt: 0, trial: i, choice: right[i]}));
  return c;
}

// The body of /api/answer for the trial of the plan at block/index.
export function answerBody(plan: any, block: string, index: number, value = 4) {
  const t = plan.session[block][index];
  const plays = block === 'smos' ? {ref: 1, stim: 1} : block === 'cmos' ? {a: 1, b: 1} : 1;
  return {tid: t.tid, block, value: t.attention ?? value, lang: 'vi', plays, played_s: 3.25, rt_ms: 4100, t_ms: 5200};
}

// A POST with Transfer-Encoding: chunked over a raw socket. body: the JSON text, or a number of filler bytes sent in
// 64 KB chunks without an end, stopping as soon as the server answers. written: the body bytes sent.
export async function chunkedPost(base: string, path: string, body: string | number, headers: Record<string, string> = {}) {
  const port = Number(new URL(base).port);
  const s = connect(port, '127.0.0.1');
  await new Promise((res, rej) => { s.once('connect', res); s.once('error', rej); });
  let got = Buffer.alloc(0), closed = false;
  s.on('data', (d: Buffer) => { got = Buffer.concat([got, d]); });
  s.on('close', () => { closed = true; });
  s.on('error', () => { closed = true; });
  const head = {Host: `127.0.0.1:${port}`, 'Content-Type': 'application/json', 'Transfer-Encoding': 'chunked', ...headers};
  s.write(`POST ${path} HTTP/1.1\r\n${Object.entries(head).map(([k, v]) => `${k}: ${v}\r\n`).join('')}\r\n`);
  let written = 0;
  if (typeof body === 'string') {
    s.write(`${Buffer.byteLength(body).toString(16)}\r\n${body}\r\n0\r\n\r\n`);
    written = Buffer.byteLength(body);
  } else {
    const chunk = Buffer.alloc(64 * 1024, 0x61);
    const frame = Buffer.concat([Buffer.from(chunk.length.toString(16) + '\r\n'), chunk, Buffer.from('\r\n')]);
    while (written < body && !got.length && !closed) {
      written += chunk.length;
      if (!s.write(frame)) {
        await new Promise<void>((res) => {
          const done = () => { clearTimeout(timer); s.off('drain', done); res(); };
          const timer = setTimeout(done, 50);
          s.once('drain', done);
        });
      }
    }
  }
  // the whole answer: headers, then Content-Length bytes
  const end = Date.now() + 5000;
  let status = 0, text = '';
  while (Date.now() < end) {
    const i = got.indexOf('\r\n\r\n');
    if (i >= 0) {
      const headText = got.subarray(0, i).toString('latin1');
      const len = Number(/\r\ncontent-length: *(\d+)/i.exec(headText)?.[1] ?? 0);
      if (got.length >= i + 4 + len) {
        status = Number(headText.split(' ')[1]);
        text = got.subarray(i + 4, i + 4 + len).toString('utf8');
        break;
      }
    }
    if (closed) break;
    await new Promise((res) => setTimeout(res, 5));
  }
  s.destroy();
  let data: any = null;
  try { data = JSON.parse(text); } catch { /* not JSON */ }
  return {status, data, written};
}
