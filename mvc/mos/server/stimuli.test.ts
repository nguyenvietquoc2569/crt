// Stimuli (/s/<sid>.<ext>: auth, consent, content type, Range, caching, traversal), what the plan reveals (no condition
// names, items or halves, only the rater's half, durations to 0.1 s), the order of every rater's trials (no item on two
// consecutive main trials, attention trials never in the first 3 nor adjacent), a full session whose every answer is
// checked for anything of the key, that key.json is unreachable through any route, and the security headers.
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { C } from './core';
import { answerBody, boot, Client, CONDITIONS, KEY_MARKER, leaks, makePackage, N_ITEMS, ready, setup,
         tempDir } from './test_util';

let t: Awaited<ReturnType<typeof boot>>;
let codes: string[];
let rater: Client & {raterId: string};
let plan: any;
let data: any;
let key: any;

beforeAll(async () => {
  const dir = tempDir();
  ({data, key} = makePackage(dir));
  t = await boot({dataDir: dir});
  await t.admin.post('/api/admin/package/import', {});
  codes = await setup(t, 12);
  rater = await ready(t.base, codes[0], '10.9.0.1');
  plan = (await rater.get('/api/plan')).data;
});
afterAll(() => t.stop());

const fileOf = (sid: string) => readFileSync(join(t.dataDir, 'package', 'stimuli', sid + '.wav'));

describe('stimuli', () => {
  test('a rater gets the whole file with the audio type, Range support and a private long cache', async () => {
    const sid = plan.session.mos[0].stim;
    expect(plan.stimuli[sid].src).toBe(`/s/${sid}.wav`);
    const full = await fetch(t.base + plan.stimuli[sid].src, {headers: {Cookie: rater.cookie}});
    expect(full.status).toBe(200);
    expect(full.headers.get('content-type')).toBe('audio/wav');
    expect(full.headers.get('accept-ranges')).toBe('bytes');
    expect(full.headers.get('cache-control')).toBe('private, max-age=31536000, immutable');
    const bytes = new Uint8Array(await full.arrayBuffer());
    expect(Buffer.from(bytes).equals(fileOf(sid))).toBe(true);
    const size = bytes.length;

    const range = async (r: string) => fetch(t.base + `/s/${sid}.wav`, {headers: {Cookie: rater.cookie, Range: r}});
    let res = await range('bytes=10-19');
    expect(res.status).toBe(206);
    expect(res.headers.get('content-range')).toBe(`bytes 10-19/${size}`);
    expect(res.headers.get('content-length')).toBe('10');
    expect(Buffer.from(await res.arrayBuffer()).equals(fileOf(sid).subarray(10, 20))).toBe(true);
    res = await range('bytes=-5');
    expect(res.status).toBe(206);
    expect(Buffer.from(await res.arrayBuffer()).equals(fileOf(sid).subarray(size - 5))).toBe(true);
    res = await range(`bytes=${size - 3}-`);
    expect(res.headers.get('content-range')).toBe(`bytes ${size - 3}-${size - 1}/${size}`);
    expect((await res.arrayBuffer()).byteLength).toBe(3);
    res = await range(`bytes=0-${size + 100}`);
    expect(res.headers.get('content-range')).toBe(`bytes 0-${size - 1}/${size}`);
    res = await range(`bytes=${size}-`);
    expect(res.status).toBe(416);
    expect(res.headers.get('content-range')).toBe(`bytes */${size}`);
    res = await range('bytes=0-1,5-6');   // multiple ranges: the whole file
    expect(res.status).toBe(200);
  });

  test('no audio without a rater session, and only the stimuli of that rater', async () => {
    const sid = plan.session.mos[0].stim;
    expect((await fetch(t.base + `/s/${sid}.wav`)).status).toBe(401);
    // a stimulus of the other half is in the package but not in this rater's plan
    const notMine = Object.keys(data.stimuli).find((s) => !plan.stimuli[s]);
    expect(notMine).toBeDefined();
    expect((await fetch(t.base + `/s/${notMine}.wav`, {headers: {Cookie: rater.cookie}})).status).toBe(404);
    expect((await fetch(t.base + `/s/${sid}.flac`, {headers: {Cookie: rater.cookie}})).status).toBe(404);   // wrong ext
  });

  test('path traversal and directory listing are impossible', async () => {
    const paths = ['/s/../private/key.json', '/s/..%2Fprivate%2Fkey.json', '/s/%2e%2e%2fprivate%2fkey.wav',
                   '/s/..%2F..%2Fkey.wav', '/s/abc.wav', '/s/' + 'a'.repeat(65) + '.wav', '/s/a.b.c.d.e.f.wav', '/s/',
                   '/s', '/stimuli/', '/package/', '/package/lt_data.json', '/package/stimuli/', '/private/',
                   '/private/key.json', '/key/key.json', '/mos.sqlite', '/admin_token', '/server_secret',
                   '/consent_v1.html', '/backup/', '/export/', '/../package/lt_data.json', '/%2e%2e/private/key.json',
                   '/public/', '/server/app.ts', '/lt_core.test.js', '/sample_lt_data.js', '/.env', '/s/-abcdef.wav',
                   '/s/_abcdef.wav'];
    for (const p of paths) {
      const res = await fetch(t.base + p, {headers: {Cookie: rater.cookie}});
      expect([404, 401], p).toContain(res.status);
      const text = await res.text();
      expect(text, p).not.toContain(KEY_MARKER);
      expect(text, p).not.toContain('"stimuli"');
    }
  });
});

describe('what the rater sees', () => {
  test('the plan holds opaque ids only: no condition names, items, halves, groups or instructions', async () => {
    const text = JSON.stringify(plan);
    expect(leaks(text)).toEqual([]);
    expect(text).not.toMatch(/"(half|item|group|instruction|condition|pair_id|kind|expected)"/);
    expect(Object.keys(plan)).toEqual(['package_id', 'sample', 'config', 'session', 'stimuli']);
    expect(Object.keys(plan.session)).toEqual(['raterId', 'practice', 'mos', 'smos', 'cmos']);
    expect(plan.session.raterId).toBe(rater.raterId);
    for (const block of ['practice', 'mos', 'smos']) {
      for (const tr of plan.session[block]) {
        expect(Object.keys(tr).every((k) => ['tid', 'stim', 'ref', 'attention'].includes(k))).toBe(true);
      }
    }
    // the rater's half only: 4 of 8 items x 5 (MOS) and 6 (SMOS) conditions, plus 2 attention trials each
    expect(plan.session.practice.length).toBe(3);
    expect(plan.session.mos.length).toBe(N_ITEMS / 2 * 5 + 2);
    expect(plan.session.smos.length).toBe(N_ITEMS / 2 * 6 + 2);
    const used = new Set<string>();
    for (const b of ['practice', 'mos', 'smos']) for (const tr of plan.session[b]) { used.add(tr.stim); if (tr.ref) used.add(tr.ref); }
    expect(Object.keys(plan.stimuli).sort()).toEqual([...used].sort());
    // durations to 0.1 s (the package's are finer)
    for (const s of Object.values(plan.stimuli) as any[]) expect(Math.round(s.dur * 10) / 10).toBe(s.dur);
    expect(Object.values(data.stimuli).some((s: any) => Math.round(s.dur * 10) / 10 !== s.dur)).toBe(true);
    // the attention trials keep the rating they ask for (the page words the instruction itself)
    expect(plan.session.mos.filter((x: any) => 'attention' in x).map((x: any) => x.attention).sort()).toEqual([2, 4]);
    expect(plan.session.smos.filter((x: any) => 'attention' in x).map((x: any) => x.attention).sort()).toEqual([2, 4]);
  });

  test('every rater: no item on two consecutive main trials, attention never in the first 3 nor adjacent', async () => {
    const itemOf = (tid: string) => key.trials[tid].pair_id;
    const orders = new Set<string>();
    for (const [k, code] of codes.slice(1, 9).entries()) {
      const c = await ready(t.base, code, '10.9.1.' + k);
      const p = (await c.get('/api/plan')).data;
      const halfOf = (tid: string) => key.trials[tid].half;
      for (const block of ['mos', 'smos']) {
        const seq = p.session[block];
        const main = seq.filter((x: any) => !C.isAttention(x));
        for (let i = 1; i < main.length; i++) expect(itemOf(main[i].tid)).not.toBe(itemOf(main[i - 1].tid));
        expect(new Set(main.map((x: any) => halfOf(x.tid))).size).toBe(1);   // one half per rater
        const pos = seq.map((x: any, i: number) => (C.isAttention(x) ? i : -1)).filter((i: number) => i >= 0);
        expect(pos.length).toBe(2);
        expect(pos[0]).toBeGreaterThanOrEqual(3);
        expect(pos[1] - pos[0]).toBeGreaterThan(1);
        orders.add(main.map((x: any) => x.tid).join());
      }
    }
    expect(orders.size).toBe(16);   // every rater and block has an order of its own
  });

  test('a full session: no answer to the rater or the admin holds anything of the key', async () => {
    const c = await ready(t.base, codes[9], '10.9.2.1');
    const bodies: [string, string][] = [];
    const get = async (p: string, who: Client = c) => bodies.push([p, (await who.get(p)).text]);
    const post = async (p: string, b: unknown) => bodies.push([p, (await c.post(p, b)).text]);
    await get('/api/info');
    await get('/api/consent');
    await get('/api/state');
    const p = (await c.get('/api/plan')).data;
    await get('/api/plan');
    for (const block of ['practice', 'mos', 'smos']) {
      for (let i = 0; i < p.session[block].length; i++) await post('/api/answer', answerBody(p, block, i));
      await get('/api/state');
    }
    await post('/api/background', {age: '18–24', first_language: 'Other', english: 'Basic', hearing: 'Yes',
                                   device: 'Other', previous_test: 'Yes'});
    await post('/api/finish', {});
    await t.admin.post('/api/admin/export', {});
    for (const q of ['/api/admin/status', '/api/admin/raters', '/api/admin/consent', `/api/admin/export/rater?rater=${c.raterId}`,
                     '/api/admin/export/answers.csv']) await get(q, t.admin);
    for (const [path, text] of bodies) expect(leaks(text), path).toEqual([]);
    expect(bodies.length).toBeGreaterThan(50);
  });

  test('the state and the consent hold no condition names either', async () => {
    const plan2 = (await rater.get('/api/plan')).data;
    await rater.post('/api/answer', answerBody(plan2, 'practice', 0));
    for (const p of ['/api/state', '/api/info', '/api/consent']) {
      const text = (await rater.get(p)).text;
      expect(leaks(text), p).toEqual([]);
      for (const c of CONDITIONS) expect(text, p).not.toContain(`"${c}"`);
    }
  });

  test('key.json is not reachable through any route, the admin exports included', async () => {
    const admin = t.admin;
    await admin.post('/api/admin/export', {});
    const gets = ['/', '/index.html', '/admin', '/style.css', '/i18n.js', '/lt_core.js', '/lt_app.js', '/admin.js',
                  '/api/info', '/api/state', '/api/plan', '/api/consent', '/api/admin/status',
                  '/api/admin/raters', '/api/admin/consent', `/api/admin/export/rater?rater=${rater.raterId}`,
                  '/api/admin/export/answers.csv', '/api/admin/export/rater?rater=../private/key',
                  '/api/admin/raters?package=../private/key.json', '/api/admin/export/answers.csv?package=key'];
    for (const p of gets) {
      for (const c of [rater, admin]) {
        const text = (await c.get(p)).text;
        expect(text, p).not.toContain(KEY_MARKER);
        expect(text, p).not.toContain('gold_src');
      }
    }
    const r = await admin.post('/api/admin/export', {package: '../private'});
    expect(r.status).toBe(404);
  });
});

describe('headers', () => {
  test('security headers on pages, API answers, errors and audio', async () => {
    const sid = plan.session.mos[0].stim;
    for (const [p, cookie] of [['/', ''], ['/api/info', ''], ['/api/nothing', ''], ['/nothing', ''],
                               [`/s/${sid}.wav`, rater.cookie]]) {
      const res = await fetch(t.base + p, {headers: cookie ? {Cookie: cookie} : {}});
      const h = res.headers;
      expect(h.get('content-security-policy'), p).toContain("default-src 'self'");
      expect(h.get('content-security-policy'), p).toContain("media-src 'self'");
      expect(h.get('content-security-policy'), p).toContain("frame-ancestors 'none'");
      expect(h.get('x-content-type-options'), p).toBe('nosniff');
      expect(h.get('referrer-policy'), p).toBe('no-referrer');
      expect(h.get('strict-transport-security'), p).toContain('max-age=');
    }
    const page = await fetch(t.base + '/');
    expect(page.headers.get('content-type')).toBe('text/html; charset=utf-8');
    expect(await page.text()).toContain('<script src="lt_app.js"></script>');
  });
});
