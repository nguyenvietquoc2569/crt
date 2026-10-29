// Admin API: the token and the server secret, the start (key files refused), the daily backup, codes (60 bits, kept
// only as an HMAC, rater IDs in issue order, the half set at issue by the consented counts that can still complete,
// the cap on the codes of both routes), package import (items), the consent text (placeholders, handlers and script
// URLs refused, also when hidden), settings, the raters list, the exports (the journal page's results format with
// rater IDs, the route and the page language, read by benchmark/listening_analysis.py, and the CSV), withdrawal (the
// database, its write-ahead log, the exports and the daily copies), the sample mode, restarts and a first-version
// database (refused when it holds codes).
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { Database } from 'bun:sqlite';
import { chmodSync, existsSync, readdirSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { createApp, MAX_CODES } from './app';
import { C, codeHmac, CSV_COLUMNS, normalizeCode, PUBLIC_DIR, sanitizeLtData } from './core';
import { MIGRATIONS, SCHEMA_VERSION } from './db';
import { answerBody, boot, Client, CONSENT, hpAnswers, KEY_MARKER, leaks, makePackage, ready, setup,
         tempDir } from './test_util';

const REPO = resolve(import.meta.dir, '..', '..', '..');
const CODE_RE = /^[0-9A-HJKMNP-TV-Z]{12}$/;
const BG = {age: '18–24', first_language: 'Vietnamese', english: 'Intermediate', hearing: 'No',
            device: 'In-ear earphones or earbuds', previous_test: 'No'};
const secretOf = (dir: string) => readFileSync(join(dir, 'server_secret'), 'utf8').trim();
// The bytes of the database files, to show that a code is in none of them.
const dbBytes = (dir: string) => ['mos.sqlite', 'mos.sqlite-wal'].filter((f) => existsSync(join(dir, f)))
  .map((f) => readFileSync(join(dir, f)).toString('latin1')).join('');

describe('admin token, server secret, start and backup', () => {
  let t: Awaited<ReturnType<typeof boot>>;
  beforeAll(async () => { t = await boot(); });
  afterAll(() => t.stop());

  test('token and secret: created with mode 600 on the first start, high entropy, never logged', async () => {
    expect(statSync(t.app.tokenPath).mode & 0o777).toBe(0o600);
    expect(statSync(join(t.dataDir, 'server_secret')).mode & 0o777).toBe(0o600);
    expect(t.token).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(secretOf(t.dataDir)).toMatch(/^[A-Za-z0-9_-]{43}$/);
    expect(secretOf(t.dataDir)).not.toBe(t.token);
    const logs = t.logs.join('\n');
    expect(logs).not.toContain(t.token);
    expect(logs).not.toContain(secretOf(t.dataDir));
    expect(logs).toContain('admin_token');
    expect(logs).toContain('server_secret');
    for (const p of ['/api/admin/status', '/api/info']) expect((await t.admin.get(p)).text).not.toContain(secretOf(t.dataDir));
  });

  test('missing, wrong, malformed and Basic tokens are refused on every admin route', async () => {
    const routes: [string, string][] = [['GET', '/api/admin/status'], ['GET', '/api/admin/raters'],
      ['POST', '/api/admin/codes'], ['POST', '/api/admin/package/import'], ['POST', '/api/admin/consent'],
      ['GET', '/api/admin/consent'], ['POST', '/api/admin/settings'], ['POST', '/api/admin/export'],
      ['GET', '/api/admin/export/answers.csv'], ['GET', '/api/admin/export/rater?rater=R001'],
      ['POST', '/api/admin/raters/disable'], ['POST', '/api/admin/raters/find'], ['POST', '/api/admin/raters/delete']];
    const bad = [{}, {Authorization: 'Bearer wrong'}, {Authorization: `Bearer ${t.token}x`},
                 {Authorization: `bearer ${t.token}`}, {Authorization: `Basic ${btoa('admin:' + t.token)}`},
                 {Authorization: `Bearer ${t.token.slice(0, -1)}`}, {Authorization: 'Bearer '}];
    for (const [method, path] of routes) {
      for (const h of bad) {
        const r = await new Client(t.base, h as Record<string, string>).req(method, path, method === 'POST' ? {n: 1} : undefined);
        expect(r.status, `${method} ${path} ${JSON.stringify(h)}`).toBe(401);
        expect(r.data.error.code).toBe('unauthorized');
      }
    }
    expect((await t.admin.get('/api/admin/status')).status).toBe(200);
    expect(t.app.store.raters().length).toBe(0);   // no refused request created a code
  });

  test('the data directory is made owner-only, also when it existed before the first start', async () => {
    expect(statSync(t.dataDir).mode & 0o777).toBe(0o700);
    const dir = tempDir();
    chmodSync(dir, 0o755);   // as the stimulus builder may leave it
    const u = await boot({dataDir: dir});
    expect(statSync(dir).mode & 0o777).toBe(0o700);
    await u.stop();
  });

  test('a restart keeps the token, the secret, the database, its schema version and the codes', async () => {
    const dir = tempDir();
    const a = await boot({dataDir: dir});
    const codes = (await a.admin.post('/api/admin/codes', {n: 2})).data.codes;
    const secret = secretOf(dir);
    await a.stop(true);
    const b = await boot({dataDir: dir});
    expect(b.token).toBe(a.token);
    expect(secretOf(dir)).toBe(secret);
    expect(b.logs.join()).not.toContain('created');
    expect(b.app.store.raters().map((r) => r.rater_id)).toEqual(['R001', 'R002']);
    expect((b.app.store.db.query('PRAGMA user_version').get() as any).user_version).toBe(SCHEMA_VERSION);
    expect(SCHEMA_VERSION).toBe(2);
    expect((b.app.store.db.query('PRAGMA journal_mode').get() as any).journal_mode).toBe('wal');
    makePackage(dir);
    await b.admin.post('/api/admin/package/import', {});
    await b.admin.post('/api/admin/consent', {html: CONSENT});
    expect((await new Client(b.base).post('/api/login', {code: codes[1].code})).data.results.rater_id).toBe('R002');
    await b.stop();
  });

  test('the server does not start while the package folder holds a key file', () => {
    for (const where of ['key.json', 'key_v1.json', join('stimuli', 'KEY.json')]) {
      const dir = tempDir();
      makePackage(dir);
      writeFileSync(join(dir, 'package', where), '{}');
      expect(() => createApp({dataDir: dir, log: () => {}})).toThrow(/key belongs in .*private.*does not start/);
      rmSync(dir, {recursive: true, force: true});
    }
  });

  test('a daily copy of the database goes to backup/, owner-only, once a day', async () => {
    const dir = tempDir();
    const a = await boot({dataDir: dir});
    await a.admin.post('/api/admin/codes', {n: 1});
    await a.stop(true);
    const files = readdirSync(join(dir, 'backup'));
    expect(files.length).toBe(1);
    expect(files[0]).toMatch(/^mos-\d{8}\.sqlite$/);
    const f = join(dir, 'backup', files[0]);
    expect(statSync(f).mode & 0o777).toBe(0o600);
    const copy = new Database(f, {readonly: true});
    expect((copy.query('PRAGMA user_version').get() as any).user_version).toBe(SCHEMA_VERSION);
    copy.close();
    const b = await boot({dataDir: dir});
    expect(readdirSync(join(dir, 'backup'))).toEqual(files);   // the same day: no second copy
    await b.stop(true);
    const c = await boot({dataDir: dir, backup: false});
    expect(c.logs.join()).not.toContain('copied');
    await c.stop();
  });
});

describe('codes, package, consent, settings', () => {
  let t: Awaited<ReturnType<typeof boot>>;
  beforeAll(async () => {
    const dir = tempDir();
    makePackage(dir, {js: true});
    t = await boot({dataDir: dir});
  });
  afterAll(() => t.stop());

  test('invite codes: 60 bits, optional prefix, unique, rater IDs in issue order, halves balanced', async () => {
    let r = await t.admin.post('/api/admin/codes', {n: 20, prefix: 'vc', note: 'pilot'});
    expect(r.status).toBe(200);
    const codes = r.data.codes.map((c: any) => c.code);
    expect(new Set(codes).size).toBe(20);
    for (const c of codes) expect(c).toMatch(/^VC-[0-9A-HJKMNP-TV-Z]{12}$/);   // 12 x 5 bits
    expect(r.data.codes.map((c: any) => c.rater_id)).toEqual(Array.from({length: 20}, (_, i) => 'R' + String(i + 1).padStart(3, '0')));
    // no rater has consented yet: the halves alternate
    expect(r.data.codes.map((c: any) => c.half).join('')).toBe('AB'.repeat(10));
    r = await t.admin.post('/api/admin/codes', {n: 1});
    expect(r.data.codes[0]).toMatchObject({rater_id: 'R021', half: 'A'});
    expect(r.data.codes[0].code).toMatch(CODE_RE);
    for (const b of [{n: 0}, {n: 501}, {n: 1.5}, {n: 2, prefix: 'TOO-LONG-1'}, {n: 2, prefix: 'a b'}, {n: 1, x: 1}]) {
      expect((await t.admin.post('/api/admin/codes', b)).status).toBe(400);
    }
    // the admin page allows what the server allows
    expect(readFileSync(join(PUBLIC_DIR, 'admin.html'), 'utf8')).toContain('<input id="n" type="number" min="1" max="500"');
    // the database keeps the HMAC of each code with the server secret, never the code
    const secret = secretOf(t.dataDir);
    for (const c of [...r.data.codes, ...codes.map((code: string) => ({code}))]) {
      expect(t.app.store.raterByHmac(codeHmac(secret, c.code))).not.toBeNull();
    }
    expect(t.app.store.raters()[0].code_hmac).toBe(codeHmac(secret, codes[0]));
    t.app.store.db.exec('PRAGMA wal_checkpoint(TRUNCATE)');
    const bytes = dbBytes(t.dataDir);
    for (const code of codes) {
      expect(bytes).not.toContain(code);
      expect(bytes).not.toContain(code.slice(3));
    }
    // the codes are shown once: no admin answer lists them again
    const list = await t.admin.get('/api/admin/raters');
    for (const code of codes) expect(list.text).not.toContain(code.slice(3));
    expect(Object.keys(list.data.raters[0])).not.toContain('code');
  });

  test('rater page reports not open until a package and a consent text exist', async () => {
    expect((await t.admin.get('/api/info')).data.open).toBe(false);
    expect((await new Client(t.base).get('/api/consent')).status).toBe(404);
    expect((await new Client(t.base).post('/api/register', {})).status).toBe(503);
  });

  test('package import: refuses a key in the package, broken data and missing files; accepts lt_data.js', async () => {
    const pkg = join(t.dataDir, 'package');
    for (const f of ['key.json', 'the_key.txt']) {
      writeFileSync(join(pkg, f), '{}');
      const r = await t.admin.post('/api/admin/package/import', {});
      expect([r.status, r.data.error.code]).toEqual([400, 'key_in_package']);
      expect(r.data.error.message).toContain('private');
      rmSync(join(pkg, f));
    }

    const good = readFileSync(join(pkg, 'lt_data.js'), 'utf8');
    const data = JSON.parse(good.slice('window.LT_DATA = '.length, -2));
    const broken = structuredClone(data);
    broken.mos[0].stim = 'missing0001';
    broken.mos[1].tid = '../x';
    broken.smos[0].attention = 9;
    broken.stimuli['bad/sid'] = {src: 'stimuli/bad/sid.wav', dur: 1};
    // ids start with a letter or digit, so that answers.csv never needs to alter one ("-x" reads as a formula)
    broken.stimuli['-abcdef1'] = {src: 'stimuli/-abcdef1.wav', dur: 1};
    broken.mos[2].tid = '-x2';
    broken.smos[1].tid = '_y1';
    broken.smos[2].item = '../i';
    writeFileSync(join(pkg, 'lt_data.js'), 'window.LT_DATA = ' + JSON.stringify(broken) + ';\n');
    let r = await t.admin.post('/api/admin/package/import', {});
    expect([r.status, r.data.error.code]).toEqual([400, 'bad_package']);
    const details = r.data.error.details.join('\n');
    expect(details).toContain('unknown stimulus stim=missing0001');
    expect(details).toContain('tid must be');
    expect(details).toContain('attention must be in 1..5');
    expect(details).toContain('is not 6-64 letters');
    expect(details).toContain('stimulus id "-abcdef1" is not 6-64 letters, digits, - or _ (starting with a letter or digit)');
    expect(details).toContain('mos[2]: tid must be 1-64 letters, digits, - or _ (starting with a letter or digit)');
    expect(details).toContain('smos[1]: tid must be');
    expect(details).toContain('item must be an opaque id');
    writeFileSync(join(pkg, 'lt_data.js'), good);

    const sid = Object.keys(data.stimuli)[0];
    const f = join(pkg, 'stimuli', sid + '.wav');
    const bytes = readFileSync(f);
    rmSync(f);
    r = await t.admin.post('/api/admin/package/import', {});
    expect(r.status).toBe(400);
    expect(r.data.error.details[0]).toContain('missing');
    symlinkSync(join(t.dataDir, 'private', 'key.json'), f);   // a link out of the package is not served
    r = await t.admin.post('/api/admin/package/import', {});
    expect(r.data.error.details[0]).toContain('not a non-empty regular file');
    rmSync(f);
    writeFileSync(f, bytes);

    r = await t.admin.post('/api/admin/package/import', {});
    expect(r.status).toBe(200);
    expect(r.data.trials).toEqual({practice: 3, mos: 8 * 5 + 2, smos: 8 * 6 + 2, cmos: 0});
    const id = r.data.package_id;
    expect((await t.admin.post('/api/admin/package/import', {})).data).toEqual({ok: true, package_id: id, unchanged: true});
  });

  test('items: from the shared SMOS reference, or as given; a trial without one or too many of one is refused', () => {
    const dirs = [tempDir(), tempDir()];
    try {
      items(dirs);
    } finally {
      for (const d of dirs) rmSync(d, {recursive: true, force: true});   // no synthetic key left in $TMPDIR
    }
  });

  function items(dirs: string[]) {
    const {data} = makePackage(dirs[0]);
    const ok = sanitizeLtData(structuredClone(data));
    expect(ok.errors).toEqual([]);
    // every main trial of an item gets the item's reference id: MOS through the SMOS trial with the same clip
    const smosRef = new Map((data.smos as any[]).filter((x) => !('attention' in x)).map((x) => [x.stim, x.ref]));
    for (const tr of ok.data!.mos as any[]) if (!('attention' in tr)) expect(tr.item).toBe(smosRef.get(tr.stim));
    for (const tr of ok.data!.smos as any[]) if (!('attention' in tr)) expect(tr.item).toBe(tr.ref);
    for (const tr of [...ok.data!.mos, ...ok.data!.smos] as any[]) if ('attention' in tr) expect(tr.item).toBeUndefined();
    // an explicit item id is kept
    const {data: given} = makePackage(dirs[1], {items: true});
    const g = sanitizeLtData(structuredClone(given));
    expect((g.data!.mos[0] as any).item).toBe((given.mos[0] as any).item);
    // a MOS clip no SMOS trial plays, without an item id
    const lone = structuredClone(data);
    const sid = 'lone0001';
    lone.stimuli[sid] = {src: `stimuli/${sid}.wav`, dur: 2};
    lone.mos[0].stim = sid;
    expect(sanitizeLtData(lone).errors.join()).toContain(`mos ${lone.mos[0].tid}: no item id`);
    // one item with more than half of a block's main trials in a half
    const crowded = structuredClone(given);
    for (const tr of crowded.mos as any[]) if (tr.half === 'A' && tr.item) tr.item = (given.mos[0] as any).item;
    expect(sanitizeLtData(crowded).errors.join()).toMatch(/mos half A: an item has 20 of 20 main trials/);
    // the journal's sample data have no items and need none
    expect(sanitizeLtData(structuredClone(data), {items: false}).data!.mos.every((x: any) => x.item === undefined)).toBe(true);
  }

  test('the consent text: placeholders, scripts and handlers refused; saved as consent_v1.html and served with its sha256', async () => {
    const template = readFileSync(join(import.meta.dir, '..', 'consent_template_v1.html'), 'utf8');
    for (const html of ['', template, '<p>Contact: [CONTACT_NAME]</p>', '<p>[CONTACT_EMAIL]</p>', '<p>[ETHICS_NOTE]</p>',
                        '<p>until [RETENTION_DATE]</p>', '<p>Contact: [CONTACT]</p>', '<p>x</p><script>alert(1)</script>',
                        '<img src=x onerror=alert(1)>', '<a href="javascript:alert(1)">x</a>', '<iframe src="/"></iframe>',
                        // handlers after "/" or a quote, and script URLs behind character references (the review's)
                        '<svg/onload=alert(1)>', '<img/src=x/onerror=alert(1)>', '<a href="java&#115;cript:alert(1)">x</a>',
                        '<IMG SRC=x ONERROR=alert(1)>', '<p title="x"onclick="alert(1)">x</p>',
                        '<a href="jav&#x09;ascript:alert(1)">x</a>', '<a href="java&Tab;script:alert(1)">x</a>',
                        '<a href=" &#106;avascript:alert(1)">x</a>', '<a href="javascript&colon;alert(1)">x</a>',
                        '<a href="vbscript:x">x</a>', '<svg><a xlink:href="javascript:alert(1)">x</a></svg>',
                        '<frame src="/">', '<p>x</p><SCRIPT SRC=/x.js></SCRIPT>']) {
      const r = await t.admin.post('/api/admin/consent', {html});
      expect(r.status, html.slice(0, 50)).toBe(400);
    }
    // words in the text itself are not tags: no false alarm
    const prose = '<p>It depends on = nothing; the word javascript: is text here.</p>';
    expect((await t.admin.post('/api/admin/consent', {html: prose})).status).toBe(200);
    const r0 = await t.admin.post('/api/admin/consent', {html: template});
    expect(r0.data.error.message).toBe('placeholders left: [CONTACT_NAME], [CONTACT_EMAIL], [ETHICS_NOTE], ' +
                                       '[RETENTION_DATE], [ETHICS');
    expect((await t.admin.post('/api/admin/consent', {lang: 'vi', html: '<p>x</p>'})).status).toBe(400);   // one text
    // the filled template is accepted
    const filled = template.replaceAll('[CONTACT_NAME]', 'Quoc Nguyen').replaceAll('[CONTACT_EMAIL]', 'contact@example.org')
      .replaceAll('[ETHICS_NOTE]', 'Ethics note.').replaceAll('[RETENTION_DATE]', '31 Dec 2027');
    const r = await t.admin.post('/api/admin/consent', {html: filled});
    expect(r.status).toBe(200);
    expect(readFileSync(join(t.dataDir, 'consent_v1.html'), 'utf8')).toBe(filled.trim() + '\n');
    expect(statSync(join(t.dataDir, 'consent_v1.html')).mode & 0o777).toBe(0o600);
    const served = (await new Client(t.base).get('/api/consent')).data;
    expect(served).toEqual({html: filled.trim() + '\n', sha256: r.data.sha256});
    expect((await t.admin.get('/api/admin/consent')).data).toEqual(served);
    expect((await new Client(t.base).get('/api/info')).data.open).toBe(true);
    // a file written to the data directory by hand with a placeholder left is refused too: the test closes
    writeFileSync(join(t.dataDir, 'consent_v1.html'), filled.replace('31 Dec 2027', '[RETENTION_DATE]'));
    expect((await new Client(t.base).get('/api/consent')).status).toBe(404);
    expect((await new Client(t.base).get('/api/info')).data.open).toBe(false);
    expect((await t.admin.get('/api/admin/status')).data.consent.problems).toEqual(['placeholders left: [RETENTION_DATE]']);
    expect((await t.admin.post('/api/admin/consent', {html: filled})).status).toBe(200);
    // a consent POST with too large a body
    expect((await t.admin.post('/api/admin/consent', {html: '<p>' + 'x'.repeat(300 * 1024) + '</p>'})).status).toBe(413);
  });

  test('settings: open registration (on by default), contact and the code cap (40 by default), validated', async () => {
    const st = (await t.admin.get('/api/admin/status')).data;
    expect(st.open_registration).toBe(true);
    expect(MAX_CODES).toBe(40);
    expect(st.codes).toEqual({issued: 21, max: 40, left: 19});
    for (const b of [{open_registration: 'yes'}, {contact: 'x'.repeat(201)}, {max_codes: -1}, {max_codes: 10001},
                     {max_codes: 1.5}, {max_codes: '40'}, {max_codes: null}, {contact: 'x', max_codes: -1}]) {
      expect((await t.admin.post('/api/admin/settings', b)).status, JSON.stringify(b)).toBe(400);
    }
    expect((await t.admin.get('/api/admin/status')).data.contact).toBe('');   // a refused request changes nothing
    const r = await t.admin.post('/api/admin/settings', {contact: 'nvquoc@uit.edu.vn', open_registration: false});
    expect(r.data).toEqual({ok: true, open_registration: false, contact: 'nvquoc@uit.edu.vn', max_codes: 40});
    expect((await new Client(t.base).get('/api/info')).data).toMatchObject({contact: 'nvquoc@uit.edu.vn', open_registration: false});
  });
});

describe('halves at issue and registration', () => {
  test('the next code goes to the half with fewer consented raters; ties alternate; disabled raters do not count', async () => {
    const dir = tempDir();
    makePackage(dir);
    const t = await boot({dataDir: dir});
    try {
      await t.admin.post('/api/admin/package/import', {});
      const first = (await t.admin.post('/api/admin/codes', {n: 6})).data.codes;   // A B A B A B
      expect(first.map((c: any) => c.half).join('')).toBe('ABABAB');
      // three raters of half A consent, one of half B
      for (const [k, c] of [first[0], first[2], first[4], first[1]].entries()) {
        if (k === 0) await t.admin.post('/api/admin/consent', {html: CONSENT});
        await ready(t.base, c.code, '10.13.0.' + k);
      }
      // A 3, B 1 consented: the next two codes go to B, then they alternate (as if the new raters consent)
      let next = (await t.admin.post('/api/admin/codes', {n: 4})).data.codes;
      expect(next.map((c: any) => c.half).join('')).toBe('BBAB');
      // registration follows the same rule, one code at a time: still A 3, B 1
      const reg = await new Client(t.base, {'CF-Connecting-IP': '10.13.1.1'}).post('/api/register', {});
      expect(reg.data.results.half).toBe('B');
      expect(t.app.store.rater(reg.data.results.rater_id)!.half).toBe('B');
      // two raters of half A disabled: A 1, B 1 consented; the tie goes to the half with fewer active codes
      await t.admin.post('/api/admin/raters/disable', {rater_id: first[0].rater_id, disabled: true});
      await t.admin.post('/api/admin/raters/disable', {rater_id: first[2].rater_id, disabled: true});
      const issued = (await t.admin.get('/api/admin/raters')).data.halves;
      expect([issued.A.issued, issued.B.issued, issued.A.consented, issued.B.consented]).toEqual([2, 7, 1, 1]);
      next = (await t.admin.post('/api/admin/codes', {n: 1})).data.codes;
      expect(next[0].half).toBe('A');
      // the stored half is the rater's half in every package, whatever the rater ID's digits say
      const st = (await new Client(t.base, {'CF-Connecting-IP': '10.13.1.2'}).post('/api/login', {code: next[0].code})).data;
      expect([next[0].rater_id, st.results.half]).toEqual(['R012', 'A']);
      // not derived from the rater ID: the digits of R001 are odd, its half is A
      expect([C.raterParity('R001'), t.app.store.rater('R001')!.half]).toEqual([1, 'A']);
    } finally {
      await t.stop();
    }
  });

  test('a rater who failed the sound check can never complete and is not counted for the half', async () => {
    const dir = tempDir();
    makePackage(dir);
    const t = await boot({dataDir: dir});
    try {
      await t.admin.post('/api/admin/package/import', {});
      const first = await setup(t, 4);   // R001 A, R002 B, R003 A, R004 B
      await ready(t.base, first[0], '10.15.0.1');   // A, passes the sound check
      await ready(t.base, first[1], '10.15.0.2');   // B, passes
      const a2 = new Client(t.base, {'CF-Connecting-IP': '10.15.0.3'});
      const id = (await a2.post('/api/login', {code: first[2]})).data.results.rater_id;
      await a2.post('/api/consent', {lang: 'vi', sha256: (await a2.get('/api/consent')).data.sha256});
      // consented A 2, B 1: the next code goes to B (R005; issued A 2, B 3)
      expect((await t.admin.post('/api/admin/codes', {n: 1})).data.codes[0].half).toBe('B');
      // A's second rater fails both attempts: A 1, B 1 can still complete, a tie, so the half with fewer codes: A
      for (const attempt of [0, 1]) {
        const wrong = hpAnswers(id, attempt, true);
        for (let i = 0; i < 6; i++) await a2.post('/api/headphone', {attempt, trial: i, choice: wrong[i]});
      }
      expect((await t.admin.post('/api/admin/codes', {n: 1})).data.codes[0].half).toBe('A');
      const h = (await t.admin.get('/api/admin/raters')).data.halves;
      expect([h.A.consented, h.A.headphone_failed, h.B.consented, h.B.headphone_failed]).toEqual([2, 1, 1, 0]);
    } finally {
      await t.stop();
    }
  });
});

describe('the cap on the codes of both routes (addendum §7.1)', () => {
  test('invite codes and registrations stop at the cap; every code ever issued counts; the admin can raise it', async () => {
    const dir = tempDir();
    makePackage(dir);
    const t = await boot({dataDir: dir});
    try {
      await t.admin.post('/api/admin/package/import', {});
      await t.admin.post('/api/admin/consent', {html: CONSENT});
      // the default: 40 codes in all
      expect((await t.admin.post('/api/admin/codes', {n: 41})).status).toBe(409);
      expect((await t.admin.post('/api/admin/codes', {n: 38})).status).toBe(200);
      const reg = (ip: string) => new Client(t.base, {'CF-Connecting-IP': ip}).post('/api/register', {});
      expect((await reg('10.16.0.1')).data.results.route).toBe('open');
      let r = await t.admin.post('/api/admin/codes', {n: 2});
      expect([r.status, r.data.error.code]).toEqual([409, 'code_cap']);
      expect(r.data.error.message).toBe('the cap of 40 codes (both routes) allows 1 more; raise it in the settings to ' +
                                        'issue more');
      expect((await reg('10.16.0.2')).status).toBe(200);   // the 40th code
      expect((await t.admin.get('/api/admin/status')).data.codes).toEqual({issued: 40, max: 40, left: 0});
      // full: no registration (the page offers none), no invite code
      expect((await new Client(t.base).get('/api/info')).data.open_registration).toBe(false);
      r = await reg('10.16.0.3');
      expect([r.status, r.data.error.code]).toEqual([403, 'closed']);
      expect((await t.admin.post('/api/admin/codes', {n: 1})).status).toBe(409);
      expect(t.app.store.raters().length).toBe(40);
      // a withdrawn code still counts as issued
      await t.admin.post('/api/admin/raters/delete', {rater_id: 'R001'});
      expect((await t.admin.get('/api/admin/status')).data.codes).toEqual({issued: 40, max: 40, left: 0});
      expect((await reg('10.16.0.4')).status).toBe(403);
      // the admin raises the cap (logged); registration and invites work again
      expect((await t.admin.post('/api/admin/settings', {max_codes: 42})).data.max_codes).toBe(42);
      expect(t.logs.join('\n')).toContain('code cap changed from 40 to 42 (40 issued)');
      expect((await new Client(t.base).get('/api/info')).data.open_registration).toBe(true);
      expect((await reg('10.16.0.5')).data.results.rater_id).toBe('R041');
      expect((await t.admin.post('/api/admin/codes', {n: 1})).data.codes[0].rater_id).toBe('R042');
      // a cap of 0 closes both routes at once
      await t.admin.post('/api/admin/settings', {max_codes: 0});
      expect((await t.admin.get('/api/admin/status')).data.codes).toEqual({issued: 42, max: 0, left: 0});
      expect((await reg('10.16.0.6')).status).toBe(403);
      // the admin page has the setting and shows the counts
      const html = readFileSync(join(PUBLIC_DIR, 'admin.html'), 'utf8');
      expect(html).toContain('<input id="maxcodes" type="number" min="0" max="10000">');
      expect(readFileSync(join(PUBLIC_DIR, 'admin.js'), 'utf8')).toContain("per('completed')");
    } finally {
      await t.stop();
    }
  });
});

describe('exports and the raters list', () => {
  let t: Awaited<ReturnType<typeof boot>>;
  let codes: string[];
  let ids: string[];
  let key: any;

  beforeAll(async () => {
    const dir = tempDir();
    key = makePackage(dir).key;
    t = await boot({dataDir: dir});
    await t.admin.post('/api/admin/package/import', {});
    codes = await setup(t, 3);
    ids = [];
    // rater 0 completes everything in English, rater 1 does part of MOS, rater 2 only logs in
    for (const [k, code] of codes.slice(0, 2).entries()) {
      const c = await ready(t.base, code, '10.8.0.' + k, k === 0 ? 'en' : 'vi');
      ids.push(c.raterId);
      const plan = (await c.get('/api/plan')).data;
      const blocks = k === 0 ? ['practice', 'mos', 'smos'] : ['practice'];
      for (const b of blocks) {
        for (let i = 0; i < plan.session[b].length; i++) {
          await c.post('/api/answer', {...answerBody(plan, b, i, 1 + (i % 5)), lang: k === 0 && b === 'smos' ? 'en' : 'vi'});
        }
      }
      if (k === 1) for (let i = 0; i < 3; i++) await c.post('/api/answer', answerBody(plan, 'mos', i));
      if (k === 0) {
        await c.post('/api/background', BG);
        await c.post('/api/finish', {});
      }
    }
    const last = await new Client(t.base, {'CF-Connecting-IP': '10.8.0.9'}).post('/api/login', {code: codes[2]});
    ids.push(last.data.results.rater_id);
  });
  afterAll(() => t.stop());

  test('the raters list shows rater IDs, route, half, consent, sound check, progress and completion, no code', async () => {
    const r = await t.admin.get('/api/admin/raters');
    for (const code of codes) expect(r.text).not.toContain(code);
    const by = Object.fromEntries(r.data.raters.map((x: any) => [x.rater_id, x]));
    const a = by[ids[0]], b = by[ids[1]], c = by[ids[2]];
    expect(ids).toEqual(['R001', 'R002', 'R003']);
    expect([a.route, a.half, b.half, c.half]).toEqual(['invite', 'A', 'B', 'A']);
    expect(a.consent_at).not.toBeNull();
    expect(a.consent_lang).toBe('en');
    expect(a.headphone).toEqual({attempts: 1, passed: true, failed: false});
    expect(a.progress.mos.answered).toBe(a.progress.mos.total);
    expect(a.finished).not.toBeNull();
    expect(a.completed_at).not.toBeNull();
    expect(b.progress.practice).toEqual({answered: 3, total: 3});
    expect(b.progress.mos.answered).toBe(3);
    expect(b.completed_at).toBeNull();
    expect(c.consent_at).toBeNull();
    expect(c.first_used).not.toBeNull();
    expect(c.headphone).toBeNull();   // nothing stored before the consent
    expect(Object.keys(a)).not.toContain('ip');
    // progress per half
    expect(r.data.halves).toEqual({A: {issued: 2, consented: 1, finished: 1, completed: 1, headphone_failed: 0},
                                   B: {issued: 1, consented: 1, finished: 0, completed: 0, headphone_failed: 0}});
  });

  test('per-rater results JSON is the journal page\'s results file, keyed by the rater ID, with the §11.3 fields', async () => {
    const r = await t.admin.get(`/api/admin/export/rater?rater=${ids[0]}`);
    expect(r.headers.get('content-disposition')).toBe(`attachment; filename="listening_${ids[0]}.json"`);
    const x = r.data;
    expect(Object.keys(x)).toEqual(['version', 'rater_id', 'route', 'parity', 'half', 'started', 'finished', 'completed',
                                    'user_agent', 'config_title', 'consent', 'headphone', 'responses', 'block_ms',
                                    'background']);
    expect(x.version).toBe(1);
    expect(x.rater_id).toBe('R001');
    expect(x.route).toBe('invite');
    expect([x.half, x.parity]).toEqual(['A', 0]);
    expect(x.config_title).toBe('Listening test: MimiFlow-VC');
    expect(x.user_agent).toBe('Other');   // the test client sends no browser user agent
    expect(Object.keys(x.consent)).toEqual(['at', 'lang', 'sha256']);
    expect(x.consent.lang).toBe('en');
    expect(x.started).toBe(x.consent.at);
    expect(Object.keys(x.headphone)).toEqual(['attempts', 'passed']);
    expect(Object.keys(x.headphone.attempts[0])).toEqual(['trials', 'correct']);
    expect(Object.keys(x.headphone.attempts[0].trials[0])).toEqual(['order', 'answer', 'choice']);
    expect(Object.keys(x.responses)).toEqual(['practice', 'mos', 'smos', 'cmos']);
    const main = (b: string) => x.responses[b].find((m: any) => !('attention' in m));
    expect(Object.keys(main('mos'))).toEqual(['tid', 'stim', 'rating', 'plays', 'rt_ms', 't', 'lang']);
    expect(Object.keys(main('smos'))).toEqual(['tid', 'stim', 'ref', 'rating', 'plays_ref', 'plays_stim', 'rt_ms', 't',
                                               'lang']);
    expect(x.responses.mos.every((m: any) => m.lang === 'vi')).toBe(true);
    expect(x.responses.smos.every((m: any) => m.lang === 'en')).toBe(true);
    const att = x.responses.mos.find((m: any) => 'attention' in m);
    expect(Object.keys(att)).toEqual(['tid', 'stim', 'rating', 'plays', 'rt_ms', 't', 'lang', 'attention']);
    expect(x.block_ms).toEqual({practice: 3 * 5200, mos: 22 * 5200, smos: 26 * 5200, cmos: 0});
    expect(x.background).toEqual(BG);
    expect(new Date(x.started).getTime()).toBeLessThanOrEqual(new Date(x.finished).getTime());
    expect(x.finished).toBe(x.responses.smos[x.responses.smos.length - 1].t);
    expect(x.completed).not.toBeNull();
    // presentation order: the order of the rater's plan
    const c = new Client(t.base, {'CF-Connecting-IP': '10.8.1.0'});
    await c.post('/api/login', {code: codes[0]});
    const plan = (await c.get('/api/plan')).data;
    for (const b of ['practice', 'mos', 'smos']) {
      expect(x.responses[b].map((m: any) => m.tid)).toEqual(plan.session[b].map((m: any) => m.tid));
    }
    for (const q of ['rater=R999', 'rater=../x', `code=${codes[0]}`, `rater=${ids[2]}`]) {
      expect((await t.admin.get(`/api/admin/export/rater?${q}`)).status, q).toBe(404);
    }
  });

  test('answers.csv: one row per answer, fixed columns with the route and the page language, no code', async () => {
    const r = await t.admin.get('/api/admin/export/answers.csv');
    expect(r.headers.get('content-type')).toBe('text/csv; charset=utf-8');
    const lines = r.text.trim().split('\n');
    expect(lines[0]).toBe(CSV_COLUMNS.join(','));
    expect(CSV_COLUMNS.slice(0, 3)).toEqual(['package_id', 'rater_id', 'route']);
    expect(CSV_COLUMNS).toContain('lang');
    expect(lines.length - 1).toBe(3 + 22 + 26 + 3 + 3);
    const row = Object.fromEntries(lines[0].split(',').map((k, i) => [k, lines[1].split(',')[i]]));
    expect(row.rater_id).toBe('R001');
    expect(row.route).toBe('invite');
    expect(row.block).toBe('practice');
    expect(row.lang).toBe('vi');
    expect(row.played_s).toBe('3.25');
    expect(row.t_ms).toBe('5200');
    expect(leaks(r.text)).toEqual([]);
    for (const code of codes) expect(r.text).not.toContain(code);
    const only = await t.admin.get('/api/admin/export/answers.csv?completed_only=1');
    expect(only.text.trim().split('\n').length - 1).toBe(3 + 22 + 26);
  });

  test('the export folder holds one results file per consented rater, by rater ID, and the CSV', async () => {
    const r = await t.admin.post('/api/admin/export', {});
    expect(r.status).toBe(200);
    expect(r.data.raters).toBe(2);   // the rater who only logged in has not consented
    expect(r.data.dir.startsWith(join(t.dataDir, 'export'))).toBe(true);
    const files = readdirSync(join(r.data.dir, 'results')).sort();
    expect(files).toEqual(['listening_R001.json', 'listening_R002.json']);
    expect(existsSync(join(r.data.dir, 'answers.csv'))).toBe(true);
    const x = JSON.parse(readFileSync(join(r.data.dir, 'results', 'listening_R001.json'), 'utf8'));
    expect(x).toEqual((await t.admin.get('/api/admin/export/rater?rater=R001')).data);
    const csv = readFileSync(join(r.data.dir, 'answers.csv'), 'utf8');
    expect(csv).not.toContain(KEY_MARKER);
    for (const code of codes) expect(csv + JSON.stringify(x)).not.toContain(code);
    const done = await t.admin.post('/api/admin/export', {completed_only: true});
    expect(done.data.raters).toBe(1);
  });

  test('benchmark/listening_analysis.py reads the export with the builder-style key (gold_src a key-only kind)', async () => {
    const py = process.env.MOS_TEST_PYTHON || join(REPO, '.venv-stats', 'bin', 'python');
    if (!existsSync(py) || !existsSync(join(REPO, 'benchmark', 'listening_analysis.py'))) {
      console.warn('listening_analysis check skipped: no ' + py);
      return;
    }
    const exp = (await t.admin.post('/api/admin/export', {})).data.dir;
    const keyPath = join(t.dataDir, 'private', 'key.json');
    const script = `
import json, sys
sys.path.insert(0, ${JSON.stringify(join(REPO, 'benchmark'))})
import listening_analysis as la
res = la.load_results(la.result_files(sys.argv[1]))
key = json.load(open(sys.argv[2]))
sc = la.screen(res, key)
keep = sorted(r for r, w in sc.items() if not w)
rat = la.ratings(res, key, keep, 'mos')
print(json.dumps({'raters': sorted(res), 'screen': sc, 'n_mos': len(rat), 'conds': sorted({c for _, _, c, _ in rat}),
                  'session': la.session_minutes(res, keep)['n'],
                  'gold_src': sum(1 for r in res.values() for x in r['responses']['smos']
                                  if key['trials'][x['tid']]['kind'] == 'gold_src')}))`;
    const p = Bun.spawnSync([py, '-I', '-c', script, join(exp, 'results'), keyPath], {stderr: 'pipe'});
    expect(p.exitCode, new TextDecoder().decode(p.stderr)).toBe(0);
    const out = JSON.parse(new TextDecoder().decode(p.stdout));
    expect(out.raters).toEqual(['R001', 'R002']);
    expect(Object.keys(out.screen).sort()).toEqual(['R001', 'R002']);
    expect(out.conds.every((c: string) => key.conditions.includes(c))).toBe(true);
    expect(out.n_mos).toBeGreaterThan(0);
    // R001 rated the 2 cross-gender items of half A with src: 2 gold_src trials, found through the key only
    expect(out.gold_src).toBe(2);
  });
});

describe('withdrawal', () => {
  test('a rater found by the code is deleted from the database and the export folders; the ID is not reused', async () => {
    const dir = tempDir();
    makePackage(dir);
    const t = await boot({dataDir: dir});
    try {
      await t.admin.post('/api/admin/package/import', {});
      const codes = await setup(t, 3);
      for (const [k, code] of codes.entries()) {
        const c = await ready(t.base, code, '10.14.0.' + k);
        const plan = (await c.get('/api/plan')).data;
        for (let i = 0; i < 3; i++) await c.post('/api/answer', answerBody(plan, 'practice', i));
      }
      const exp = (await t.admin.post('/api/admin/export', {})).data.dir;
      const found = await t.admin.post('/api/admin/raters/find', {code: codes[1].toLowerCase()});
      expect(found.data).toEqual({rater_id: 'R002', route: 'invite', half: 'B', disabled: false});
      for (const b of [{code: 'ZZZZZZZZZZZZ'}, {code: 'x'}]) expect((await t.admin.post('/api/admin/raters/find', b)).status).toBe(404);
      for (const b of [{}, {code: 1}, {rater_id: 'R002', code: codes[1]}, {rater_id: 'x'}]) {
        expect((await t.admin.post('/api/admin/raters/delete', b)).status, JSON.stringify(b)).toBe(400);
      }
      expect((await t.admin.post('/api/admin/raters/delete', {rater_id: 'R009'})).status).toBe(404);

      // two earlier daily copies that hold the rater, and one that cannot be read
      const hmac = codeHmac(secretOf(dir), normalizeCode(codes[1])!);
      const copies = ['mos-20261001.sqlite', 'mos-20261002.sqlite'].map((f) => join(dir, 'backup', f));
      for (const f of copies) t.app.store.backupTo(f);
      writeFileSync(join(dir, 'backup', 'mos-20261003.sqlite'), 'not a database');
      expect(dbBytes(dir).includes(hmac)).toBe(true);   // the database files hold the rater before the deletion
      expect(readFileSync(copies[0]).toString('latin1').includes(hmac)).toBe(true);

      const r = await t.admin.post('/api/admin/raters/delete', {code: codes[1]});
      expect(r.data).toEqual({ok: true, rater_id: 'R002', export_files: 2, backup_files: 3, wal_cleared: true});
      const count = (tb: string) => (t.app.store.db.query(`SELECT COUNT(*) AS n FROM ${tb} WHERE rater_id = 'R002'`).get() as any).n;
      for (const tb of ['raters', 'sessions', 'answers', 'auth']) expect(count(tb), tb).toBe(0);
      // nothing of the rater in the database files: the log is emptied, deleted rows are overwritten
      expect(statSync(join(dir, 'mos.sqlite-wal')).size).toBe(0);
      const left = dbBytes(dir);   // (booleans: a failure would print the whole file)
      expect([left.includes(hmac), left.includes('R002'), left.includes('R001')]).toEqual([false, false, true]);
      // the daily copies: the rater removed, the others kept; the unreadable copy deleted; no journal left
      for (const f of copies) {
        const bytes = readFileSync(f).toString('latin1');
        expect([bytes.includes(hmac), bytes.includes('R002')]).toEqual([false, false]);
        const copy = new Database(f, {readonly: true});
        expect((copy.query('SELECT rater_id FROM raters ORDER BY rater_id').all() as any[]).map((x) => x.rater_id))
          .toEqual(['R001', 'R003']);
        expect((copy.query("SELECT COUNT(*) AS n FROM answers WHERE rater_id = 'R001'").get() as any).n).toBe(3);
        copy.close();
      }
      const today = `mos-${new Date().toISOString().slice(0, 10).replace(/-/g, '')}.sqlite`;   // made at the start
      expect(readdirSync(join(dir, 'backup')).sort()).toEqual(['mos-20261001.sqlite', 'mos-20261002.sqlite', today]);
      expect(t.logs.join('\n')).toContain('could not be cleaned and was deleted');
      expect(readdirSync(join(exp, 'results')).sort()).toEqual(['listening_R001.json', 'listening_R003.json']);
      const csv = readFileSync(join(exp, 'answers.csv'), 'utf8').trim().split('\n');
      expect(csv.length).toBe(1 + 2 * 3);
      expect(csv.some((l) => l.includes(',R002,'))).toBe(false);
      expect((await new Client(t.base, {'CF-Connecting-IP': '10.14.1.1'}).post('/api/login', {code: codes[1]})).status).toBe(401);
      // the next rater gets a new ID, and a deletion by rater ID works too
      expect((await t.admin.post('/api/admin/codes', {n: 1})).data.codes[0].rater_id).toBe('R004');
      expect((await t.admin.post('/api/admin/raters/delete', {rater_id: 'R003'})).data).toMatchObject(
        {export_files: 2, backup_files: 2, wal_cleared: true});
      expect(t.app.store.raters().map((x) => x.rater_id)).toEqual(['R001', 'R004']);
      expect((await t.admin.get('/api/admin/raters')).data.raters.length).toBe(2);
    } finally {
      await t.stop();
    }
  });
});

describe('sample mode', () => {
  test('without a package the journal sample data run as TEST DATA, with synthetic audio', async () => {
    const t = await boot({sample: true});
    try {
      const info = (await new Client(t.base).get('/api/info')).data;
      expect(info).toMatchObject({open: true, sample: true});
      expect((await new Client(t.base).get('/api/consent')).data.html).toContain('TEST DATA');
      const [code] = (await t.admin.post('/api/admin/codes', {n: 1})).data.codes.map((c: any) => c.code);
      const c = await ready(t.base, code, '10.10.0.1');
      const plan = (await c.get('/api/plan')).data;
      expect(plan.sample).toBe(true);
      expect(plan.config.require_full_play).toBe(false);
      expect(plan.session.practice.length).toBe(3);
      const sid = plan.session.mos[0].stim;
      const res = await fetch(t.base + plan.stimuli[sid].src, {headers: {Cookie: c.cookie}});
      expect(res.headers.get('content-type')).toBe('audio/wav');
      const bytes = new Uint8Array(await res.arrayBuffer());
      expect(new TextDecoder().decode(bytes.subarray(0, 4))).toBe('RIFF');
      expect((await t.admin.get('/api/admin/status')).data.package.sample).toBe(true);
      // importing a real package replaces the sample, and a restart without the flag never shows it again
      makePackage(t.dataDir);
      expect((await t.admin.post('/api/admin/package/import', {})).status).toBe(200);
      expect((await new Client(t.base).get('/api/info')).data.sample).toBe(false);
    } finally {
      await t.stop();
    }
    const u = await boot({sample: true});
    await u.stop(true);
    const v = await boot({dataDir: u.dataDir});
    expect((await new Client(v.base).get('/api/info')).data).toMatchObject({open: false, sample: false});
    await v.stop();
  });
});

describe('dry-run builds', () => {
  test('a package marked dry_run by the builder is shown to raters as TEST DATA', async () => {
    const dir = tempDir();
    makePackage(dir, {dryRun: true});
    const t = await boot({dataDir: dir});
    try {
      expect((await t.admin.post('/api/admin/package/import', {})).status).toBe(200);
      const [code] = await setup(t, 1);
      expect((await new Client(t.base).get('/api/info')).data.sample).toBe(true);
      const c = await ready(t.base, code, '10.12.0.1');
      expect((await c.get('/api/plan')).data.sample).toBe(true);
      expect((await c.get('/api/state')).data.sample).toBe(true);
      expect((await t.admin.get('/api/admin/status')).data.package).toMatchObject({sample: false, dry_run: true});
    } finally {
      await t.stop();
    }
  });
});

describe('package replacement', () => {
  test('a package with answers is replaced only with force; its answers stay exportable', async () => {
    const dir = tempDir();
    makePackage(dir);
    const t = await boot({dataDir: dir});
    try {
      const first = (await t.admin.post('/api/admin/package/import', {})).data.package_id;
      const [code] = await setup(t, 1);
      const c = await ready(t.base, code, '10.11.0.1');
      const plan = (await c.get('/api/plan')).data;
      await c.post('/api/answer', answerBody(plan, 'practice', 0));
      // a new build: one practice clip fewer
      const p = join(dir, 'package', 'lt_data.json');
      const d = JSON.parse(readFileSync(p, 'utf8'));
      d.practice = d.practice.slice(0, 1);
      writeFileSync(p, JSON.stringify(d));
      let r = await t.admin.post('/api/admin/package/import', {});
      expect([r.status, r.data.error.code]).toEqual([409, 'answers_exist']);
      r = await t.admin.post('/api/admin/package/import', {force: true});
      expect(r.status).toBe(200);
      expect(r.data.package_id).not.toBe(first);
      // the rater starts over in the new package (a new consent, a new sound check), in the same half
      const st = (await c.get('/api/state')).data;
      expect(st.package_id).toBe(r.data.package_id);
      expect(st.consented).toBe(false);
      expect(st.results.half).toBe('A');
      const old = await t.admin.get(`/api/admin/export/rater?rater=${c.raterId}&package=${first}`);
      expect(old.data.responses.practice.length).toBe(1);
    } finally {
      await t.stop();
    }
  });
});

describe('a database of the first version', () => {
  // A first-version database: plain 10-character codes (50 bits), a package, settings, a session and an answer.
  function v1Database(dir: string, withCodes: boolean) {
    const {data: raw} = makePackage(dir);
    const data = sanitizeLtData(raw).data!;
    const db = new Database(join(dir, 'mos.sqlite'), {create: true});
    db.exec('PRAGMA journal_mode = WAL');
    db.exec(MIGRATIONS[0] as string);
    db.exec('PRAGMA user_version = 1');
    db.query("INSERT INTO packages VALUES ('pkg1', '2026-10-29T01:00:00Z', 0, ?)").run(JSON.stringify(data));
    db.query("INSERT INTO meta VALUES ('current_package', 'pkg1')").run();
    db.query("INSERT INTO meta VALUES ('contact', 'lab@example.org')").run();
    const old = ['VC-7K3M9QX2TA', 'R2D2C3P0XY'];
    if (withCodes) {
      db.query("INSERT INTO raters VALUES (?, '2026-10-29T01:00:00Z', 'invite', 'n', 0, NULL)").run(old[0]);
      db.query("INSERT INTO raters VALUES (?, '2026-10-29T02:00:00Z', 'open', NULL, 0, NULL)").run(old[1]);
      db.query(`INSERT INTO sessions VALUES (?, 'pkg1', '{}', '2026-10-29T01:05:00Z', 'ua', '2026-10-29T01:06:00Z', 'vi',
                'abc', '{"attempts":[],"passed":false}', NULL, NULL, NULL)`).run(old[0]);
      db.query(`INSERT INTO answers VALUES (?, 'pkg1', 'practice', 0, 't1', 4, '1', 2.5, 100, 200, '2026-10-29T01:07:00Z')`)
        .run(old[0]);
    }
    db.close();
    return old;
  }

  test('one that holds codes is refused: the server does not start and the file is left as it was', () => {
    const dir = tempDir();
    try {
      const old = v1Database(dir, true);
      let err: Error | null = null;
      try {
        createApp({dataDir: dir, log: () => {}, backup: false});
      } catch (e) {
        err = e as Error;
      }
      expect(err!.message).toMatch(/first version of this server with 2 access codes of 50 bits.*move mos\.sqlite, /);
      for (const code of old) expect(err!.message).not.toContain(code);
      const db = new Database(join(dir, 'mos.sqlite'), {readonly: true});
      expect((db.query('PRAGMA user_version').get() as any).user_version).toBe(1);
      expect((db.query('SELECT COUNT(*) AS n FROM raters').get() as any).n).toBe(2);
      expect((db.query('SELECT COUNT(*) AS n FROM answers').get() as any).n).toBe(1);
      db.close();
    } finally {
      rmSync(dir, {recursive: true, force: true});
    }
  });

  test('one without codes is converted, keeping its packages and settings; 10-character codes never sign in', async () => {
    const dir = tempDir();
    v1Database(dir, false);
    const t = await boot({dataDir: dir});
    try {
      const store = t.app.store;
      expect((store.db.query('PRAGMA user_version').get() as any).user_version).toBe(SCHEMA_VERSION);
      expect(SCHEMA_VERSION).toBe(2);
      expect(store.get('current_package')).toBe('pkg1');
      expect(store.getPackage('pkg1')).not.toBeNull();
      expect((await t.admin.get('/api/admin/status')).data).toMatchObject({contact: 'lab@example.org', package: {id: 'pkg1'}});
      await t.admin.post('/api/admin/consent', {html: CONSENT});
      const [code] = (await t.admin.post('/api/admin/codes', {n: 1})).data.codes;
      expect(code.rater_id).toBe('R001');
      // only 12-character bodies are codes: the first version's 10-character ones are refused like any wrong code
      expect(normalizeCode('VC-7K3M9QX2TA')).toBeNull();
      expect(normalizeCode('r2d2c3p0xy')).toBeNull();
      expect(normalizeCode(code.code + '0')).toBeNull();
      const c = new Client(t.base, {'CF-Connecting-IP': '10.17.0.1'});
      for (const bad of ['VC-7K3M9QX2TA', 'R2D2C3P0XY']) {
        expect((await c.post('/api/login', {code: bad})).data).toEqual({error: {code: 'invalid_code', message: 'this code is not valid'}});
      }
      expect((await c.post('/api/login', {code: code.code})).data.results.rater_id).toBe('R001');
    } finally {
      await t.stop();
    }
  });
});
