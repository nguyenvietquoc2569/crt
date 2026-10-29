// Rater API: login and resume, nothing stored before the consent (and nothing built or kept in memory for a rater
// who has not consented), the consent, the sound check, answers (validation, full plays, order, idempotence, long
// trials, page language), background (6 questions) and finish (which signs out), the login rate limits (per IPv4
// address or IPv6 /64), the limit on registrations from all clients, disabled codes, open registration (on by
// default; the code shown once), the host, origin and body checks (chunked bodies included).
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { fallbackError, RateLimiter } from './app';
import { C, ipKey, sha256 } from './core';
import { answerBody, boot, chunkedPost, Client, CONSENT, hpAnswers, makePackage, ready, setup, tempDir } from './test_util';

let t: Awaited<ReturnType<typeof boot>>;
let codes: string[];

beforeAll(async () => {
  const dir = tempDir();
  makePackage(dir);
  t = await boot({dataDir: dir, loginLimit: 5});
  expect((await t.admin.post('/api/admin/package/import', {})).status).toBe(200);
  codes = await setup(t, 18);
});
afterAll(() => t.stop());

const RESULT_KEYS = ['version', 'rater_id', 'route', 'parity', 'half', 'started', 'finished', 'completed', 'user_agent',
                     'config_title', 'consent', 'headphone', 'responses', 'block_ms', 'background'];
const BG = {age: '25–34', first_language: 'Vietnamese', english: 'Advanced', hearing: 'No', device: 'Prefer not to say',
            previous_test: 'Not sure'};

describe('login and resume', () => {
  test('unknown, malformed and disabled codes are refused with the same JSON error', async () => {
    const c = new Client(t.base, {'CF-Connecting-IP': '10.1.0.1'});
    // also 10-character bodies (the 50-bit codes of the first version) and one character too many
    for (const [k, code] of ['ZZZZZZZZZZZZ', 'abc', 42, '', codes[0].slice(0, -1), codes[0].slice(0, -2),
                             codes[0] + '0'].entries()) {
      const r = await new Client(t.base, {'CF-Connecting-IP': `10.1.0.1${k}`}).post('/api/login', {code});
      expect(r.status).toBe(401);
      expect(r.data).toEqual({error: {code: 'invalid_code', message: 'this code is not valid'}});
    }
    expect((await c.get('/api/state')).data).toEqual({auth: false});
    expect((await c.get('/api/plan')).status).toBe(401);
  });

  test('login sets a random HttpOnly Secure SameSite=Strict session cookie; the code is never sent back', async () => {
    const c = new Client(t.base, {'CF-Connecting-IP': '10.1.0.2'});
    const r = await c.post('/api/login', {code: codes[0].toLowerCase()});   // codes are case-insensitive
    expect(r.status).toBe(200);
    const cookie = r.headers.get('set-cookie')!;
    // no Max-Age: the cookie ends with the browser session; the sign-in itself ends after 24 h on the server
    expect(cookie).toMatch(/^mos_sid=[A-Za-z0-9_-]{43}; Path=\/; HttpOnly; SameSite=Strict; Secure$/);
    const auth = t.app.store.db.query('SELECT expires_at FROM auth WHERE token_hash = ?')
      .get(sha256(cookie.split(';')[0].slice('mos_sid='.length))) as {expires_at: string};
    expect(Math.abs(new Date(auth.expires_at).getTime() - Date.now() - 24 * 36e5)).toBeLessThan(60000);
    expect(r.text).not.toContain(codes[0]);
    expect(r.data.results.rater_id).toMatch(/^R\d{3}$/);
    expect(r.data.consented).toBe(false);
    expect(Object.keys(r.data.results)).toEqual(RESULT_KEYS);
    expect((await c.get('/api/state')).text).not.toContain(codes[0]);
    const again = await new Client(t.base, {'CF-Connecting-IP': '10.1.0.2'}).post('/api/login', {code: codes[0]});
    expect(again.headers.get('set-cookie')!.split(';')[0]).not.toBe(cookie.split(';')[0]);
  });

  test('Crockford look-alikes (O for 0, I and L for 1) and spaces are accepted', async () => {
    const code = codes[1];
    const typed = code.replace(/0/g, 'O').replace(/1/g, 'l').replace(/(.{4})/g, '$1 ');
    const r = await new Client(t.base, {'CF-Connecting-IP': '10.1.0.3'}).post('/api/login', {code: ` ${typed} `});
    expect(r.status).toBe(200);
    expect(r.data.results.rater_id).toBe((await t.admin.post('/api/admin/raters/find', {code})).data.rater_id);
  });

  test('a rater who returns on another device resumes at the first unanswered trial', async () => {
    const code = codes[2];
    const a = await ready(t.base, code, '10.1.0.4');
    const plan = (await a.get('/api/plan')).data;
    for (let i = 0; i < plan.session.practice.length; i++) {
      expect((await a.post('/api/answer', answerBody(plan, 'practice', i))).status).toBe(200);
    }
    for (let i = 0; i < 3; i++) expect((await a.post('/api/answer', answerBody(plan, 'mos', i, 2 + i))).status).toBe(200);

    const b = new Client(t.base, {'CF-Connecting-IP': '10.1.0.5'});   // a new device: no cookie, no local copy
    const st = (await b.post('/api/login', {code})).data;
    expect(st.consented).toBe(true);
    expect(st.results.headphone.passed).toBe(true);
    expect(st.results.responses.practice.length).toBe(3);
    expect(st.results.responses.mos.map((x: any) => x.tid)).toEqual(plan.session.mos.slice(0, 3).map((x: any) => x.tid));
    expect(st.results.responses.mos.map((x: any) => x.rating)).toEqual([2, 3, 4]);
    const plan2 = (await b.get('/api/plan')).data;
    expect(plan2).toEqual(plan);   // the same trials in the same order
    // the next trial is mos[3]: mos[4] is refused, mos[3] is taken
    expect((await b.post('/api/answer', answerBody(plan, 'mos', 4))).status).toBe(409);
    expect((await b.post('/api/answer', answerBody(plan, 'mos', 3))).status).toBe(200);
    expect((await a.get('/api/state')).data.results.responses.mos.length).toBe(4);
  });

  test('logout ends the session cookie', async () => {
    const c = new Client(t.base, {'CF-Connecting-IP': '10.1.0.6'});
    await c.post('/api/login', {code: codes[3]});
    const old = c.cookie;
    expect((await c.post('/api/logout', {all: true})).status).toBe(400);   // no fields
    const r = await c.post('/api/logout', {});
    expect(r.status).toBe(200);
    expect(r.headers.get('set-cookie')).toMatch(/^mos_sid=; .*SameSite=Strict; Max-Age=0/);
    const stale = new Client(t.base);
    stale.cookie = old;
    expect((await stale.get('/api/state')).data).toEqual({auth: false});
  });
});

describe('consent and sound check', () => {
  test('before the consent only the use of the code is stored: no session, no browser, no plan, no audio', async () => {
    const code = codes[15];
    const c = new Client(t.base, {'CF-Connecting-IP': '10.2.0.9', 'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) ' +
                                  'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.6723.91 Safari/537.36'});
    const st = (await c.post('/api/login', {code})).data;
    const id = st.results.rater_id;
    expect(st.consented).toBe(false);
    expect(st.results.started).toBeNull();
    const rater = t.app.store.rater(id)!;
    expect(rater.first_used).not.toBeNull();
    expect(rater.last_seen).not.toBeNull();
    const count = (q: string) => (t.app.store.db.query(q).get(id) as {n: number}).n;
    expect(count('SELECT COUNT(*) AS n FROM sessions WHERE rater_id = ?')).toBe(0);
    expect(count('SELECT COUNT(*) AS n FROM answers WHERE rater_id = ?')).toBe(0);
    let r = await c.get('/api/plan');
    expect([r.status, r.data.error.code]).toEqual([409, 'consent_required']);
    for (const [p, b] of [['/api/headphone', {attempt: 0, trial: 0, choice: 0}], ['/api/background', BG],
                          ['/api/finish', {}]] as const) {
      r = await c.post(p, b);
      expect([p, r.status, r.data.error.code]).toEqual([p, 409, 'consent_required']);
    }
    // the plan exists only from the consent on, and so does the audio: take a clip of another rater's plan
    const other = await ready(t.base, codes[16], '10.2.0.10');
    const sid = (await other.get('/api/plan')).data.session.mos[0].stim;
    r = await c.get(`/s/${sid}.wav`);
    expect(r.status).toBe(409);
    expect(count('SELECT COUNT(*) AS n FROM sessions WHERE rater_id = ?')).toBe(0);

    const consent = (await c.get('/api/consent')).data;
    expect((await c.post('/api/consent', {lang: 'en', sha256: consent.sha256})).data).toEqual({ok: true, duplicate: false});
    const s = t.app.store.session(id, st.package_id)!;
    expect(s.browser).toBe('Chrome 130');   // the browser family and major version only
    expect([s.consent_lang, s.consent_sha]).toEqual(['en', consent.sha256]);
    expect(JSON.stringify(t.app.store.db.query('SELECT * FROM sessions').all())).not.toContain('Mozilla');
    expect((await c.post('/api/consent', {lang: 'vi', sha256: consent.sha256})).data.duplicate).toBe(true);
    expect(t.app.store.session(id, st.package_id)!.consent_lang).toBe('en');   // the first agreement stays
    expect((await c.get('/api/plan')).status).toBe(200);
  });

  test('answers need consent and a passed sound check; the server scores the check itself', async () => {
    const code = codes[4];
    const c = new Client(t.base, {'CF-Connecting-IP': '10.2.0.1'});
    const id = (await c.post('/api/login', {code})).data.results.rater_id;
    let r = await c.post('/api/headphone', {attempt: 0, trial: 0, choice: 0});
    expect([r.status, r.data.error.code]).toEqual([409, 'consent_required']);
    const consent = (await c.get('/api/consent')).data;
    expect(consent).toEqual({html: CONSENT + '\n', sha256: sha256(CONSENT + '\n')});
    expect(consent.sha256).toBe(sha256(readFileSync(join(t.dataDir, 'consent_v1.html'))));
    r = await c.post('/api/consent', {lang: 'en', sha256: 'f'.repeat(64)});
    expect([r.status, r.data.error.code]).toEqual([409, 'consent_changed']);
    expect((await c.post('/api/consent', {lang: 'fr', sha256: consent.sha256})).status).toBe(400);
    expect((await c.post('/api/consent', {lang: 'en', sha256: consent.sha256})).status).toBe(200);
    const plan = (await c.get('/api/plan')).data;
    r = await c.post('/api/answer', answerBody(plan, 'practice', 0));
    expect([r.status, r.data.error.code]).toEqual([409, 'headphone_required']);

    // attempt 0: all wrong -> failed; attempt 1: all right -> passed. The tones follow from the rater ID.
    const w0 = hpAnswers(id, 0, true);
    for (let i = 0; i < 6; i++) {
      r = await c.post('/api/headphone', {attempt: 0, trial: i, choice: w0[i]});
      expect(r.status).toBe(200);
    }
    expect(r.data.headphone.attempts[0].correct).toBe(0);
    expect(r.data.headphone.passed).toBe(false);
    expect((await c.post('/api/headphone', {attempt: 0, trial: 2, choice: 0})).data.duplicate).toBe(true);
    expect((await c.post('/api/headphone', {attempt: 1, trial: 1, choice: 0})).status).toBe(409);   // skips trial 0
    const t1 = C.headphoneTrials(6, C.rng(id + ':headphone:1'));
    for (let i = 0; i < 6; i++) r = await c.post('/api/headphone', {attempt: 1, trial: i, choice: t1[i].answer});
    expect(r.data.headphone.passed).toBe(true);
    expect(r.data.headphone.attempts[1]).toEqual({trials: t1.map((x: any) => ({order: x.order, answer: x.answer,
                                                                               choice: x.answer})), correct: 6});
    for (const bad of [{attempt: 2, trial: 0, choice: 0}, {attempt: 0, trial: 0, choice: 3}, {attempt: '0', trial: 0, choice: 1}]) {
      expect((await c.post('/api/headphone', bad)).status).toBe(400);
    }
  });

  test('two failed attempts end the session, which /api/finish then records', async () => {
    const code = codes[5];
    const c = new Client(t.base, {'CF-Connecting-IP': '10.2.0.2'});
    const id = (await c.post('/api/login', {code})).data.results.rater_id;
    const consent = (await c.get('/api/consent')).data;
    await c.post('/api/consent', {lang: 'vi', sha256: consent.sha256});
    expect((await c.post('/api/finish', {})).status).toBe(409);
    for (const attempt of [0, 1]) {
      const wrong = hpAnswers(id, attempt, true);
      for (let i = 0; i < 6; i++) await c.post('/api/headphone', {attempt, trial: i, choice: wrong[i]});
    }
    expect((await c.post('/api/headphone', {attempt: 2, trial: 0, choice: 0})).status).toBe(400);
    const r = await c.post('/api/finish', {});
    expect(r.status).toBe(200);
    expect(r.headers.get('set-cookie')).toMatch(/^mos_sid=; .*Max-Age=0/);
    expect((await c.get('/api/state')).data).toEqual({auth: false});
    expect((await c.post('/api/login', {code})).data.completed).toBe(true);
  });
});

describe('answers', () => {
  test('validation: types, ranges, plays, language, unknown tids and fields', async () => {
    const c = await ready(t.base, codes[6], '10.3.0.1');
    const plan = (await c.get('/api/plan')).data;
    const good = answerBody(plan, 'practice', 0);
    expect(plan.config.require_full_play).toBe(true);
    let r = await c.post('/api/answer', {...good, plays: 0});   // the package asks for every clip to be played
    expect([r.status, r.data.error.code]).toEqual([400, 'not_played']);
    const bad: [string, Record<string, unknown>][] = [
      ['value 6', {...good, value: 6}], ['value 0', {...good, value: 0}], ['value 2.5', {...good, value: 2.5}],
      ['value "3"', {...good, value: '3'}], ['no value', {...good, value: undefined}],
      ['unknown block', {...good, block: 'abx'}], ['block of another tid', {...good, block: 'mos'}],
      ['unknown tid', {...good, tid: 'nope123'}], ['tid with a slash', {...good, tid: '../x'}],
      ['plays object for one clip', {...good, plays: {stim: 1}}], ['plays -1', {...good, plays: -1}],
      ['played_s negative', {...good, played_s: -1}], ['rt_ms float', {...good, rt_ms: 1.5}],
      ['extra field', {...good, condition: 'x'}], ['no lang', {...good, lang: undefined}],
      ['lang fr', {...good, lang: 'fr'}], ['lang upper case', {...good, lang: 'VI'}],
    ];
    for (const [why, b] of bad) {
      r = await c.post('/api/answer', b);
      expect(r.status, why).toBe(400);
      expect(typeof r.data.error.message, why).toBe('string');
      expect(r.text, why).not.toMatch(/at .*\.ts|Error:/);   // no stack traces
    }
    expect((await c.post('/api/answer', good)).status).toBe(200);
    // SMOS needs {ref, stim}
    for (let i = 1; i < plan.session.practice.length; i++) await c.post('/api/answer', answerBody(plan, 'practice', i));
    for (let i = 0; i < plan.session.mos.length; i++) {
      expect((await c.post('/api/answer', answerBody(plan, 'mos', i))).status).toBe(200);
    }
    const s0 = answerBody(plan, 'smos', 0);
    for (const plays of [{ref: 0, stim: 1}, {ref: 1, stim: 0}]) {
      r = await c.post('/api/answer', {...s0, plays});
      expect([r.status, r.data.error.code]).toEqual([400, 'not_played']);
    }
    expect((await c.post('/api/answer', {...s0, plays: 1})).status).toBe(400);
    expect((await c.post('/api/answer', {...s0, plays: {ref: 1}})).status).toBe(400);
    expect((await c.post('/api/answer', {...s0, plays: {ref: 1, stim: 1, a: 1}})).status).toBe(400);
    expect((await c.post('/api/answer', s0)).status).toBe(200);
  });

  test('a repeated answer is acknowledged and the first answer stays; the page language is stored per answer', async () => {
    const c = await ready(t.base, codes[7], '10.3.0.2');
    const plan = (await c.get('/api/plan')).data;
    const first = {...answerBody(plan, 'practice', 0, 5), lang: 'en'};
    let r = await c.post('/api/answer', first);
    expect(r.data).toEqual({ok: true, finished: false});
    r = await c.post('/api/answer', first);
    expect(r.data).toEqual({ok: true, duplicate: true});
    r = await c.post('/api/answer', {...first, value: 1, plays: 7, lang: 'vi'});
    expect(r.data).toEqual({ok: true, duplicate: true});
    await c.post('/api/answer', answerBody(plan, 'practice', 1));
    const st = (await c.get('/api/state')).data;
    expect(st.results.responses.practice[0]).toEqual(
      {tid: first.tid, stim: plan.session.practice[0].stim, rating: 5, plays: 1, rt_ms: 4100,
       t: st.results.responses.practice[0].t, lang: 'en'});
    expect(st.results.responses.practice[1].lang).toBe('vi');
    expect(st.results.block_ms.practice).toBe(2 * 5200);
  });

  test('a trial left open for over 24 h is taken, with rt_ms and t_ms stored capped at 24 h', async () => {
    const c = await ready(t.base, codes[12], '10.3.0.5');
    const plan = (await c.get('/api/plan')).data;
    const b = answerBody(plan, 'practice', 0);
    for (const bad of [{...b, t_ms: -1}, {...b, t_ms: 1.5}, {...b, rt_ms: '9'}, {...b, t_ms: 2 ** 60}]) {
      expect((await c.post('/api/answer', bad)).status).toBe(400);
    }
    expect((await c.post('/api/answer', {...b, rt_ms: 9e7, t_ms: 25 * 36e5})).data).toEqual({ok: true, finished: false});
    const a = t.app.store.answers(c.raterId, plan.package_id)[0];
    expect([a.rt_ms, a.t_ms]).toEqual([864e5, 864e5]);
    expect((await c.post('/api/answer', {...answerBody(plan, 'practice', 1), rt_ms: null, t_ms: 100})).status).toBe(200);
  });

  test('a full session: finished at the last rating, then six background answers, and finish signs out everywhere', async () => {
    const code = codes[8];
    const c = await ready(t.base, code, '10.3.0.3');
    const other = new Client(t.base, {'CF-Connecting-IP': '10.3.0.4'});   // the same code open on a second device
    expect((await other.post('/api/login', {code})).status).toBe(200);
    const plan = (await c.get('/api/plan')).data;
    let last: any;
    for (const block of ['practice', 'mos', 'smos', 'cmos']) {
      for (let i = 0; i < plan.session[block].length; i++) {
        if (block === 'smos' && i === 0) {
          expect((await c.post('/api/background', BG)).status).toBe(409);   // not before the last rating
        }
        last = await c.post('/api/answer', answerBody(plan, block, i));
        expect(last.status).toBe(200);
      }
    }
    expect(last.data.finished).toBe(true);
    let st = (await c.get('/api/state')).data;
    const lastT = st.results.responses.smos[st.results.responses.smos.length - 1].t;
    expect(st.results.finished).toBe(lastT);
    expect((await c.post('/api/finish', {})).status).toBe(409);   // background first
    expect((await c.post('/api/background', {...BG, age: '17'})).status).toBe(400);
    expect((await c.post('/api/background', {...BG, extra: 'x'})).status).toBe(400);
    const {previous_test, ...five} = BG;
    expect((await c.post('/api/background', five)).status).toBe(400);   // question 6 is asked too
    expect((await c.post('/api/background', {...BG, previous_test: 'Maybe'})).status).toBe(400);
    expect((await c.post('/api/background', BG)).status).toBe(200);
    expect((await c.post('/api/background', {...BG, age: '18–24'})).data.duplicate).toBe(true);
    const fin = await c.post('/api/finish', {});
    expect(fin.status).toBe(200);
    // the next person at either computer is not signed in as this rater
    expect(fin.headers.get('set-cookie')).toMatch(/^mos_sid=; Path=\/; HttpOnly; SameSite=Strict; Max-Age=0; Secure$/);
    expect((await c.get('/api/state')).data).toEqual({auth: false});
    expect((await other.get('/api/state')).data).toEqual({auth: false});
    expect((await c.post('/api/finish', {})).status).toBe(401);
    expect((await c.post('/api/login', {code})).status).toBe(200);   // the code itself still works
    st = (await c.get('/api/state')).data;
    expect(st.completed).toBe(true);
    expect(st.results.background).toEqual(BG);
    expect(st.results.completed).not.toBeNull();
    // attention trials carry the rating they ask for in the results: 2 in MOS (2 and 4), 2 in SMOS (4 and 2)
    for (const block of ['mos', 'smos']) {
      const att = st.results.responses[block].filter((x: any) => 'attention' in x);
      expect(att.map((x: any) => x.attention).sort()).toEqual([2, 4]);
      expect(att.every((x: any) => x.rating === x.attention)).toBe(true);
    }
  });
});

describe('request checks', () => {
  test('body limit, content type, malformed JSON, non-object bodies, cross-origin POST', async () => {
    const c = await ready(t.base, codes[9], '10.4.0.1');
    let r = await c.post('/api/answer', JSON.stringify({tid: 'x'.repeat(17000)}));
    expect([r.status, r.data.error.code]).toEqual([413, 'too_large']);
    r = await c.post('/api/answer', '{"tid": 1}', {'Content-Type': 'text/plain'});
    expect([r.status, r.data.error.code]).toEqual([415, 'unsupported_media_type']);
    r = await c.post('/api/answer', '{"tid": ');
    expect([r.status, r.data.error.code]).toEqual([400, 'bad_json']);
    r = await c.post('/api/answer', '[1, 2]');
    expect(r.status).toBe(400);
    r = await c.post('/api/login', {code: codes[9]}, {Origin: 'https://evil.example'});
    expect([r.status, r.data.error.code]).toEqual([403, 'bad_origin']);
    r = await c.post('/api/login', {code: codes[9]}, {Origin: 'null'});
    expect(r.status).toBe(403);
    r = await c.post('/api/login', {code: codes[9]}, {Origin: t.base});
    expect(r.status).toBe(200);
    expect((await c.get('/api/nothing')).status).toBe(404);
    expect((await c.req('PUT', '/api/answer', {})).status).toBe(404);
  });

  test('a chunked body is refused once it passes the limit, without being read whole', async () => {
    // Bun's maxRequestBodySize checks only Content-Length; this body has none and never ends
    let r = await chunkedPost(t.base, '/api/login', 32 * 1024 * 1024, {'CF-Connecting-IP': '10.4.1.1'});
    expect([r.status, r.data?.error?.code]).toEqual([413, 'too_large']);
    expect(r.written).toBeLessThan(8 * 1024 * 1024);
    const c = await ready(t.base, codes[13], '10.4.1.2');
    r = await chunkedPost(t.base, '/api/answer', 1024 * 1024, {Cookie: c.cookie});
    expect(r.status).toBe(413);
    // a small chunked body is read as usual (a proxy may forward bodies chunked)
    r = await chunkedPost(t.base, '/api/login', JSON.stringify({code: codes[13]}), {'CF-Connecting-IP': '10.4.1.3'});
    expect([r.status, r.data.results.rater_id]).toEqual([200, c.raterId]);
    r = await chunkedPost(t.base, '/api/login', '{"code": ', {'CF-Connecting-IP': '10.4.1.3'});
    expect(r.status).toBe(400);
  });

  test('the login limit counts failed attempts per IP and answers 429 as JSON with the time left', async () => {
    // successful logins do not count: raters of a lab session share one public IP (limit 5 here)
    const lab = new Client(t.base, {'CF-Connecting-IP': '10.5.0.3'});
    for (let i = 0; i < 12; i++) expect((await lab.post('/api/login', {code: codes[i % 4]})).status).toBe(200);
    const c = new Client(t.base, {'CF-Connecting-IP': '10.5.0.1'});
    for (let i = 0; i < 5; i++) expect((await c.post('/api/login', {code: 'ZZZZZZZZZZZZ'})).status).toBe(401);
    const r = await c.post('/api/login', {code: codes[10]});   // even a valid code, once over the limit
    expect(r.status).toBe(429);
    expect(r.data.error.code).toBe('rate_limited');
    const wait = Number(r.headers.get('retry-after'));
    expect(wait > 590 && wait <= 600).toBe(true);
    expect((await lab.post('/api/login', {code: codes[10]})).status).toBe(200);
    expect((await new Client(t.base, {'CF-Connecting-IP': '10.5.0.2'}).post('/api/login', {code: codes[10]})).status).toBe(200);
    // no IP address and no access code is stored anywhere in the database
    const dump = JSON.stringify(t.app.store.db.query('SELECT * FROM sqlite_master').all()) +
      ['meta', 'raters', 'sessions', 'answers', 'auth', 'packages'].map((tb) => JSON.stringify(t.app.store.db.query(`SELECT * FROM ${tb}`).all())).join();
    expect(dump).not.toContain('10.5.0.1');
    expect(dump).not.toContain('10.1.0.2');
    for (const code of codes) expect(dump).not.toContain(code);
  });

  test('IPv6 clients are counted per /64, IPv4-mapped addresses as IPv4', async () => {
    expect(ipKey('2001:db8:1:2::1')).toBe(ipKey('2001:DB8:1:2:ffff:ffff:ffff:ffff'));
    expect(ipKey('2001:db8:1:2::1')).not.toBe(ipKey('2001:db8:1:3::1'));
    expect(ipKey('::ffff:10.20.0.1')).toBe('10.20.0.1');
    expect(ipKey('[2001:db8:1:2::7]')).toBe('2001:db8:1:2::/64');
    // five failures from five addresses of one /64 (limit 5 here) lock the whole /64, not its neighbour
    for (let i = 1; i <= 5; i++) {
      const r = await new Client(t.base, {'CF-Connecting-IP': `2001:db8:1:2::${i}`}).post('/api/login', {code: 'ZZZZZZZZZZZZ'});
      expect(r.status).toBe(401);
    }
    expect((await new Client(t.base, {'CF-Connecting-IP': '2001:db8:1:2:abcd::99'}).post('/api/login', {code: codes[10]})).status).toBe(429);
    expect((await new Client(t.base, {'CF-Connecting-IP': '2001:db8:1:3::1'}).post('/api/login', {code: codes[10]})).status).toBe(200);
    for (let i = 1; i <= 5; i++) {
      await new Client(t.base, {'CF-Connecting-IP': i % 2 ? '10.20.0.1' : '::ffff:10.20.0.1'}).post('/api/login', {code: 'ZZZZZZZZZZZZ'});
    }
    expect((await new Client(t.base, {'CF-Connecting-IP': '10.20.0.1'}).post('/api/login', {code: codes[10]})).status).toBe(429);
  });

  test('the limiter: checks add no entry, and expired entries are dropped', async () => {
    const l = new RateLimiter(2, 50);
    expect(Array.from({length: 20000}, (_, i) => l.wait('k' + i)).every((w) => w === 0)).toBe(true);
    expect(l.size).toBe(0);   // a flood of first requests from new addresses keeps nothing
    l.hit('a');
    l.hit('a');
    expect(l.wait('a')).toBe(1);
    for (let i = 0; i < 10001; i++) l.hit('k' + i);
    expect(l.size).toBe(10002);
    await new Promise((res) => setTimeout(res, 60));
    expect(l.wait('a')).toBe(0);   // the window is over
    l.hit('b');
    expect(l.size).toBe(1);
  });

  test('all logins and registrations of one IP are capped at 10 times the failure limit', async () => {
    const c = new Client(t.base, {'CF-Connecting-IP': '10.5.0.4'});
    for (let i = 0; i < 50; i++) expect((await c.post('/api/login', {code: codes[14]})).status).toBe(200);
    expect((await c.post('/api/login', {code: codes[14]})).status).toBe(429);
    expect((await c.post('/api/register', {})).status).toBe(429);
  });

  test('the limits reset after the window, and Retry-After gives the time left', async () => {
    const dir = tempDir();
    makePackage(dir);
    const u = await boot({dataDir: dir, loginLimit: 2, loginWindowMs: 700});
    try {
      await u.admin.post('/api/admin/package/import', {});
      const [code] = await setup(u, 1);
      const c = new Client(u.base, {'CF-Connecting-IP': '10.5.1.1'});
      for (let i = 0; i < 2; i++) expect((await c.post('/api/login', {code: 'ZZZZZZZZZZZZ'})).status).toBe(401);
      const r = await c.post('/api/login', {code});
      expect([r.status, r.headers.get('retry-after')]).toEqual([429, '1']);
      await new Promise((res) => setTimeout(res, 750));
      expect((await c.post('/api/login', {code})).status).toBe(200);
    } finally {
      await u.stop();
    }
  });

  test('registrations from all clients together are limited, so rotating addresses cannot flood the server', async () => {
    const dir = tempDir();
    makePackage(dir);
    const u = await boot({dataDir: dir, registerLimit: 3, loginWindowMs: 700});
    try {
      await u.admin.post('/api/admin/package/import', {});
      await u.admin.post('/api/admin/consent', {html: CONSENT});
      const reg = (ip: string) => new Client(u.base, {'CF-Connecting-IP': ip}).post('/api/register', {});
      for (let i = 0; i < 3; i++) expect((await reg(`2001:db8:${i}::1`)).status).toBe(200);
      const r = await reg('2001:db8:99::1');   // a fresh address, over the limit of all clients
      expect([r.status, r.data.error.code, r.headers.get('retry-after')]).toEqual([429, 'rate_limited', '1']);
      expect(u.app.store.raters().length).toBe(3);
      // logins are not limited by it
      expect((await new Client(u.base).post('/api/login', {code: 'ZZZZZZZZZZZZ'})).status).toBe(401);
      await new Promise((res) => setTimeout(res, 750));
      expect((await reg('2001:db8:99::1')).status).toBe(200);
    } finally {
      await u.stop();
    }
  });

  test('the defaults: 100 failed logins per address, 30 registrations from all clients, per 10 minutes', async () => {
    const dir = tempDir();
    makePackage(dir);
    const u = await boot({dataDir: dir});
    try {
      await u.admin.post('/api/admin/package/import', {});
      await u.admin.post('/api/admin/consent', {html: CONSENT});
      await u.admin.post('/api/admin/settings', {max_codes: 100});
      const c = new Client(u.base, {'CF-Connecting-IP': '10.21.0.1'});
      for (let i = 0; i < 100; i++) expect((await c.post('/api/login', {code: 'ZZZZZZZZZZZZ'})).status).toBe(401);
      expect((await c.post('/api/login', {code: 'ZZZZZZZZZZZZ'})).status).toBe(429);
      for (let i = 0; i < 30; i++) {
        expect((await new Client(u.base, {'CF-Connecting-IP': `10.21.1.${i}`}).post('/api/register', {})).status).toBe(200);
      }
      const r = await new Client(u.base, {'CF-Connecting-IP': '10.21.2.1'}).post('/api/register', {});
      expect(r.status).toBe(429);
      expect(Number(r.headers.get('retry-after'))).toBeGreaterThan(590);
    } finally {
      await u.stop();
    }
  });

  test('nothing is built or kept for a rater before the consent: registrations cost a database row only', async () => {
    const dir = tempDir();
    makePackage(dir);
    const u = await boot({dataDir: dir, registerLimit: 1000});
    const build = C.buildSession;
    let built = 0;
    C.buildSession = (...a: unknown[]) => { built++; return build(...a); };
    try {
      await u.admin.post('/api/admin/package/import', {});
      await u.admin.post('/api/admin/consent', {html: CONSENT});
      await u.admin.post('/api/admin/settings', {max_codes: 1000});
      let last: Client | null = null;
      for (let i = 0; i < 300; i++) {
        last = new Client(u.base, {'CF-Connecting-IP': `2001:db8:${i.toString(16)}::1`});
        const r = await last.post('/api/register', {});
        expect(r.status).toBe(200);
        expect(r.data.results.responses).toEqual({practice: [], mos: [], smos: [], cmos: []});
        expect((await last.get('/api/state')).data.consented).toBe(false);
      }
      expect(built).toBe(0);
      // the state before the consent shows the rater's half as the session will have it
      const st = (await last!.get('/api/state')).data.results;
      expect([st.half, st.parity, st.config_title]).toEqual([u.app.store.rater(st.rater_id)!.half,
                                                             st.half === 'A' ? 0 : 1, 'Listening test: MimiFlow-VC']);
      // the consent builds the session once and stores it; the plan and the audio read the stored one
      await last!.post('/api/consent', {lang: 'vi', sha256: (await last!.get('/api/consent')).data.sha256});
      expect(built).toBe(1);
      const plan = (await last!.get('/api/plan')).data;
      expect([plan.session.raterId, plan.session.mos.length > 0]).toEqual([st.rater_id, true]);
      await last!.get('/api/plan');
      expect((await fetch(u.base + plan.stimuli[plan.session.mos[0].stim].src, {headers: {Cookie: last!.cookie}})).status).toBe(200);
      expect(built).toBe(1);
    } finally {
      C.buildSession = build;
      await u.stop();
    }
  });

  test('only the configured host names and the loopback addresses are served (DNS rebinding)', async () => {
    const dir = tempDir();
    makePackage(dir);
    const u = await boot({dataDir: dir, hosts: ['mos.betaschool.edu.vn']});
    try {
      await u.admin.post('/api/admin/package/import', {});
      await u.admin.post('/api/admin/consent', {html: CONSENT});
      const port = new URL(u.base).port;
      const as = (host: string, path = '/api/info', init: RequestInit = {}) =>
        fetch(u.base + path, {...init, headers: {...(init.headers as Record<string, string> ?? {}), Host: host}});
      for (const host of ['mos.betaschool.edu.vn', 'MOS.BetaSchool.edu.vn:443', `127.0.0.1:${port}`, `localhost:${port}`,
                          `[::1]:${port}`]) {
        expect((await as(host)).status, host).toBe(200);
      }
      for (const host of ['evil.example', `evil.example:${port}`, 'mos.betaschool.edu.vn.evil.example', '10.0.0.1',
                          `127.0.0.2:${port}`]) {
        for (const path of ['/', '/admin', '/api/info', '/s/abcdef.wav']) {
          const r = await as(host, path);
          expect(r.status, `${host} ${path}`).toBe(421);
          expect(r.headers.get('content-security-policy'), host).toContain("default-src 'self'");
        }
      }
      // a rebound page cannot register raters, even with a matching Origin
      const r = await as(`evil.example:${port}`, '/api/register', {method: 'POST', body: '{}', headers: {
        'Content-Type': 'application/json', Origin: `http://evil.example:${port}`}});
      expect([r.status, (await r.json()).error.code]).toEqual([421, 'bad_host']);
      expect(u.app.store.raters().length).toBe(0);
      // the tunnel's host name works end to end
      const ok = await as('mos.betaschool.edu.vn', '/api/register', {method: 'POST', body: '{}', headers: {
        'Content-Type': 'application/json', Origin: 'https://mos.betaschool.edu.vn'}});
      expect(ok.status).toBe(200);
      // without configured names only the loopback addresses are served
      expect((await fetch(t.base + '/api/info', {headers: {Host: 'mos.betaschool.edu.vn'}})).status).toBe(421);
    } finally {
      await u.stop();
    }
  });

  test('Bun\'s last-resort error answer carries the security headers too', async () => {
    const r = fallbackError();
    expect(r.status).toBe(500);
    for (const h of ['content-security-policy', 'x-content-type-options', 'x-frame-options', 'referrer-policy',
                     'strict-transport-security', 'cross-origin-resource-policy']) {
      expect(r.headers.get(h), h).toBeTruthy();
    }
    expect(await r.text()).toBe('internal error\n');
  });

  test('a disabled code can no longer log in, and its open sessions end', async () => {
    const code = codes[11];
    const c = new Client(t.base, {'CF-Connecting-IP': '10.6.0.1'});
    const id = (await c.post('/api/login', {code})).data.results.rater_id;
    expect((await t.admin.post('/api/admin/raters/disable', {rater_id: id, disabled: true})).status).toBe(200);
    expect((await c.get('/api/state')).data).toEqual({auth: false});
    expect((await c.post('/api/login', {code})).status).toBe(401);
    expect((await t.admin.post('/api/admin/raters/disable', {rater_id: id, disabled: false})).status).toBe(200);
    expect((await c.post('/api/login', {code})).status).toBe(200);
  });

  test('open registration: on by default; the new code is shown once and resumes like an invite', async () => {
    const c = new Client(t.base, {'CF-Connecting-IP': '10.7.0.1'});
    expect((await c.get('/api/info')).data.open_registration).toBe(true);
    expect((await c.post('/api/register', {code: 'X'})).status).toBe(400);   // no fields
    const r = await c.post('/api/register', {});
    expect(r.status).toBe(200);
    expect(r.data.code).toMatch(/^[0-9A-HJKMNP-TV-Z]{12}$/);
    expect(r.data.consented).toBe(false);
    const id = r.data.results.rater_id;
    // after this answer the code is never sent again: not in the state, not in the admin list
    expect((await c.get('/api/state')).text).not.toContain(r.data.code);
    const list = await t.admin.get('/api/admin/raters');
    expect(list.text).not.toContain(r.data.code);
    expect(list.data.raters.find((x: any) => x.rater_id === id).route).toBe('open');
    const again = await new Client(t.base, {'CF-Connecting-IP': '10.7.0.2'}).post('/api/login', {code: r.data.code});
    expect([again.status, again.data.results.rater_id, again.data.code]).toEqual([200, id, undefined]);
    expect((await t.admin.post('/api/admin/settings', {open_registration: false})).status).toBe(200);
    expect((await c.get('/api/info')).data.open_registration).toBe(false);
    expect((await c.post('/api/register', {})).status).toBe(403);
    expect((await new Client(t.base, {'CF-Connecting-IP': '10.7.0.3'}).post('/api/login', {code: r.data.code})).status).toBe(200);
    await t.admin.post('/api/admin/settings', {open_registration: true});
  });
});
