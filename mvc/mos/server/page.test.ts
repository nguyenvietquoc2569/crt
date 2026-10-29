// The rater page (public/lt_app.js) against the server, in the small browser of fake_page.ts: the access code before
// the consent (nothing stored before the agreement), a consent text changed while it was on screen, a whole session
// (a trial left open for over 24 h, the break screens in the middle of each block, six background questions, then
// nothing of the rater left in the browser), "Use another code" (nothing of the rater left unless answers wait to be
// sent), the 12 characters of a code, registration closed while the page was open, the language switch on a trial and
// on a sound-check trial (the plays and the choice stay; the language is stored per answer), a finish whose answer was
// lost, and a server that keeps refusing a response (the page stops instead of looping).
import { afterAll, beforeAll, describe, expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { sha256 } from './core';
import { agree, background, Browser, EN, enterCode, Page, rate, soundCheck, VI } from './fake_page';
import { boot, Client, hpAnswers, makePackage, setup, tempDir } from './test_util';

let t: Awaited<ReturnType<typeof boot>>;
let codes: string[];
let pkgId: string;

beforeAll(async () => {
  const dir = tempDir();
  makePackage(dir);
  t = await boot({dataDir: dir});
  pkgId = (await t.admin.post('/api/admin/package/import', {})).data.package_id;
  codes = await setup(t, 7);
  await t.admin.post('/api/admin/settings', {contact: 'lab@example.org'});
});
afterAll(() => t.stop());

const idOf = async (code: string) => (await t.admin.post('/api/admin/raters/find', {code})).data.rater_id as string;
const right = (id: string, attempt: number) => (i: number) => hpAnswers(id, attempt)[i];
const wrong = (id: string, attempt: number) => (i: number) => hpAnswers(id, attempt, true)[i];
const N = 3 + 22 + 26;   // practice, MOS and SMOS trials of a rater

describe('the rater page', () => {
  test('the code comes first; a consent text changed while it was on screen is shown again and the new one recorded', async () => {
    const code = codes[0];
    const id = await idOf(code);
    const page = new Page(t.base, new Browser()).open();
    await page.waitFor(() => page.text().includes(EN.codeTitle), 'the code screen');
    expect(page.text()).not.toContain('Information (en)');
    await enterCode(page, code);
    await page.waitFor(() => page.text().includes('Information (en)'), 'the consent text');
    expect(page.text()).toContain(EN.consentTitle);
    expect(page.text()).not.toContain(code);   // the code is not shown back
    expect(t.app.store.session(id, pkgId)).toBeNull();   // nothing stored before the agreement
    // the organiser edits the text while the rater reads it
    const r = await t.admin.post('/api/admin/consent', {html: '<p>Thông tin (vi)</p><hr><p>Information, version 2 (en)</p>'});
    expect(r.status).toBe(200);
    await agree(page);
    await page.waitFor(() => page.text().includes('version 2'), 'the new consent text');
    expect(page.text()).toContain(EN.consentChanged);
    await new Promise((res) => setTimeout(res, 100));   // a loop would keep sending in the meantime
    expect(page.count('POST /api/consent')).toBe(1);
    expect(page.requests.length).toBeLessThan(12);
    expect(t.app.store.session(id, pkgId)).toBeNull();

    await agree(page);
    await page.waitFor(() => page.button(EN.hpStart), 'the sound check');
    const s = t.app.store.session(id, pkgId)!;
    expect(s.consent_lang).toBe('en');
    expect(s.consent_sha).toBe(sha256(readFileSync(join(t.dataDir, 'consent_v1.html'), 'utf8')));
    expect(page.count('POST /api/consent')).toBe(2);
    expect(page.requests.indexOf('GET /api/plan')).toBeGreaterThan(page.requests.lastIndexOf('POST /api/consent'));
  });

  test('a whole session: breaks in the middle of each block, a trial open for over 24 h, six questions, then nothing left', async () => {
    const code = codes[1];
    const id = await idOf(code);
    const browser = new Browser();
    const page = new Page(t.base, browser).open();
    await enterCode(page, code);
    await agree(page);
    await soundCheck(page, right(id, 0));
    const plan = await new Client(t.base, {Cookie: browser.cookieHeader()}).get('/api/plan');
    const n = ['practice', 'mos', 'smos', 'cmos'].reduce((s, b) => s + plan.data.session[b].length, 0);
    expect(n).toBe(N);
    // the rater leaves the first practice trial open for 25 hours before rating it
    const breaks = await rate(page, n, 4, (k) => { if (k === 0) page.clock += 25 * 36e5; });
    // MOS: 22 trials, the break after trial 11; SMOS: 26 trials, the break after trial 13; none in the practice
    expect(breaks).toEqual([3 + 11, 3 + 22 + 13]);
    expect(await background(page)).toBe(6);
    await page.waitFor(() => page.text().includes(EN.finishTitle), 'the end screen');
    await page.waitFor(() => t.app.store.session(id, pkgId)!.completed_at, 'the finish on the server');

    const answers = t.app.store.answers(id, pkgId);
    expect(answers.length).toBe(n);
    expect(answers.every((a) => a.lang === 'en')).toBe(true);
    const first = answers.find((a) => a.block === 'practice' && a.pos === 0)!;
    expect(first.t_ms).toBe(864e5);
    expect(first.rt_ms).toBe(864e5);
    expect(Object.keys(t.app.store.session(id, pkgId)!.background!)).toEqual(
      ['age', 'first_language', 'english', 'hearing', 'device', 'previous_test']);
    expect(page.requests.filter((x) => x === 'GET /api/state').length).toBe(2);   // at the start and after the consent
    expect(page.text()).not.toContain(EN.resynced);

    // the sign-in cookie is gone, and so is the local copy (the language choice stays)
    await page.waitFor(() => !browser.cookies.has('mos_sid'), 'the cookie to be cleared');
    await page.waitFor(() => ![...browser.storage.map.keys()].some((k) => k.startsWith(`mos:${id}:`)),
                       'the local copy to be cleared');
    expect([...browser.storage.map.keys()]).toEqual(['mos:lang']);
    // the next person at this computer starts at the code screen
    const next = new Page(t.base, browser).open();
    await next.waitFor(() => next.text().includes(EN.codeTitle), 'the code screen');
    expect(next.text()).not.toContain(code);
  });

  test('"Use another code" leaves nothing of the rater in the browser, unless answers still wait to be sent', async () => {
    const code = codes[5];
    const id = await idOf(code);
    const browser = new Browser();
    const page = new Page(t.base, browser).open();
    // the Continue button needs the 12 characters of a code
    const input = await page.waitFor(() => page.all('INPUT').find((e) => e.type === 'text'), 'the code field');
    for (const [typed, off] of [[code.slice(0, 11), true], [code, false], [' ' + code.slice(0, 11) + ' ', true]] as const) {
      input.value = typed;
      input.dispatch('input');
      expect(page.button(EN.continue)!.disabled, typed).toBe(off);
    }
    await enterCode(page, code);
    await agree(page);
    await soundCheck(page, right(id, 0));
    await rate(page, 2);
    await page.waitFor(() => t.app.store.answers(id, pkgId).length === 2, 'two answers on the server');
    const mine = () => [...browser.storage.map.keys()].filter((k) => k.startsWith(`mos:${id}:`));
    expect(mine().length).toBe(1);
    await page.click(EN.logout);
    await page.waitFor(() => page.text().includes(EN.codeTitle), 'the code screen');
    expect(mine()).toEqual([]);
    expect([...browser.storage.map.keys()]).toEqual(['mos:lang']);

    // again, with the third answer refused by a failing server: it waits in the local copy, which stays
    await enterCode(page, code);
    await page.waitFor(() => page.buttons().some((b) => b.classList.contains('play')), 'the third trial');
    page.intercept = (method, path) => (method === 'POST' && path === '/api/answer'
      ? new Response('{}', {status: 503, headers: {'Content-Type': 'application/json'}}) : null);
    await rate(page, 1);
    await page.waitFor(() => page.text().includes(EN.pending), 'the pending notice');
    await page.click(EN.logout);
    await page.waitFor(() => page.text().includes(EN.codeTitle), 'the code screen');
    expect(mine().length).toBe(1);
    // the server is back: the code is entered again and the waiting answer is sent
    page.intercept = null;
    await enterCode(page, code);
    await page.waitFor(() => t.app.store.answers(id, pkgId).length === 3, 'the third answer on the server');
  });

  test('registration closed while the page was open: the page says so and stops offering it', async () => {
    const page = new Page(t.base, new Browser()).open();
    await page.waitFor(() => page.button(EN.startNew), 'the registration button');
    await t.admin.post('/api/admin/settings', {open_registration: false});
    try {
      await page.click(EN.startNew);
      await page.waitFor(() => page.text().includes(EN.regClosed), 'the closed message');
      expect(page.button(EN.startNew)!.hidden).toBe(true);
      expect(page.text()).not.toContain(EN.startNew);
    } finally {
      await t.admin.post('/api/admin/settings', {open_registration: true});
    }
  });

  test('the language switch on a sound-check trial and on a rating trial keeps what was played and chosen', async () => {
    const code = codes[4];
    const id = await idOf(code);
    const browser = new Browser('en');
    const page = new Page(t.base, browser).open();
    await enterCode(page, code);
    await agree(page);
    // sound-check trial 0: after the tones, switch to Vietnamese and back; the tones need not (and cannot) play again
    await soundCheck(page, right(id, 0), (i) => {
      if (i !== 0) return;
      page.switchLanguage();
      expect(page.text()).toContain(VI.hpQuestion);
      const play = page.buttons().find((b) => b.classList.contains('play'))!;
      expect(play.disabled).toBe(true);
      expect(page.buttons().filter((b) => b.classList.contains('rate')).every((b) => !b.disabled)).toBe(true);
      page.switchLanguage();
    });
    // rating trial 5 (a MOS trial): played in English, then switched to Vietnamese; the rest is rated in Vietnamese
    await rate(page, N, 3, (k) => {
      if (k !== 5) return;
      page.switchLanguage();
      expect(page.text()).toContain(VI.q_mos);
      expect(page.text()).toContain(VI.playedBadge);
      expect(page.buttons().filter((b) => b.classList.contains('rate')).every((b) => !b.disabled)).toBe(true);
    });
    expect(await background(page)).toBe(6);
    await page.waitFor(() => page.text().includes(VI.finishTitle), 'the end screen in Vietnamese');
    await page.waitFor(() => t.app.store.session(id, pkgId)?.completed_at, 'the finish on the server');
    const answers = t.app.store.answers(id, pkgId).sort((x, y) => ['practice', 'mos', 'smos'].indexOf(x.block) -
                                                                   ['practice', 'mos', 'smos'].indexOf(y.block) || x.pos - y.pos);
    expect(answers.map((a) => a.lang)).toEqual([...Array(5).fill('en'), ...Array(N - 5).fill('vi')]);
    const fifth = answers[5];
    expect([fifth.block, fifth.pos, fifth.plays, fifth.rating]).toEqual(['mos', 2, 1, 3]);   // one play, not two
    expect(t.app.store.session(id, pkgId)!.headphone.attempts[0].correct).toBe(6);
    expect(browser.storage.getItem('mos:lang')).toBe('vi');
  });

  test('a finish whose answer was lost: the repeat gets 401, and the page still ends the session', async () => {
    const code = codes[3];
    const id = await idOf(code);
    const browser = new Browser();
    const page = new Page(t.base, browser).open();
    await enterCode(page, code);
    await agree(page);
    await soundCheck(page, wrong(id, 0));
    let lost = 0;
    page.intercept = (method, path) => (method === 'POST' && path === '/api/finish' && !lost++ ? 'lose' : null);
    await soundCheck(page, wrong(id, 1));
    await page.waitFor(() => page.count('POST /api/finish') === 2, 'the repeated finish');
    await page.waitFor(() => ![...browser.storage.map.keys()].some((k) => k.startsWith(`mos:${id}:`)),
                       'the local copy to be cleared');
    expect(t.app.store.session(id, pkgId)!.completed_at).not.toBeNull();
    expect(page.text()).toContain(EN.hpFailTitle);
    expect(page.text()).not.toContain(EN.expired);
    expect(page.text()).not.toContain(EN.pending);
    // the cookie the browser kept (it never saw the answer that cleared it) no longer signs in
    expect(browser.cookies.has('mos_sid')).toBe(true);
    expect((await new Client(t.base, {Cookie: browser.cookieHeader()}).get('/api/state')).data).toEqual({auth: false});
  });

  test('a server that keeps refusing a response makes the page stop after 3 resyncs, not loop', async () => {
    const code = codes[2];
    const id = await idOf(code);
    const page = new Page(t.base, new Browser()).open();
    await enterCode(page, code);
    await agree(page);
    await soundCheck(page, wrong(id, 0));
    page.intercept = (method, path) => (method === 'POST' && path === '/api/finish'
      ? new Response(JSON.stringify({error: {code: 'not_finished', message: 'the session is not finished'}}),
                     {status: 409, headers: {'Content-Type': 'application/json'}})
      : null);
    await soundCheck(page, wrong(id, 1));   // two failed attempts: the page sends finish, which is refused every time
    const msg = EN.syncFailed.replace('{contact}', 'lab@example.org');
    await page.waitFor(() => page.text().includes(msg), 'the stop message');
    const sent = page.requests.length;
    await new Promise((res) => setTimeout(res, 150));
    expect(page.requests.length).toBe(sent);
    expect(page.count('POST /api/finish')).toBe(4);   // the first try and 3 after a resync each
  });
});
