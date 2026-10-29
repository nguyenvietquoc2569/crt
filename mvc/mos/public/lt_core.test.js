// Copied from benchmark/listening/page/lt_core.test.js, with tests of what mvc/mos changed in lt_core.js for
// mvc/LISTENING_ADDENDUM_mvc_v1.md: background question 6, the half given by the server, no item on two consecutive
// main trials, and the break inside a block. sample_lt_data.js is the journal page's, unchanged. The page-structure
// tests differ too: index.html here loads i18n.js instead of lt_data.js (the server sends the data), and the server's
// Content-Security-Policy allows no inline script or style.
// Run with: node --test mvc/mos/public/lt_core.test.js
'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const C = require('./lt_core.js');

const here = (f) => path.join(__dirname, f);

function loadSample() {
  const sandbox = {window: {}};
  vm.runInNewContext(fs.readFileSync(here('sample_lt_data.js'), 'utf8'), sandbox);
  return sandbox.window.LT_DATA;
}

const ids = (xs) => xs.map((t) => t.tid);
const mains = (n, pre = 'm') => Array.from({length: n}, (_, i) => ({tid: pre + i, stim: 's' + i}));
const atts = (n) => Array.from({length: n}, (_, i) => ({tid: 'att' + i, stim: 'sa' + i, attention: 3,
                                                       instruction: 'select 3'}));

function testData(halves) {
  const stimuli = {};
  const add = (sid) => { stimuli[sid] = {src: 'stimuli/' + sid + '.flac', dur: 3}; return sid; };
  const mos = [], smos = [], cmos = [];
  for (let i = 0; i < 12; i++) mos.push({tid: 'mos' + i, stim: add('m' + i), half: i % 2 ? 'B' : 'A'});
  for (let i = 0; i < 4; i++) mos.push({tid: 'mosgold' + i, stim: add('g' + i)});
  mos.push({tid: 'mosatt', stim: add('ma'), attention: 3, instruction: 'select 3'});
  for (let i = 0; i < 8; i++) smos.push({tid: 'smos' + i, stim: add('s' + i), ref: add('r' + (i % 2)),
                                         half: i % 2 ? 'B' : 'A'});
  smos.push({tid: 'smosatt', stim: add('sa'), ref: add('r0'), attention: 1, instruction: 'select 1'});
  for (let i = 0; i < 10; i++) cmos.push({tid: 'cmos' + i, x: add('x' + i), y: add('y' + i)});
  cmos.push({tid: 'cmosatt', x: add('xa'), y: add('ya'), attention: 0, instruction: 'select 0'});
  return {
    version: 1,
    config: {title: 'T', consent_html: '', contact: '', halves, require_full_play: true,
             headphone_check: {trials: 6, pass: 5, attempts: 2}, submit_url: null},
    stimuli,
    practice: [{tid: 'p0', stim: add('p0')}, {tid: 'p1', stim: add('p1')}],
    mos, smos, cmos,
  };
}

test('hashString is 32-bit FNV-1a', () => {
  assert.equal(C.hashString(''), 0x811c9dc5);
  assert.equal(C.hashString('a'), 0xe40c292c);
  assert.equal(C.hashString('foobar'), 0xbf9cf968);
  for (const s of ['R07', 'pilot', 'x:y:z']) {
    const h = C.hashString(s);
    assert.ok(Number.isInteger(h) && h >= 0 && h < 2 ** 32);
  }
});

test('rng is deterministic, in [0, 1), and differs between seeds', () => {
  const draw = (seed, n) => { const r = C.rng(seed); return Array.from({length: n}, () => r()); };
  const a = draw('R07:mos', 1000);
  assert.deepEqual(a, draw('R07:mos', 1000));
  assert.ok(a.every((v) => v >= 0 && v < 1));
  assert.notDeepEqual(a, draw('R08:mos', 1000));
  assert.notDeepEqual(draw('R07:mos', 20), draw('R07:smos', 20));
  const mean = a.reduce((s, v) => s + v, 0) / a.length;
  assert.ok(Math.abs(mean - 0.5) < 0.05);
});

test('shuffle returns a permutation and leaves the input unchanged', () => {
  const input = Array.from({length: 50}, (_, i) => i);
  const copy = input.slice();
  const out = C.shuffle(input, C.rng('s'));
  assert.deepEqual(input, copy);
  assert.notEqual(out, input);
  assert.deepEqual(out.slice().sort((x, y) => x - y), copy);
  assert.notDeepEqual(out, copy);
  assert.deepEqual(out, C.shuffle(input, C.rng('s')));
  assert.deepEqual(C.shuffle([], C.rng('s')), []);
});

test('raterParity uses trailing digits, else the hash', () => {
  assert.equal(C.raterParity('R07'), 1);
  assert.equal(C.raterParity('R10'), 0);
  assert.equal(C.raterParity('7'), 1);
  assert.equal(C.raterParity('R2a4'), 0);
  assert.equal(C.raterParity('R12345678901234567891'), 1);  // beyond 2**53: last digit decides
  const p = C.raterParity('pilot');
  assert.equal(p, C.hashString('pilot') % 2);
  assert.equal(p, 1);  // FNV-1a('pilot') = 0x10066ae1
  assert.equal(C.raterParity('pilot'), p);
});

test('selectHalf keeps trials without a half and those of the chosen half', () => {
  const trials = [{tid: 'a', half: 'A'}, {tid: 'b', half: 'B'}, {tid: 'c'}, {tid: 'd', half: 'A'}];
  assert.deepEqual(ids(C.selectHalf(trials, 'A')), ['a', 'c', 'd']);
  assert.deepEqual(ids(C.selectHalf(trials, 'B')), ['b', 'c']);
  assert.deepEqual(ids(C.selectHalf(trials, null)), ['a', 'b', 'c', 'd']);
  assert.equal(trials.length, 4);
});

test('placeAttention: each trial once, never in the first 3, not adjacent when possible', () => {
  const cases = [[40, 3], [72, 2], [10, 4], [5, 3], [4, 2], [3, 1], [4, 3], [2, 2], [0, 2], [30, 0]];
  for (const [M, K] of cases) {
    const main = mains(M), att = atts(K);
    const p = Math.min(3, M), G = M - p + 1;
    const firstPositions = new Set();
    for (let seed = 0; seed < 300; seed++) {
      const seq = C.placeAttention(main, att, C.rng('seed' + seed));
      assert.equal(seq.length, M + K);
      assert.deepEqual(ids(seq).sort(), ids(main.concat(att)).sort());
      assert.deepEqual(ids(seq.filter((t) => !C.isAttention(t))), ids(main));  // main order kept
      const pos = seq.map((t, i) => (C.isAttention(t) ? i : -1)).filter((i) => i >= 0);
      if (K) firstPositions.add(pos[0]);
      assert.ok(pos.every((i) => i >= p), `M=${M} K=${K}: attention before position ${p}`);
      let run = 0, maxRun = 0;
      for (const t of seq) { run = C.isAttention(t) ? run + 1 : 0; maxRun = Math.max(maxRun, run); }
      if (K <= G) assert.ok(maxRun <= 1, `M=${M} K=${K}: adjacent attention trials`);
      else assert.ok(maxRun <= Math.ceil(K / G), `M=${M} K=${K}: uneven spread`);
    }
    if (K && G > K) assert.ok(firstPositions.size > 1, `M=${M} K=${K}: placement never varies`);
  }
  assert.equal(C.placeAttention(mains(4), [], C.rng('x')).length, 4);
});

test('abOrder alternates A/B exactly and complements between parities', () => {
  const pairs = Array.from({length: 32}, (_, i) => ({tid: 'c' + i, x: 'x' + i, y: 'y' + i}));
  const o0 = C.abOrder(pairs, 0), o1 = C.abOrder(pairs, 1);
  for (const o of [o0, o1]) {
    assert.equal(o.length, 32);
    assert.equal(o.filter((t, i) => t.a === pairs[i].x).length, 16);
    o.forEach((t, i) => assert.deepEqual([t.a, t.b].sort(), [pairs[i].x, pairs[i].y].sort()));
    assert.deepEqual(Object.keys(o[0]), ['tid', 'a', 'b']);
  }
  assert.equal(o0[0].a, 'x0');
  o0.forEach((t, i) => { assert.equal(t.a, o1[i].b); assert.equal(t.b, o1[i].a); });

  // Attention trials keep x as A and do not break the alternation of the main pairs.
  const mixed = pairs.slice();
  mixed.splice(5, 0, {tid: 'att1', x: 'ax1', y: 'ay1', attention: 2, instruction: 'select +2'});
  mixed.splice(20, 0, {tid: 'att2', x: 'ax2', y: 'ay2', attention: 0, instruction: 'select 0'});
  for (const parity of [0, 1]) {
    const o = C.abOrder(mixed, parity);
    const att = o.filter((t) => t.tid.startsWith('att'));
    assert.deepEqual(att.map((t) => [t.a, t.b, t.attention]), [['ax1', 'ay1', 2], ['ax2', 'ay2', 0]]);
    assert.equal(att[0].instruction, 'select +2');
    const main = o.filter((t) => !t.tid.startsWith('att'));
    assert.equal(main.filter((t) => t.a.startsWith('x')).length, 16);
    main.forEach((t, k) => assert.equal(t.a.startsWith('x'), (k + parity) % 2 === 0));
  }
});

test('buildSession: seeds, attention placement, A/B order and halves', () => {
  for (const halves of [false, true]) {
    const data = testData(halves);
    for (const id of ['R07', 'R10', 'pilot']) {
      const s = C.buildSession(data, id);
      const parity = C.raterParity(id);
      const half = halves ? (parity === 0 ? 'A' : 'B') : null;
      assert.deepEqual(Object.keys(s), ['raterId', 'parity', 'half', 'config_title', 'practice', 'mos', 'smos',
                                        'cmos']);
      assert.equal(s.raterId, id);
      assert.equal(s.parity, parity);
      assert.equal(s.half, half);
      assert.deepEqual(ids(s.practice), ['p0', 'p1']);
      for (const block of ['mos', 'smos', 'cmos']) {
        const main = C.selectHalf(data[block].filter((t) => !C.isAttention(t)), half);
        const att = data[block].filter(C.isAttention);
        let expected = C.placeAttention(C.shuffle(main, C.rng(id + ':' + block)), att,
                                        C.rng(id + ':' + block + ':attention'));
        if (block === 'cmos') expected = C.abOrder(expected, parity);
        assert.deepEqual(s[block], expected);
        assert.ok(s[block].slice(0, 3).every((t) => !C.isAttention(t)));
      }
      assert.deepEqual(C.buildSession(data, id), s);
    }
  }
  const off = C.buildSession(testData(false), 'R07');
  assert.equal(off.mos.length, 17);
  assert.equal(off.smos.length, 9);
  const on = C.buildSession(testData(true), 'R07');
  assert.equal(on.half, 'B');
  assert.equal(on.mos.length, 6 + 4 + 1);
  assert.ok(on.mos.every((t) => t.half === undefined || t.half === 'B'));
  assert.equal(on.smos.length, 4 + 1);
  assert.equal(on.cmos.length, 11);  // CMOS trials have no half: every rater does all pairs
  assert.equal(C.buildSession(testData(true), 'R10').half, 'A');
  assert.notDeepEqual(ids(C.buildSession(testData(false), 'R08').mos), ids(off.mos));
});

test('emptyResults has the results schema', () => {
  const s = C.buildSession(testData(true), 'R10');
  const r = C.emptyResults(s, 'UA/1.0', '2026-10-19T08:00:00.000Z');
  assert.deepEqual(Object.keys(r), ['version', 'rater_id', 'parity', 'half', 'started', 'finished', 'user_agent',
                                    'config_title', 'headphone', 'responses', 'block_ms', 'background']);
  assert.equal(r.background, null);
  assert.deepEqual(r.headphone, {attempts: [], passed: false});
  assert.deepEqual(r.responses, {practice: [], mos: [], smos: [], cmos: []});
  assert.deepEqual(r.block_ms, {practice: 0, mos: 0, smos: 0, cmos: 0});  // numbers: the analysis sums them
  assert.equal(r.version, 1);
  assert.equal(r.rater_id, 'R10');
  assert.equal(r.parity, 0);
  assert.equal(r.half, 'A');
  assert.equal(r.started, '2026-10-19T08:00:00.000Z');
  assert.equal(r.finished, null);
  assert.equal(r.user_agent, 'UA/1.0');
  assert.equal(r.config_title, 'T');
});

test('headphoneTrials: permutations of the three sounds, answer points at quiet', () => {
  const trials = C.headphoneTrials(600, C.rng('R07:headphone:0'));
  assert.equal(trials.length, 600);
  const orders = new Set();
  for (const t of trials) {
    assert.deepEqual(t.order.slice().sort(), ['antiphase', 'normal', 'quiet']);
    assert.equal(t.order[t.answer], 'quiet');
    orders.add(t.order.join());
  }
  assert.equal(orders.size, 6);
  assert.deepEqual(C.headphoneTrials(6, C.rng('a')), C.headphoneTrials(6, C.rng('a')));
  assert.equal(C.headphoneTrials(6, C.rng('a')).length, 6);
});

test('headphoneSignal: 6 dB quieter tone, inverted channel for antiphase, ramps and silences', () => {
  const sr = 48000, P = C.HP_TONE;
  const order = ['normal', 'quiet', 'antiphase'];
  const {left, right} = C.headphoneSignal(order, sr);
  const [nTone, nGap, nLead] = [P.dur, P.gap, P.lead].map((s) => Math.round(s * sr));
  assert.equal(left.length, nLead + 3 * nTone + 2 * nGap);
  const seg = (k) => [nLead + k * (nTone + nGap), nLead + k * (nTone + nGap) + nTone];
  const rms = (x, [a, b]) => Math.sqrt(x.subarray(a, b).reduce((s, v) => s + v * v, 0) / (b - a));
  const db = 20 * Math.log10(rms(left, seg(1)) / rms(left, seg(0)));
  assert.ok(Math.abs(db - P.quietDb) < 0.01, `quiet tone is ${db} dB`);
  assert.ok(Math.abs(rms(left, seg(2)) - rms(left, seg(0))) < 1e-6);
  for (const k of [0, 1]) for (let i = seg(k)[0]; i < seg(k)[1]; i++) assert.equal(right[i], left[i]);
  for (let i = seg(2)[0]; i < seg(2)[1]; i++) assert.equal(right[i], -left[i]);
  for (const k of [0, 1, 2]) {
    assert.ok(Math.abs(left[seg(k)[0]]) < 1e-6 && Math.abs(left[seg(k)[1] - 1]) < 1e-3);
  }
  const silent = [[0, nLead], [seg(0)[1], seg(1)[0]], [seg(1)[1], seg(2)[0]]];
  for (const [a, b] of silent) assert.ok(left.subarray(a, b).every((v) => v === 0));
  const peak = left.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  assert.ok(peak <= P.amp + 1e-6 && peak > 0.9 * P.amp);
});

test('checkData accepts the sample data and reports unknown stimuli and duplicate tids', () => {
  const data = loadSample();
  assert.deepEqual(C.checkData(data), []);
  const bad = JSON.parse(JSON.stringify(data));
  bad.smos[0].ref = 'nope';
  bad.mos.push(Object.assign({}, bad.mos[0]));
  const errors = C.checkData(bad);
  assert.equal(errors.length, 2);
  assert.ok(errors.some((e) => e.includes('unknown stimulus ref=nope')));
  assert.ok(errors.some((e) => e.includes('duplicate tid')));
  assert.deepEqual(C.checkData(undefined), ['LT_DATA is missing']);
});

test('sample data has the requested sizes', () => {
  const data = loadSample();
  const count = (block) => [data[block].length, data[block].filter(C.isAttention).length];
  assert.equal(data.practice.length, 3);
  assert.deepEqual(count('mos'), [6, 1]);
  assert.deepEqual(count('smos'), [4, 1]);
  assert.deepEqual(count('cmos'), [4, 1]);
  const s = C.buildSession(data, 'R07');
  assert.equal(s.mos.length + s.smos.length + s.cmos.length, 14);
});

test('index.html loads style.css, i18n.js, lt_core.js, lt_app.js in order, without modules or inline code', () => {
  const html = fs.readFileSync(here('index.html'), 'utf8');
  const refs = [...html.matchAll(/<(?:link[^>]*href|script[^>]*src)="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(refs, ['style.css', 'i18n.js', 'lt_core.js', 'lt_app.js']);
  assert.ok(!/type="module"/.test(html));
  assert.match(html, /<title>Listening test<\/title>/);
  assert.ok(!/https?:\/\//.test(html), 'no external resources');
  for (const page of ['index.html', 'admin.html']) {
    const h = fs.readFileSync(here(page), 'utf8');
    assert.ok(!/<script(?![^>]*\ssrc=)[^>]*>/.test(h), page + ': no inline script');
    assert.ok(!/\son\w+=|\sstyle=|<style/.test(h), page + ': no inline handler or style');
  }
});

test('lt_app.js, i18n.js and admin.js parse and use no external resources or inline styles', () => {
  for (const f of ['lt_app.js', 'i18n.js', 'admin.js']) {
    const src = fs.readFileSync(here(f), 'utf8');
    assert.doesNotThrow(() => new vm.Script(src), f);
    assert.ok(!/https?:\/\//.test(src), f);
    assert.ok(!/style:\s*'|setAttribute\('style'/.test(src), f + ': inline style attributes are blocked by the CSP');
  }
  assert.ok(!/@import|url\(/.test(fs.readFileSync(here('style.css'), 'utf8')));
});

test('abOrder balances each comparison group on its own', () => {
  const pairs = [];
  for (let i = 0; i < 32; i++) {
    pairs.push({tid: 'p' + i, x: 'px' + i, y: 'py' + i, group: 'gP'});
    pairs.push({tid: 'k' + i, x: 'kx' + i, y: 'ky' + i, group: 'gK'});
    if (i % 4 === 0) pairs.push({tid: 'h' + i, x: 'hx' + i, y: 'hy' + i, group: 'gH'});
  }
  for (const seed of ['R01', 'R02', 'pilot-3']) {
    const shuffled = C.shuffle(pairs, C.rng(seed));
    const o0 = C.abOrder(shuffled, 0), o1 = C.abOrder(shuffled, 1);
    for (const [g, n] of [['p', 32], ['k', 32], ['h', 8]]) {
      for (const o of [o0, o1]) {
        const mine = o.filter((t) => t.tid[0] === g);
        assert.equal(mine.length, n);
        assert.equal(mine.filter((t) => t.a[1] === 'x').length, n / 2, g + ' balanced');
      }
    }
    o0.forEach((t, i) => { assert.equal(t.a, o1[i].b); assert.equal(t.b, o1[i].a); });
  }
});

test('background questions: every one has options and a no-answer, and checkBackground names what is missing', () => {
  const keys = C.BACKGROUND.map((q) => q.key);
  assert.deepEqual(keys, ['age', 'first_language', 'english', 'hearing', 'device', 'previous_test']);
  // addendum §7.4, question 6
  assert.deepEqual(C.BACKGROUND[5], {key: 'previous_test', options: ['Yes', 'No', 'Not sure'],
                                     question: 'Have you taken part in a listening test of this research group before?'});
  for (const q of C.BACKGROUND) {
    assert.ok(q.question && q.options.length >= 3 && !q.options.includes(C.NO_ANSWER));
  }
  assert.deepEqual(C.checkBackground(null), keys);
  assert.deepEqual(C.checkBackground({}), keys);
  const full = Object.fromEntries(C.BACKGROUND.map((q) => [q.key, q.options[0]]));
  assert.deepEqual(C.checkBackground(full), []);
  assert.deepEqual(C.checkBackground(Object.fromEntries(keys.map((k) => [k, C.NO_ANSWER]))), []);
  assert.deepEqual(C.checkBackground({...full, english: 'Fluent'}), ['english']);   // not an offered option
  const {device, ...noDevice} = full;
  assert.deepEqual(C.checkBackground(noDevice), ['device']);
});

test('lt_app.js asks the background questions after the last rating and keeps finished at the last rating', () => {
  const src = fs.readFileSync(here('lt_app.js'), 'utf8');
  assert.match(src, /return R\.background \? \{name: 'finish'\} : \{name: 'background'\};/);
  assert.match(src, /else if \(s\.name === 'background'\) renderBackground\(\);/);
  assert.match(src, /\['background', 'finish'\]\.includes\(screen\(\)\.name\) && !R\.finished\) R\.finished = rec\.t;/);
  assert.match(src, /state\.results\.background = Object\.assign\(\{\}, answers\);/);
  assert.match(src, /C\.checkBackground\(answers\)\.length > 0/);
});

// ── mvc/mos changes (addendum §6) ────────────────────────────

// count trials per item, k items: [{tid, stim, item}]
const itemTrials = (k, count, pre = 't') => {
  const out = [];
  for (let i = 0; i < k; i++) for (let c = 0; c < count; c++) out.push({tid: pre + i + '_' + c, stim: 's' + i + c, item: 'i' + i});
  return out;
};

test('itemsApart: trials without an item never clash', () => {
  assert.equal(C.itemsApart([{tid: 'a', item: 'x'}, {tid: 'b', item: 'x'}]), false);
  assert.equal(C.itemsApart([{tid: 'a', item: 'x'}, {tid: 'b', item: 'y'}, {tid: 'c', item: 'x'}]), true);
  assert.equal(C.itemsApart([{tid: 'a'}, {tid: 'b'}]), true);
  assert.equal(C.itemsApart([]), true);
});

test('orderMain: no item on two consecutive trials, deterministic per seed, a permutation', () => {
  // the real design per rater: 16 items x 5 MOS conditions, 16 items x 6 SMOS conditions; and smaller ones
  for (const [k, count] of [[16, 5], [16, 6], [4, 5], [4, 6], [8, 3]]) {
    const trials = itemTrials(k, count);
    const firsts = new Set();
    for (let seed = 0; seed < 200; seed++) {
      const out = C.orderMain(trials, C.rng('R' + seed + ':mos'));
      assert.ok(C.itemsApart(out), `k=${k} count=${count} seed=${seed}`);
      assert.deepEqual(ids(out).sort(), ids(trials).sort());
      assert.deepEqual(ids(C.orderMain(trials, C.rng('R' + seed + ':mos'))), ids(out));
      firsts.add(out[0].tid);
    }
    assert.ok(firsts.size > k, `k=${k}: the order varies between raters`);
  }
  // without item ids it is the journal page's shuffle
  const plain = mains(30);
  assert.deepEqual(C.orderMain(plain, C.rng('R07:mos')), C.shuffle(plain, C.rng('R07:mos')));
});

test('orderMain: every position is equally likely for an item (uniform among valid orders)', () => {
  const trials = itemTrials(4, 5);
  const n = 4000, first = {i0: 0, i1: 0, i2: 0, i3: 0};
  for (let seed = 0; seed < n; seed++) first[C.orderMain(trials, C.rng('u' + seed))[0].item]++;
  for (const v of Object.values(first)) assert.ok(Math.abs(v / n - 0.25) < 0.03, JSON.stringify(first));
});

test('separateItems repairs any order that can be repaired, keeping the trials', () => {
  for (let seed = 0; seed < 500; seed++) {
    const r = C.rng('rep' + seed);
    const k = 2 + Math.floor(r() * 5);
    const trials = [];
    for (let i = 0; i < k; i++) {
      const count = 1 + Math.floor(r() * 6);
      for (let c = 0; c < count; c++) trials.push({tid: 'x' + i + '_' + c, item: 'i' + i});
    }
    const most = Math.max(...Object.values(trials.reduce((m, t) => ((m[t.item] = (m[t.item] || 0) + 1), m), {})));
    const seq = C.shuffle(trials, r);
    const out = C.separateItems(seq);
    assert.deepEqual(ids(out).sort(), ids(trials).sort());
    if (most <= Math.ceil(trials.length / 2)) assert.ok(C.itemsApart(out), `seed ${seed}: ${out.map((t) => t.item)}`);
  }
  // worst cases: one item fills half the block, at the start or at the end
  const half = (n) => [...itemTrials(1, n, 'a'), ...itemTrials(3, 1, 'b').map((t, i) => ({...t, item: 'j' + i}))];
  for (const seq of [half(4), half(4).reverse(), [...itemTrials(1, 3, 'a'), {tid: 'b', item: 'q'}, {tid: 'c', item: 'r'}]]) {
    assert.ok(C.itemsApart(C.separateItems(seq)), seq.map((t) => t.item).join());
  }
  assert.deepEqual(C.separateItems([]), []);
});

test('buildSession: the half given by the server decides, and main trials of an item are never consecutive', () => {
  const data = testData(true);
  for (const t of data.mos) if (!C.isAttention(t)) t.item = 'item' + (Number(t.tid.replace(/\D/g, '')) % 3);
  for (const [id, half] of [['R001', 'A'], ['R001', 'B'], ['R002', 'A']]) {
    const s = C.buildSession(data, id, half);
    assert.equal(s.half, half);
    assert.equal(s.parity, half === 'A' ? 0 : 1);
    assert.ok(s.mos.filter((t) => !C.isAttention(t)).every((t) => t.half === undefined || t.half === half));
    assert.ok(C.itemsApart(s.mos.filter((t) => !C.isAttention(t))));
    assert.ok(C.itemsApart(s.mos));
    assert.ok(s.mos.slice(0, 3).every((t) => !C.isAttention(t)));
    assert.deepEqual(C.buildSession(data, id, half), s);
  }
  // without halves in the data, the half is null and the parity follows the given half
  assert.equal(C.buildSession(testData(false), 'R001', 'B').half, null);
  assert.equal(C.buildSession(testData(false), 'R001', 'B').parity, 1);
  // another rater of the same half: another order
  assert.notDeepEqual(ids(C.buildSession(data, 'R001', 'A').mos), ids(C.buildSession(data, 'R002', 'A').mos));
});

test('buildSession: two SMOS attention trials are never in the first 3 positions nor next to each other', () => {
  const data = testData(true);
  data.smos.push({tid: 'smosatt2', stim: 's1', ref: 'r1', attention: 4});
  for (let seed = 0; seed < 300; seed++) {
    for (const half of ['A', 'B']) {
      const s = C.buildSession(data, 'R' + seed, half);
      const pos = s.smos.map((t, i) => (C.isAttention(t) ? i : -1)).filter((i) => i >= 0);
      assert.equal(pos.length, 2);
      assert.ok(pos[0] >= 3 && pos[1] - pos[0] > 1, `${seed} ${half}: ${pos}`);
    }
  }
});

test('midBreak: after trial n/2 (82 -> 41, 98 -> 49), none for a single trial', () => {
  assert.equal(C.midBreak(82), 41);
  assert.equal(C.midBreak(98), 49);
  assert.equal(C.midBreak(22), 11);
  assert.equal(C.midBreak(5), 2);
  assert.equal(C.midBreak(2), 1);
  assert.equal(C.midBreak(1), null);
  assert.equal(C.midBreak(0), null);
});

test('lt_app.js shows the break screen in the middle of each rating block, not in the practice', () => {
  const src = fs.readFileSync(here('lt_app.js'), 'utf8');
  assert.match(src, /block !== 'practice' && done === C\.midBreak\(S\[block\]\.length\) && !breakSeen\.has\(block\)/);
  assert.match(src, /else if \(s\.name === 'break'\) renderBreak\(s\.block\);/);
});
