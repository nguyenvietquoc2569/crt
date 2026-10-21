/* Listening test: pure logic (seeded randomisation, session layout, results skeleton, headphone-check signal).
   No DOM access. In the browser it is window.LTCore; in Node, module.exports. */
(function (root) {
  'use strict';

  const BLOCKS = ['mos', 'smos', 'cmos'];
  const HP_SOUNDS = ['quiet', 'antiphase', 'normal'];
  // Headphone-check tones (Woods et al., 2017): 200 Hz, 1 s, 100 ms raised-cosine ramps, 'quiet' 6 dB down.
  // The 0.3 s lead-in (lets wireless headphones wake up) and the 0.5 s gaps are this page's choice.
  const HP_TONE = {freq: 200, dur: 1.0, ramp: 0.1, gap: 0.5, lead: 0.3, amp: 0.3, quietDb: -6};
  // Background questions, asked once after the last rating to describe the listeners in the paper. Every question
  // also offers NO_ANSWER, and there is no free text, so the answers cannot identify a rater.
  const BACKGROUND = [
    {key: 'age', question: 'Your age group', options: ['18–24', '25–34', '35–44', '45 or older']},
    {key: 'first_language', question: 'Your first language', options: ['Vietnamese', 'English', 'Other']},
    {key: 'english', question: 'How well do you understand spoken English?',
     options: ['Basic', 'Intermediate', 'Advanced', 'Native or near-native']},
    {key: 'hearing', question: 'Do you have a known hearing problem?', options: ['No', 'Yes', 'Not sure']},
    {key: 'device', question: 'What did you listen with?',
     options: ['Over-ear or on-ear headphones', 'In-ear earphones or earbuds', 'Other']},
  ];
  const NO_ANSWER = 'Prefer not to say';

  // 32-bit FNV-1a over the UTF-8 bytes of s. Python:
  //   h = 0x811c9dc5; for b in s.encode(): h = ((h ^ b) * 0x01000193) & 0xffffffff
  function hashString(s) {
    const bytes = new TextEncoder().encode(String(s));
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) h = Math.imul(h ^ bytes[i], 0x01000193);
    return h >>> 0;
  }

  // mulberry32 seeded with hashString(seed); returns floats in [0, 1).
  function rng(seed) {
    let a = hashString(seed);
    return function () {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), a | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Fisher-Yates on a copy.
  function shuffle(array, rand) {
    const a = array.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const t = a[i]; a[i] = a[j]; a[j] = t;
    }
    return a;
  }

  // Trailing digits: their integer value mod 2 (= last digit mod 2). Otherwise hashString(id) mod 2.
  function raterParity(raterId) {
    const m = /\d+$/.exec(raterId);
    return m ? Number(m[0].slice(-1)) % 2 : hashString(raterId) % 2;
  }

  function isAttention(t) {
    return t.attention !== undefined && t.attention !== null;
  }

  function selectHalf(trials, half) {
    return trials.filter((t) => half == null || t.half == null || t.half === half);
  }

  // Inserts the attention trials, in random order, into the gaps of the main sequence (gap g lies before main
  // trial g; gap M is the end). Only gaps g >= p = min(minPos, M) are used, so each attention trial has at least
  // p trials before it, and one trial per gap keeps them apart. If there are more attention trials than usable
  // gaps, they are spread as evenly as possible (adjacent only where unavoidable). Main order is unchanged.
  function placeAttention(mainTrials, attentionTrials, rand, minPos = 3) {
    const M = mainTrials.length, K = attentionTrials.length;
    if (K === 0) return mainTrials.slice();
    const p = Math.min(minPos, M);
    const G = M - p + 1;
    const counts = new Array(G).fill(Math.floor(K / G));
    const gaps = shuffle(Array.from({length: G}, (_, g) => g), rand);
    for (let i = 0; i < K % G; i++) counts[gaps[i]]++;
    const att = shuffle(attentionTrials, rand);
    const out = [];
    let next = 0;
    for (let g = 0; g <= M; g++) {
      if (g >= p) for (let c = counts[g - p]; c > 0; c--) out.push(att[next++]);
      if (g < M) out.push(mainTrials[g]);
    }
    return out;
  }

  // For the k-th non-attention trial in presented order, x is shown as A iff (k + parity) % 2 === 0, so the main
  // pairs alternate exactly even with attention trials between them. Attention trials keep x as A and carry their
  // attention value and instruction.
  function abOrder(cmosTrialsInPresentedOrder, parity) {
    // x is A on every other main pair of its group (an opaque `group` per comparison: CRT-AL vs FAISS, vs CRT
    // tau = 0.4, hidden pairs), so each comparison is balanced within a rater and complementary between parities.
    const k = {};
    return cmosTrialsInPresentedOrder.map((t) => {
      if (isAttention(t)) return {tid: t.tid, a: t.x, b: t.y, attention: t.attention, instruction: t.instruction};
      const g = t.group == null ? '' : String(t.group);
      k[g] = (k[g] || 0) + 1;
      const xFirst = (k[g] - 1 + parity) % 2 === 0;
      return {tid: t.tid, a: xFirst ? t.x : t.y, b: xFirst ? t.y : t.x};
    });
  }

  function buildSession(data, raterId) {
    const config = data.config || {};
    const parity = raterParity(raterId);
    const half = config.halves ? (parity === 0 ? 'A' : 'B') : null;
    const session = {
      raterId, parity, half,
      config_title: config.title == null ? null : config.title,
      practice: (data.practice || []).slice(),
    };
    for (const block of BLOCKS) {
      const trials = data[block] || [];
      const main = selectHalf(trials.filter((t) => !isAttention(t)), half);
      const attention = trials.filter(isAttention);
      session[block] = placeAttention(shuffle(main, rng(raterId + ':' + block)), attention,
                                      rng(raterId + ':' + block + ':attention'));
    }
    session.cmos = abOrder(session.cmos, parity);
    return session;
  }

  function emptyResults(session, userAgent, nowIso) {
    return {
      version: 1,
      rater_id: session.raterId,
      parity: session.parity,
      half: session.half,
      started: nowIso,
      finished: null,
      user_agent: userAgent,
      config_title: session.config_title == null ? null : session.config_title,
      headphone: {attempts: [], passed: false},
      responses: {practice: [], mos: [], smos: [], cmos: []},
      block_ms: {practice: 0, mos: 0, smos: 0, cmos: 0},
      background: null,
    };
  }

  // Keys of BACKGROUND whose answer is missing or not one of its options (or NO_ANSWER); empty when complete.
  function checkBackground(answers) {
    const a = answers || {};
    return BACKGROUND.filter((q) => !(q.options.includes(a[q.key]) || a[q.key] === NO_ANSWER)).map((q) => q.key);
  }

  function headphoneTrials(n, rand) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const order = shuffle(HP_SOUNDS, rand);
      out.push({order, answer: order.indexOf('quiet')});
    }
    return out;
  }

  // Stereo signal for a sequence of headphone-check tones ('quiet' | 'antiphase' | 'normal'); 'antiphase' has the
  // right channel inverted, which partly cancels over loudspeakers but not over headphones.
  function headphoneSignal(order, sampleRate) {
    const P = HP_TONE;
    const nTone = Math.round(P.dur * sampleRate), nRamp = Math.round(P.ramp * sampleRate);
    const nGap = Math.round(P.gap * sampleRate), nLead = Math.round(P.lead * sampleRate);
    const n = nLead + order.length * nTone + Math.max(0, order.length - 1) * nGap;
    const left = new Float32Array(n), right = new Float32Array(n);
    order.forEach((kind, k) => {
      const amp = kind === 'quiet' ? P.amp * Math.pow(10, P.quietDb / 20) : P.amp;
      const sign = kind === 'antiphase' ? -1 : 1;
      const off = nLead + k * (nTone + nGap);
      for (let i = 0; i < nTone; i++) {
        const edge = Math.min(i, nTone - 1 - i);
        const env = edge < nRamp ? 0.5 - 0.5 * Math.cos(Math.PI * edge / nRamp) : 1;
        const v = amp * env * Math.sin(2 * Math.PI * P.freq * i / sampleRate);
        left[off + i] = v;
        right[off + i] = sign * v;
      }
    });
    return {left, right};
  }

  // Problems that would break a session part-way: wrong version, unknown stimulus ids, duplicate tids.
  function checkData(data) {
    if (!data || typeof data !== 'object') return ['LT_DATA is missing'];
    const errors = [];
    if (data.version !== 1) errors.push('unsupported LT_DATA version ' + data.version);
    const stimuli = data.stimuli || {};
    const fields = {practice: ['stim'], mos: ['stim'], smos: ['stim', 'ref'], cmos: ['x', 'y']};
    for (const block of Object.keys(fields)) {
      const seen = new Set();
      for (const t of data[block] || []) {
        if (seen.has(t.tid)) errors.push(block + ': duplicate tid ' + t.tid);
        seen.add(t.tid);
        for (const f of fields[block]) {
          if (!Object.prototype.hasOwnProperty.call(stimuli, t[f])) {
            errors.push(block + ' ' + t.tid + ': unknown stimulus ' + f + '=' + t[f]);
          }
        }
      }
    }
    return errors;
  }

  const api = {
    BLOCKS, HP_SOUNDS, HP_TONE, BACKGROUND, NO_ANSWER, hashString, rng, shuffle, raterParity, isAttention,
    selectHalf, placeAttention, abOrder, buildSession, emptyResults, checkBackground, headphoneTrials,
    headphoneSignal, checkData,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.LTCore = api;
})(typeof window !== 'undefined' ? window : this);
