/* MOS/SMOS listening test UI for the server in mvc/mos/server, in the order of mvc/LISTENING_ADDENDUM_mvc_v1.md §6.1:
   access code (or a new code from open registration), consent, headphone check, practice, the MOS block, a break, the
   SMOS block, each block with a break screen in its middle, background questions, thanks. Adapted from
   benchmark/listening/page/lt_app.js: the trials come from the server (GET /api/plan, only after the consent), every
   response is sent to the server at once with the page language (an ordered outbox, kept in localStorage until the
   server has it), and a rater who returns with the same code on any device resumes at the first unanswered trial.
   Every screen has the language switch. Needs window.LTCore (lt_core.js) and window.MOSI18n (i18n.js). */
(() => {
  'use strict';
  const C = window.LTCore, I = window.MOSI18n;
  const ORDER = ['practice', 'mos', 'smos', 'cmos'];
  const PLAY = '▶︎', STOP = '■︎', CHECK = '✓';
  const RETRY_MS = 5000;

  let lang = I ? I.initialLang() : 'vi';
  let app;
  let INFO = {};           // GET /api/info
  let D = null;            // GET /api/plan: {package_id, sample, config, session, stimuli}
  let CFG = {}, HP, REQUIRE_FULL;
  let state = null;        // {raterId, packageId, session, results, consented, completed}
  let outbox = [];         // responses not yet confirmed by the server, in order: [{path, body}]
  let flushing = false, retryTimer = null;
  let notice = '', noticeCls = 'ok';
  let redraw = null;       // re-renders the current screen after a language change
  let cur = null;          // the rating trial on screen
  let hpCur = null;        // the headphone trial on screen: {attempt, i, started, choice}
  let codeDraft = '';      // the code being typed, kept across a language change
  let bgDraft = {};        // the background answers being given, kept across a language change
  const introSeen = new Set();  // blocks whose introduction was dismissed in this page load
  const breakSeen = new Set();  // blocks whose middle break was dismissed in this page load
  let actx = null, hpSource = null;
  const pool = new Map();  // <audio> elements of the current and the next trial, by block|tid|role
  let pendingEl = null;
  let sendFailed = false;  // the last send failed: responses wait in the outbox and are retried
  let finishShown = false; // the end screen is on, so it is redrawn once the outbox is empty
  let resyncs = [];        // times of the recent resyncs: a page that keeps disagreeing with the server stops

  const T = (key, vars) => I.t(lang, key, vars);

  // ── DOM helpers ───────────────────────────────────────────
  function el(tag, attrs, ...children) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
    for (const c of children.flat(Infinity)) {
      if (c != null && c !== false) e.append(c instanceof Node ? c : String(c));
    }
    return e;
  }

  const card = (...children) => el('section', {class: 'card'}, ...children);
  const actions = (...buttons) => el('div', {class: 'actions'}, ...buttons);
  const contact = () => INFO.contact || T('organiser');

  function langToggle() {
    const other = lang === 'vi' ? 'en' : 'vi';
    return el('button', {class: 'link', lang: other, onclick: () => {
      lang = other;
      I.saveLang(lang);
      document.documentElement.lang = lang;
      document.title = T('title');
      if (redraw) redraw();
    }}, I.t(other, 'langName'));
  }

  // Every screen has the language switch; a trial keeps its plays and rating when it is redrawn in the other language.
  function show(nodes) {
    stopAudio();
    pendingEl = el('p', {class: 'warn', hidden: !sendFailed}, T('pending'));
    const sample = (D && D.sample) || (!D && INFO.sample);
    const parts = [
      el('header', {class: 'top'},
        el('span', {class: 'title'}, T('title')),
        el('span', {class: 'tools'},
          state ? el('button', {class: 'link', onclick: logout}, T('logout')) : null,
          langToggle())),
      sample ? el('p', {class: 'testdata'}, T('testData')) : null,
      pendingEl,
      notice ? el('p', {class: noticeCls}, notice) : null,
      ...nodes,
      el('footer', {}, T('footer', {contact: contact()})),
    ];
    app.replaceChildren(...parts.flat(Infinity).filter(Boolean));
    notice = '';
    noticeCls = 'ok';
    window.scrollTo(0, 0);
  }

  function fatal(key, vars) {
    redraw = () => fatal(key, vars);
    cur = null;
    show([card(el('h1', {}, T('cannotStart')), el('p', {class: 'error'}, T(key, vars)))]);
  }

  function bar(text, frac, cls) {
    const pct = Math.round(100 * frac);
    const fill = el('div', {class: 'fill'});
    fill.style.width = (100 * frac).toFixed(1) + '%';
    return el('div', {class: 'bar ' + (cls || '')},
      el('div', {class: 'bar-label'}, text),
      el('div', {class: 'track', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100,
                 'aria-valuenow': pct, 'aria-label': text}, fill));
  }

  const fmt = (kind, v) => (kind !== 'cmos' ? String(v) : v > 0 ? '+' + v : v < 0 ? '−' + -v : '0');
  const kindOf = (block) => (block === 'practice' ? 'mos' : block);
  const scaleOf = (kind) => I.scale(lang, kind);
  const labelOf = (kind, v) => (scaleOf(kind).find((x) => x[0] === v) || [v, ''])[1];

  function scaleList(kind) {
    return el('ul', {class: 'scale-preview'},
      scaleOf(kind).map(([v, label]) => el('li', {}, el('b', {}, fmt(kind, v)), ' ', label)));
  }

  // ── server and local copy ─────────────────────────────────
  async function api(path, body) {
    const opt = body === undefined
      ? {method: 'GET', credentials: 'same-origin', headers: {Accept: 'application/json'}}
      : {method: 'POST', credentials: 'same-origin', headers: {'Content-Type': 'application/json'},
         body: JSON.stringify(body)};
    let res;
    try {
      res = await fetch(path, opt);
    } catch (e) {
      const err = new Error('network');
      err.network = true;
      throw err;
    }
    let data = null;
    try { data = await res.json(); } catch (e) { /* not JSON */ }
    if (!res.ok) {
      const err = new Error((data && data.error && data.error.message) || 'HTTP ' + res.status);
      err.status = res.status;
      err.code = data && data.error && data.error.code;
      throw err;
    }
    return data;
  }

  const localKey = () => 'mos:' + state.raterId + ':' + state.packageId;

  function loadLocal() {
    try {
      const s = localStorage.getItem(localKey());
      return s ? JSON.parse(s) : null;
    } catch (e) {
      return null;
    }
  }

  function saveLocal() {
    if (!state || state.completed) return;
    try {
      localStorage.setItem(localKey(), JSON.stringify({results: state.results, outbox}));
    } catch (e) { /* the server copy is the one that counts */ }
  }

  // A finished session leaves nothing of the rater in this browser (shared computers).
  function clearLocal() {
    try {
      const prefix = 'mos:' + state.raterId + ':';
      for (let i = localStorage.length - 1; i >= 0; i--) {
        const k = localStorage.key(i);
        if (k && k.startsWith(prefix)) localStorage.removeItem(k);
      }
    } catch (e) { /* nothing was stored */ }
  }

  function setPending(on) {
    sendFailed = on;
    if (pendingEl) pendingEl.hidden = !on;
  }

  // Responses go to the server in the order they were given; the server ignores a repeat of one it already has.
  // What the server refuses is never sent again: the page drops its outbox and takes the server's state.
  function queue(path, body) {
    outbox.push({path, body});
    saveLocal();
    flush();
  }

  async function flush() {
    if (flushing || !state) return;
    flushing = true;
    clearTimeout(retryTimer);
    try {
      while (outbox.length && state) {
        const item = outbox[0];
        try {
          await api(item.path, item.body);
        } catch (e) {
          if (!state) return;
          if (e.network || e.status >= 500 || e.status === 429) {
            setPending(true);
            retryTimer = setTimeout(flush, RETRY_MS);
            return;
          }
          // 401 on a repeated finish: the first one reached the server, which then ended the sign-in
          if (!(e.status === 401 && item.path === '/api/finish')) {
            if (e.status === 401) {
              setPending(true);
              expired();
            } else {
              outbox = [];
              saveLocal();
              resync(T('resynced'));
            }
            return;
          }
        }
        if (!state) return;
        outbox.shift();
        if (item.path === '/api/finish') {
          state.completed = true;   // the server has also ended the sign-in cookie
          clearLocal();
        }
        saveLocal();
      }
      const wasFailing = sendFailed;
      setPending(false);
      if (wasFailing && finishShown && redraw) redraw();
    } finally {
      flushing = false;
    }
  }

  const progressCount = (r) => ORDER.reduce((s, b) => s + ((r.responses || {})[b] || []).length, 0) +
    ((r.headphone || {}).attempts || []).reduce((s, a) => s + a.trials.length, 0) + (r.background ? 1 : 0);

  function setPlan(plan) {
    D = plan;
    CFG = plan.config || {};
    const hp = CFG.headphone_check || {};
    HP = {trials: hp.trials ?? 6, pass: hp.pass ?? 5, attempts: hp.attempts ?? 2};
    REQUIRE_FULL = CFG.require_full_play !== false;
  }

  // Takes the server's state: the consent first, then the plan; a local copy with responses the server has not
  // confirmed is kept and sent.
  async function enter(st, welcome) {
    if (!st.package_id) return renderNotOpen();
    state = {raterId: st.results.rater_id, packageId: st.package_id, session: null, results: st.results,
             consented: st.consented, completed: st.completed};
    if (!state.consented) return renderConsent();
    let plan;
    try {
      plan = await api('/api/plan');
    } catch (e) {
      return fatal('serverError', {msg: e.message});
    }
    setPlan(plan);
    state.session = plan.session;
    const local = loadLocal();
    if (local && Array.isArray(local.outbox) && local.outbox.length && local.results &&
        progressCount(local.results) > progressCount(st.results)) {
      state.results = local.results;
      outbox = local.outbox;
    } else {
      outbox = [];
    }
    saveLocal();
    flush();
    const n = answered(state.results);
    if (welcome && n) {
      notice = T('welcomeBack', {n, total: ORDER.reduce((s, b) => s + state.session[b].length, 0)});
    }
    render();
  }

  // Reloads the server's state after a refused response; more than 3 times in 30 s ends the page instead of looping.
  async function resync(message) {
    const now = Date.now();
    resyncs = resyncs.filter((x) => now - x < 30000).concat(now);
    if (resyncs.length > 3) {
      outbox = [];
      saveLocal();
      return fatal('syncFailed', {contact: contact()});
    }
    try {
      const st = await api('/api/state');
      if (!st.auth) return renderCode('expired');
      notice = message;
      noticeCls = 'warn';
      outbox = [];
      await enter(st);
    } catch (e) {
      fatal('serverError', {msg: e.message});
    }
  }

  // The responses not yet sent stay in localStorage and are sent after the code is entered again.
  function expired() {
    cur = null;
    renderCode('expired');
  }

  // "Use another code": the local copy of the rater's answers goes too, unless some are still waiting to be sent
  // (they are sent once the code is entered again), so that the next person at a shared computer finds nothing.
  async function logout() {
    try { await api('/api/logout', {}); } catch (e) { /* the cookie expires anyway */ }
    if (state && !outbox.length) clearLocal();
    state = null;
    D = null;
    outbox = [];
    renderCode();
  }

  // ── flow ──────────────────────────────────────────────────
  // The screen follows from the results, so a reload resumes at the right place (a block with no answer yet shows
  // its introduction again).
  function screen() {
    const R = state.results, S = state.session, hp = R.headphone;
    if (!hp.passed) {
      const last = hp.attempts[hp.attempts.length - 1];
      if (last && last.correct === null) return {name: 'hp-trial'};
      return {name: hp.attempts.length < HP.attempts ? 'hp-intro' : 'hp-fail'};
    }
    for (const block of ORDER) {
      const done = R.responses[block].length;
      if (done < S[block].length) {
        if (done === 0 && !introSeen.has(block)) return {name: 'intro', block};
        if (block !== 'practice' && done === C.midBreak(S[block].length) && !breakSeen.has(block)) {
          return {name: 'break', block};
        }
        return {name: 'trial', block, index: done};
      }
    }
    return R.background ? {name: 'finish'} : {name: 'background'};
  }

  function render() {
    redraw = render;
    finishShown = false;
    const s = screen();
    if (s.name === 'hp-intro') renderHpIntro();
    else if (s.name === 'hp-trial') renderHpTrial();
    else if (s.name === 'hp-fail') renderHpFail();
    else if (s.name === 'intro') renderIntro(s.block);
    else if (s.name === 'break') renderBreak(s.block);
    else if (s.name === 'trial') renderTrial(s.block, s.index);
    else if (s.name === 'background') renderBackground();
    else renderFinish();
  }

  function renderNotOpen() {
    redraw = renderNotOpen;
    show([card(el('h1', {}, T('title')), el('p', {}, T('notOpen', {contact: contact()})))]);
  }

  // The consent comes after the code: until the rater agrees, the server stores only that the code was used.
  async function renderConsent() {
    redraw = renderConsent;
    let c;
    try {
      c = await api('/api/consent');
    } catch (e) {
      return e.network ? fatal('serverDown') : renderNotOpen();
    }
    if (redraw !== renderConsent) return;   // another screen was shown meanwhile
    const box = el('input', {type: 'checkbox'});
    const err = el('p', {class: 'error', hidden: true});
    const next = el('button', {class: 'primary', disabled: true, onclick: async () => {
      next.disabled = true;
      try {
        await api('/api/consent', {lang, sha256: c.sha256});
      } catch (e) {
        if (e.code === 'consent_changed') {
          notice = T('consentChanged');
          noticeCls = 'warn';
          return renderConsent();
        }
        if (e.status === 401) return renderCode('expired');
        err.hidden = false;
        err.textContent = e.network ? T('serverDown') : T('serverError', {msg: e.message});
        next.disabled = !box.checked;
        return;
      }
      try {
        const st = await api('/api/state');
        return st.auth ? enter(st) : renderCode('expired');
      } catch (e) {
        return fatal(e.network ? 'serverDown' : 'serverError', {msg: e.message});
      }
    }}, T('continue'));
    box.addEventListener('change', () => { next.disabled = !box.checked; });
    show([card(
      el('h1', {}, T('consentTitle')),
      el('div', {class: 'consent', html: c.html}),
      el('label', {class: 'check'}, box, T('agree')),
      err,
      actions(next))]);
  }

  function renderCode(messageKey) {
    redraw = () => renderCode(messageKey);
    state = null;
    D = null;
    cur = null;
    hpCur = null;
    finishShown = false;
    const input = el('input', {type: 'text', maxlength: 24, autocomplete: 'off', autocapitalize: 'characters',
                               spellcheck: 'false', placeholder: T('codePlaceholder'), 'aria-label': T('codeTitle'),
                               value: codeDraft || null});
    const note = el('p', {class: 'hint'}, T('codeHint'));
    const err = el('p', {class: 'error', hidden: true});
    const go = el('button', {class: 'primary', disabled: codeDraft.trim().length < 12}, T('continue'));
    input.addEventListener('input', () => {
      codeDraft = input.value;
      go.disabled = input.value.trim().length < 12;   // a code has 12 characters (and maybe a prefix)
    });
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !go.disabled) go.click(); });
    const fail = (e) => {
      err.hidden = false;
      err.textContent = e.status === 429 ? T('tooMany') : e.status === 401 || e.status === 400 ? T('codeInvalid')
        : e.code === 'closed' ? T('regClosed') : e.network ? T('serverDown') : T('serverError', {msg: e.message});
    };
    go.addEventListener('click', async () => {
      go.disabled = true;
      try {
        const st = await api('/api/login', {code: input.value});
        codeDraft = '';
        await enter(st, true);
      } catch (e) {
        fail(e);
        go.disabled = false;
      }
    });
    const buttons = [go];
    if (INFO.open_registration) {
      const fresh = el('button', {class: 'secondary', onclick: async () => {
        fresh.disabled = true;
        try {
          renderNewCode(await api('/api/register', {}));
        } catch (e) {
          fail(e);
          if (e.code === 'closed') {
            INFO.open_registration = false;   // registration was switched off, or every code has been issued
            fresh.hidden = true;
          } else fresh.disabled = false;
        }
      }}, T('startNew'));
      buttons.unshift(fresh);
    }
    show([card(
      el('h1', {}, T('codeTitle')),
      messageKey ? el('p', {class: 'warn'}, T(messageKey)) : null,
      el('p', {}, T('codeText')),
      input, note, err, actions(...buttons))]);
    input.focus();
  }

  const answered = (r) => ORDER.reduce((s, b) => s + r.responses[b].length, 0);

  // The only time a code from open registration is shown; the server keeps only its HMAC.
  function renderNewCode(st) {
    redraw = () => renderNewCode(st);
    show([card(
      el('h1', {}, T('newCodeTitle')),
      el('p', {class: 'code'}, st.code),
      el('p', {}, T('newCodeText')),
      actions(el('button', {class: 'primary', onclick: () => enter(st, false)}, T('continue'))))]);
  }

  // ── headphone check (Woods et al., 2017) ──────────────────
  function stopAudio() {
    for (const a of pool.values()) if (!a.paused) a.pause();
    if (hpSource) {
      const s = hpSource;
      hpSource = null;
      try { s.stop(); } catch (e) { /* already stopped */ }
    }
  }

  function playTones(order, onEnd) {
    try {
      actx = actx || new (window.AudioContext || window.webkitAudioContext)();
      if (actx.state === 'suspended') actx.resume();
      stopAudio();
      const sig = C.headphoneSignal(order, actx.sampleRate);
      const buf = actx.createBuffer(2, sig.left.length, actx.sampleRate);
      buf.getChannelData(0).set(sig.left);
      buf.getChannelData(1).set(sig.right);
      const src = actx.createBufferSource();
      src.buffer = buf;
      src.connect(actx.destination);
      src.onended = () => {
        if (hpSource !== src) return;  // stopped because the screen changed
        hpSource = null;
        if (onEnd) onEnd();
      };
      hpSource = src;
      src.start();
    } catch (e) {
      fatal('audioCtxError', {msg: e.message, contact: contact()});
    }
  }

  function renderHpIntro() {
    const R = state.results, n = R.headphone.attempts.length;
    const calib = el('button', {class: 'play', onclick: () => playTones(['normal', 'normal', 'normal'])},
                     PLAY + ' ' + T('playTest'));
    const go = el('button', {class: 'primary', onclick: () => {
      R.headphone.attempts.push({trials: [], correct: null});
      saveLocal();
      render();
    }}, n ? T('tryAgain') : T('hpStart'));
    show([card(
      el('h1', {}, T('hpTitle')),
      n ? el('p', {class: 'warn'}, T('hpFailedOnce', {n: n + 1, max: HP.attempts})) : null,
      el('p', {}, T('hpIntro')),
      el('ol', {},
        el('li', {}, T('hpStep1')),
        el('li', {}, T('hpStep2')),
        el('li', {}, T('hpStep3', {n: HP.trials}))),
      el('div', {class: 'clips'}, calib),
      actions(go))]);
  }

  // The tones of a trial play once; a redraw in the other language keeps whether they were played and the choice.
  function renderHpTrial() {
    const R = state.results, attempts = R.headphone.attempts;
    const a = attempts.length - 1, attempt = attempts[a], i = attempt.trials.length;
    const trial = C.headphoneTrials(HP.trials, C.rng(state.session.raterId + ':headphone:' + a))[i];
    if (!hpCur || hpCur.attempt !== a || hpCur.i !== i) hpCur = {attempt: a, i, started: false, choice: null};
    const h = hpCur;
    const next = el('button', {class: 'primary', disabled: h.choice === null, onclick: () => {
      attempt.trials.push({order: trial.order, answer: trial.answer, choice: h.choice});
      if (attempt.trials.length >= HP.trials) {
        attempt.correct = attempt.trials.filter((t) => t.choice === t.answer).length;
        R.headphone.passed = attempt.correct >= HP.pass;
        if (R.headphone.passed) notice = T('hpPassed');
      }
      queue('/api/headphone', {attempt: a, trial: i, choice: h.choice});
      render();
    }}, T('next'));
    const options = ['tone1', 'tone2', 'tone3'].map((label, k) => el('button', {
      class: 'rate' + (h.choice === k ? ' selected' : ''), disabled: !h.started, 'aria-pressed': String(h.choice === k),
      onclick: () => {
        h.choice = k;
        options.forEach((o, j) => { o.classList.toggle('selected', j === k); o.setAttribute('aria-pressed', j === k); });
        next.disabled = false;
        next.focus();
      }}, el('span', {class: 'num'}, k + 1), el('span', {class: 'lab'}, T(label))));
    const play = el('button', {class: 'play', disabled: h.started, onclick: () => {
      h.started = true;
      play.disabled = true;
      play.textContent = T('playing');
      playTones(trial.order, () => {
        play.textContent = CHECK + ' ' + T('played');
        options.forEach((o) => { o.disabled = false; });
      });
    }}, h.started ? CHECK + ' ' + T('played') : PLAY + ' ' + T('play'));
    show([
      el('div', {class: 'progress'}, bar(T('hpProgress', {i: i + 1, n: HP.trials}), i / HP.trials)),
      card(
        el('p', {class: 'question'}, T('hpQuestion')),
        el('p', {class: 'hint'}, T('hpOnce')),
        el('div', {class: 'clips'}, play),
        el('div', {class: 'scale'}, options),
        actions(next))]);
    (h.started ? next : play).focus();
  }

  function renderHpFail() {
    queueFinish();
    renderFinishScreen('hpFailTitle', 'hpFailText');
  }

  // ── block introductions (they double as the breaks) ───────
  function partLabel(block) {
    const parts = ['mos', 'smos', 'cmos'].filter((b) => state.session[b].length);
    return T('partLabel', {i: parts.indexOf(block) + 1, n: parts.length, name: T('name_' + block)});
  }

  function renderIntro(block) {
    const S = state.session, n = S[block].length;
    const fullOne = REQUIRE_FULL ? T('fullOne') : T('fullAdvice');
    const fullBoth = REQUIRE_FULL ? T('fullBoth') : T('fullAdvice');
    const mosText = [
      el('p', {}, T('mosIntro1')),
      el('p', {}, T('mosIntro2', {full: fullOne})),
      scaleList('mos'),
    ];
    const breakText = ORDER.slice(0, ORDER.indexOf(block)).some((b) => S[b].length && b !== 'practice')
      ? el('p', {class: 'ok'}, T('breakText')) : null;
    let body, label = T('start');
    if (block === 'practice') {
      body = [mosText, el('p', {}, T('practiceIntro', {n}))];
      label = T('startPractice');
    } else if (block === 'mos') {
      body = S.practice.length
        ? [el('p', {}, T('mosAfterPractice', {n})), scaleList('mos')]
        : [mosText, el('p', {}, T('mosCount', {n}))];
    } else if (block === 'smos') {
      body = [breakText, el('p', {}, T('smosIntro')), el('p', {}, T('playBoth') + ' ' + fullBoth), scaleList('smos'),
              el('p', {}, T('trialsCount', {n}))];
    } else {
      body = [breakText, el('p', {}, T('cmosIntro')), el('p', {}, T('playBoth') + ' ' + fullBoth), scaleList('cmos'),
              el('p', {}, T('trialsCount', {n}))];
    }
    const start = el('button', {class: 'primary', onclick: () => {
      introSeen.add(block);
      render();
    }}, label);
    const first = ['mos', 'smos', 'cmos'].find((b) => S[b].length);
    show([card(el('h1', {}, partLabel(block === 'practice' ? first : block)), body, actions(start))]);
    preload([[block, 0]]);
    start.focus();
  }

  // The break in the middle of a rating block (after trial n/2, rounded down).
  function renderBreak(block) {
    const go = el('button', {class: 'primary', onclick: () => {
      breakSeen.add(block);
      render();
    }}, T('continue'));
    show([card(el('h1', {}, T('midBreakTitle')), el('p', {}, partLabel(block)), el('p', {class: 'ok'}, T('midBreakText')),
               actions(go))]);
    preload([[block, state.results.responses[block].length]]);
    go.focus();
  }

  // ── rating trials ─────────────────────────────────────────
  function clipSpecs(block, trial) {
    if (block === 'smos') return [{role: 'ref', label: T('clipRef'), sid: trial.ref},
                                  {role: 'stim', label: T('clipSample'), sid: trial.stim}];
    if (block === 'cmos') return [{role: 'a', label: 'A', sid: trial.a}, {role: 'b', label: 'B', sid: trial.b}];
    return [{role: 'stim', label: T('clipPlay'), sid: trial.stim}];
  }

  const audioKey = (block, trial, role) => block + '|' + trial.tid + '|' + role;

  function audioFor(key, sid) {
    let a = pool.get(key);
    if (!a) {
      a = new Audio();
      a.preload = 'auto';
      a.src = D.stimuli[sid].src;
      pool.set(key, a);
    }
    return a;
  }

  function nextPosition(block, index) {
    for (let b = ORDER.indexOf(block), i = index + 1; b < ORDER.length; b++, i = 0) {
      if (i < state.session[ORDER[b]].length) return [ORDER[b], i];
    }
    return null;
  }

  // Loads the audio of the given trials and drops every other element.
  function preload(positions) {
    const keep = new Set();
    for (const pos of positions) {
      if (!pos || pos[1] >= state.session[pos[0]].length) continue;
      const trial = state.session[pos[0]][pos[1]];
      for (const c of clipSpecs(pos[0], trial)) {
        const key = audioKey(pos[0], trial, c.role);
        audioFor(key, c.sid);
        keep.add(key);
      }
    }
    for (const [key, a] of pool) {
      if (keep.has(key)) continue;
      a.onended = a.onpause = a.onerror = null;
      a.pause();
      a.removeAttribute('src');
      a.load();
      pool.delete(key);
    }
  }

  function rewind(a) {
    try { a.currentTime = 0; } catch (e) { /* not loaded yet */ }
  }

  // Adds the time since the clip started to the seconds listened in this trial.
  function stopClock(c) {
    if (c.since == null) return;
    cur.played += (performance.now() - c.since) / 1000;
    c.since = null;
  }

  function renderTrial(block, index) {
    const S = state.session, R = state.results;
    const trial = S[block][index], kind = kindOf(block);
    // the same trial drawn again (a language change): its plays, completed clips, rating and times stay
    const prev = cur && cur.block === block && cur.trial.tid === trial.tid ? cur : null;
    if (prev) {
      for (const c of prev.clips) {
        stopClock(c);
        c.el.onended = c.el.onpause = c.el.onerror = null;
      }
    }
    const clips = clipSpecs(block, trial).map((c, k) => Object.assign(c, {
      el: audioFor(audioKey(block, trial, c.role), c.sid), plays: prev ? prev.clips[k].plays : 0,
      full: prev ? prev.clips[k].full : false, since: null}));
    cur = {block, index, trial, kind, clips, rating: prev ? prev.rating : null, ratedAt: prev ? prev.ratedAt : null,
           playing: null, played: prev ? prev.played : 0};

    for (const c of clips) {
      c.btn = el('button', {class: 'play', onclick: () => toggle(c)});
      c.badge = el('span', {class: 'badge'});
    }
    cur.rateBtns = scaleOf(kind).map(([v, label]) => {
      const b = el('button', {class: 'rate', 'aria-pressed': 'false', onclick: () => choose(v)},
                   el('span', {class: 'num'}, fmt(kind, v)), el('span', {class: 'lab'}, label));
      b.dataset.v = v;
      return b;
    });
    cur.nextBtn = el('button', {class: 'primary', onclick: commit}, T('next'));
    cur.hint = el('p', {class: 'hint'}, clips.length > 1 ? T('hintBoth') : T('hintOne'));
    cur.errEl = el('p', {class: 'error', hidden: true});

    const total = ORDER.reduce((s, b) => s + S[b].length, 0);
    const done = ORDER.reduce((s, b) => s + R.responses[b].length, 0);
    const head = block === 'practice' ? T('practice') : partLabel(block);
    const attention = C.isAttention(trial)
      ? T(kind === 'mos' ? 'att_one' : 'att_pair', {r: fmt(kind, trial.attention) + ' (' + labelOf(kind, trial.attention) + ')'})
      : null;
    show([
      el('div', {class: 'progress'},
        bar(T('trialProgress', {head, i: index + 1, n: S[block].length}), index / S[block].length),
        bar(T('overall', {p: Math.round(100 * done / total)}), done / total, 'overall')),
      el('section', {class: 'card trial'},
        el('p', {class: 'question'}, T('q_' + kind)),
        attention ? el('p', {class: 'instruction'}, attention) : null,
        el('div', {class: 'clips'}, clips.map((c) => el('div', {class: 'clip'}, c.btn, c.badge))),
        cur.errEl, cur.hint,
        el('div', {class: 'scale ' + kind}, cur.rateBtns),
        actions(cur.nextBtn))]);

    for (const c of clips) {
      const a = c.el;
      a.onended = () => {
        c.full = true;
        stopClock(c);
        if (cur && cur.playing === c) cur.playing = null;
        refresh();
      };
      a.onpause = () => {
        if (cur && cur.playing === c) { stopClock(c); cur.playing = null; refresh(); }
      };
      a.onerror = () => audioError(c);
      if (a.error) audioError(c);
    }
    cur.shownAt = prev ? prev.shownAt : performance.now();
    refresh();
    preload([[block, index], nextPosition(block, index)]);
    clips[0].btn.focus();
  }

  function refresh() {
    if (!cur) return;
    const ready = !REQUIRE_FULL || cur.clips.every((c) => c.full);
    for (const c of cur.clips) {
      const playing = cur.playing === c;
      c.btn.textContent = playing ? STOP + ' ' + T('stop') : PLAY + ' ' + c.label;
      c.btn.classList.toggle('playing', playing);
      c.badge.textContent = c.full ? CHECK + ' ' + T('playedBadge') : '';
    }
    for (const b of cur.rateBtns) {
      const selected = Number(b.dataset.v) === cur.rating;
      b.disabled = !ready;
      b.classList.toggle('selected', selected);
      b.setAttribute('aria-pressed', selected);
    }
    cur.hint.hidden = ready;
    cur.nextBtn.disabled = cur.rating === null;
  }

  // Only one clip plays at a time; pressing the playing clip's button stops it.
  function toggle(c) {
    if (cur.playing === c) {
      stopClock(c);
      cur.playing = null;
      c.el.pause();
      rewind(c.el);
      refresh();
      return;
    }
    for (const o of cur.clips) {
      if (o !== c) { stopClock(o); o.el.pause(); rewind(o.el); }
    }
    rewind(c.el);
    cur.playing = c;
    c.plays++;
    c.since = performance.now();
    const p = c.el.play();
    if (p && p.catch) {
      p.catch((e) => {
        if (e.name === 'AbortError' || !cur || !cur.clips.includes(c)) return;
        c.plays--;
        c.since = null;
        if (cur.playing === c) cur.playing = null;
        audioError(c);
        refresh();
      });
    }
    refresh();
  }

  function audioError(c) {
    if (!cur || !cur.clips.includes(c)) return;
    cur.errEl.hidden = false;
    cur.errEl.textContent = T('audioError', {contact: contact()});
  }

  function choose(v) {
    if (!cur || cur.rateBtns[0].disabled) return;
    cur.rating = v;
    cur.ratedAt = performance.now();
    refresh();
    cur.nextBtn.focus();
  }

  function commit() {
    if (!cur || cur.rating === null) return;
    const {block, trial, clips} = cur, R = state.results;
    for (const c of clips) stopClock(c);
    const plays = Object.fromEntries(clips.map((c) => [c.role, c.plays]));
    let rec;
    if (block === 'smos') {
      rec = {tid: trial.tid, stim: trial.stim, ref: trial.ref, rating: cur.rating,
             plays_ref: plays.ref, plays_stim: plays.stim};
    } else if (block === 'cmos') {
      rec = {tid: trial.tid, a: trial.a, b: trial.b, rating: cur.rating, plays_a: plays.a, plays_b: plays.b};
    } else {
      rec = {tid: trial.tid, stim: trial.stim, rating: cur.rating, plays: plays.stim};
    }
    const now = performance.now();
    const tMs = Math.round(now - cur.shownAt);
    rec.rt_ms = Math.round(cur.ratedAt - cur.shownAt);
    rec.t = new Date().toISOString();
    rec.lang = lang;
    if (C.isAttention(trial)) rec.attention = trial.attention;
    R.responses[block].push(rec);
    R.block_ms[block] += tMs;
    queue('/api/answer', {tid: trial.tid, block, value: cur.rating, lang, plays: clips.length > 1 ? plays : plays.stim,
                          played_s: Math.round(cur.played * 100) / 100, rt_ms: rec.rt_ms, t_ms: tMs});
    for (const c of clips) c.el.onended = c.el.onpause = c.el.onerror = null;
    cur = null;
    // finished = the last rating, so the background questions do not count towards the session length
    if (['background', 'finish'].includes(screen().name) && !R.finished) R.finished = rec.t;
    saveLocal();
    render();
  }

  // ── end screens ───────────────────────────────────────────
  const pad = (n) => String(n).padStart(2, '0');
  const stamp = (d) => d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + '-' + pad(d.getHours()) +
                       pad(d.getMinutes());

  // Only offered while responses have not reached the server: the same results file as the journal page.
  function downloadButton() {
    const msg = el('p', {class: 'hint'});
    const dl = el('button', {class: 'secondary', onclick: () => {
      const name = 'listening_' + state.results.rater_id + '_' + stamp(new Date()) + '.json';
      const blob = new Blob([JSON.stringify(state.results, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = el('a', {href: url, download: name});
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      msg.textContent = T('savedAs', {name});
    }}, T('download'));
    return [el('div', {class: 'actions left'}, dl), msg];
  }

  function renderBackground() {
    const answers = bgDraft;
    const next = el('button', {class: 'primary', disabled: C.checkBackground(answers).length > 0, onclick: () => {
      state.results.background = Object.assign({}, answers);
      bgDraft = {};
      queue('/api/background', state.results.background);
      render();
    }}, T('continue'));
    const groups = C.BACKGROUND.map((q) => el('fieldset', {class: 'bg'},
      el('legend', {}, I.bgQuestion(lang, q)),
      q.options.concat([C.NO_ANSWER]).map((opt) => {
        const input = el('input', {type: 'radio', name: 'bg-' + q.key, value: opt, checked: answers[q.key] === opt});
        input.addEventListener('change', () => {
          answers[q.key] = opt;
          next.disabled = C.checkBackground(answers).length > 0;
        });
        return el('label', {class: 'check'}, input, I.bgOption(lang, opt));
      })));
    show([card(
      el('h1', {}, T('bgTitle')),
      el('p', {}, T('bgText', {no: I.bgOption(lang, C.NO_ANSWER)})),
      groups,
      actions(next))]);
  }

  function renderFinish() {
    queueFinish();
    renderFinishScreen('finishTitle', 'finishText');
  }

  // Tells the server once per page load that the session is over (it ignores a repeat).
  function queueFinish() {
    if (state.completed || state.finishQueued) return;
    state.finishQueued = true;
    queue('/api/finish', {});
  }

  function renderFinishScreen(titleKey, textKey) {
    redraw = () => renderFinishScreen(titleKey, textKey);
    finishShown = true;
    show([card(
      el('h1', {}, T(titleKey)),
      el('p', {}, T(textKey)),
      outbox.length && sendFailed ? [el('p', {class: 'warn'}, T('finishPending', {contact: contact()})),
                                     downloadButton()] : null)]);
  }

  // ── start ─────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (!cur || cur.kind === 'cmos' || e.altKey || e.ctrlKey || e.metaKey || e.target.tagName === 'INPUT') return;
    if (/^[1-5]$/.test(e.key)) {
      e.preventDefault();
      choose(Number(e.key));
    }
  });

  async function init() {
    app = document.getElementById('app');
    if (!C || !I) {
      app.textContent = (C ? 'i18n.js' : 'lt_core.js') + ' is missing.';
      return;
    }
    document.documentElement.lang = lang;
    document.title = T('title');
    try {
      INFO = await api('/api/info');
    } catch (e) {
      return fatal('serverDown');
    }
    if (!INFO.open) return renderNotOpen();
    let st;
    try {
      st = await api('/api/state');
    } catch (e) {
      return fatal(e.network ? 'serverDown' : 'serverError', {msg: e.message});
    }
    if (st.auth) return enter(st, true);
    renderCode();
  }

  init();
})();
