/* Listening test UI: screens, autosave and resume, audio playback, the headphone check and the results file.
   Needs window.LT_DATA (lt_data.js) and window.LTCore (lt_core.js). index.html?sample loads sample_lt_data.js. */
(() => {
  'use strict';
  const C = window.LTCore;
  const ORDER = ['practice', 'mos', 'smos', 'cmos'];
  const NAMES = {mos: 'Naturalness', smos: 'Speaker similarity', cmos: 'Comparison'};
  const QUESTIONS = {
    mos: 'How natural does the speech in this recording sound?',
    smos: 'How similar is the voice in the Sample to the voice in the Reference? Ignore the words and the audio ' +
          'quality.',
    cmos: 'Compared with A, the quality of B is:',
  };
  const SCALES = {
    mos: [[5, 'Excellent'], [4, 'Good'], [3, 'Fair'], [2, 'Poor'], [1, 'Bad']],
    smos: [[5, 'Definitely the same speaker'], [4, 'Probably the same'], [3, 'Not sure'],
           [2, 'Probably different'], [1, 'Definitely different speakers']],
    cmos: [[3, 'Much better'], [2, 'Better'], [1, 'Slightly better'], [0, 'About the same'],
           [-1, 'Slightly worse'], [-2, 'Worse'], [-3, 'Much worse']],
  };
  const ID_RE = /^[A-Za-z0-9_-]{2,20}$/;
  const PLAY = '▶︎', STOP = '■︎', CHECK = '✓';

  let D, CFG, HP, REQUIRE_FULL;
  let app;
  let state = null;        // {version, session, results}; saved under storeKey after every response
  let storeKey = null;
  let saveFailed = false;
  let notice = '';         // one-off message shown at the top of the next screen
  let cur = null;          // the rating trial on screen
  const introSeen = new Set();  // blocks whose introduction was dismissed in this page load
  let actx = null, hpSource = null;
  const pool = new Map();  // <audio> elements of the current and the next trial, by block|tid|role

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
  const contact = () => CFG.contact || 'the organiser';

  function show(...nodes) {
    stopAudio();
    const parts = [
      el('header', {class: 'top'}, CFG.title || 'Listening test'),
      saveFailed ? el('p', {class: 'warn'}, 'Your progress cannot be saved in this browser. Please keep this ' +
                      'page open until you have downloaded your results.') : null,
      notice ? el('p', {class: 'ok'}, notice) : null,
      ...nodes,
      CFG.contact ? el('footer', {}, 'Questions or problems: ' + CFG.contact) : null,
    ];
    app.replaceChildren(...parts.flat(Infinity).filter(Boolean));
    notice = '';
    window.scrollTo(0, 0);
  }

  function fatal(message) {
    app.replaceChildren(card(el('h1', {}, 'The test cannot start'), el('p', {class: 'error'}, message)));
  }

  function bar(text, frac, cls) {
    const pct = Math.round(100 * frac);
    return el('div', {class: 'bar ' + (cls || '')},
      el('div', {class: 'bar-label'}, text),
      el('div', {class: 'track', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100,
                 'aria-valuenow': pct, 'aria-label': text},
        el('div', {class: 'fill', style: 'width:' + (100 * frac).toFixed(1) + '%'})));
  }

  const fmt = (kind, v) => (kind !== 'cmos' ? String(v) : v > 0 ? '+' + v : v < 0 ? '−' + -v : '0');
  const kindOf = (block) => (block === 'practice' ? 'mos' : block);

  function scaleList(kind) {
    return el('ul', {class: 'scale-preview'},
      SCALES[kind].map(([v, label]) => el('li', {}, el('b', {}, fmt(kind, v)), ' ', label)));
  }

  // ── storage ───────────────────────────────────────────────
  function load(key) {
    try {
      const s = localStorage.getItem(key);
      return s ? JSON.parse(s) : null;
    } catch (e) {
      return null;
    }
  }

  function save() {
    try {
      localStorage.setItem(storeKey, JSON.stringify(state));
      saveFailed = false;
    } catch (e) {
      saveFailed = true;
    }
  }

  // ── flow ──────────────────────────────────────────────────
  // The screen follows from the saved results, so a reload resumes at the right place (a block with no answer
  // yet shows its introduction again).
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
        return done === 0 && !introSeen.has(block) ? {name: 'intro', block} : {name: 'trial', block, index: done};
      }
    }
    return R.background ? {name: 'finish'} : {name: 'background'};
  }

  function render() {
    const s = screen();
    if (s.name === 'hp-intro') renderHpIntro();
    else if (s.name === 'hp-trial') renderHpTrial();
    else if (s.name === 'hp-fail') renderHpFail();
    else if (s.name === 'intro') renderIntro(s.block);
    else if (s.name === 'trial') renderTrial(s.block, s.index);
    else if (s.name === 'background') renderBackground();
    else renderFinish();
  }

  function renderConsent() {
    const box = el('input', {type: 'checkbox'});
    const next = el('button', {class: 'primary', disabled: true, onclick: renderId}, 'Continue');
    box.addEventListener('change', () => { next.disabled = !box.checked; });
    show(card(
      el('h1', {}, 'Information and consent'),
      el('div', {class: 'consent', html: CFG.consent_html || ''}),
      el('label', {class: 'check'}, box, 'I agree'),
      actions(next)));
  }

  function progressText(saved) {
    const n = ORDER.reduce((s, b) => s + saved.results.responses[b].length, 0);
    const total = ORDER.reduce((s, b) => s + saved.session[b].length, 0);
    return n + ' of ' + total + ' answers';
  }

  function idStatus(id) {
    const saved = load('lt:' + id);
    if (!saved) return {kind: 'new'};
    const same = saved.version === 1 && saved.session && saved.results &&
                 JSON.stringify(saved.session) === JSON.stringify(C.buildSession(D, id));
    return {kind: same ? 'resume' : 'stale', saved};
  }

  function renderId() {
    const input = el('input', {type: 'text', maxlength: 20, autocomplete: 'off', autocapitalize: 'off',
                               spellcheck: 'false', placeholder: 'e.g. R07', 'aria-label': 'Rater ID'});
    const note = el('p', {class: 'hint'});
    const go = el('button', {class: 'primary', disabled: true}, 'Continue');
    let status = null;
    const update = () => {
      const id = input.value.trim();
      status = ID_RE.test(id) ? idStatus(id) : null;
      go.disabled = !status;
      go.textContent = status && status.kind === 'resume' ? 'Resume' : 'Continue';
      if (!status) note.textContent = 'Use the ID you were given: 2 to 20 letters, digits, "-" or "_".';
      else if (status.kind === 'resume') {
        note.textContent = 'A saved session was found for this ID (' + progressText(status.saved) + '). Press ' +
                           'Resume to continue where you stopped.';
      } else if (status.kind === 'stale') {
        note.textContent = 'A saved session for this ID belongs to a different version of the test. It is kept ' +
                           'as a backup in this browser, and a new session starts.';
      } else note.textContent = '';
    };
    input.addEventListener('input', update);
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !go.disabled) go.click(); });
    go.addEventListener('click', () => begin(input.value.trim(), status));
    update();
    show(card(
      el('h1', {}, 'Your rater ID'),
      el('p', {}, 'Please enter the rater ID you received from the organiser. Your answers are saved in this ' +
                  'browser under this ID, so you can close the page and come back later.'),
      input, note, actions(go)));
    input.focus();
  }

  function begin(id, status) {
    storeKey = 'lt:' + id;
    if (status.kind === 'resume') {
      state = status.saved;
      notice = 'Welcome back. Your saved answers were loaded.';
    } else {
      if (status.kind === 'stale') {
        try {
          localStorage.setItem('lt-backup:' + id + ':' + Date.now(), JSON.stringify(status.saved));
        } catch (e) { /* keep going: the new session matters more */ }
      }
      const session = C.buildSession(D, id);
      state = {version: 1, session, results: C.emptyResults(session, navigator.userAgent, new Date().toISOString())};
    }
    save();
    render();
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
      fatal('This browser cannot play the sound-check tones (' + e.message + '). Please use a current version ' +
            'of Chrome, Firefox or Safari, or contact ' + contact() + '.');
    }
  }

  function renderHpIntro() {
    const R = state.results, n = R.headphone.attempts.length;
    const calib = el('button', {class: 'play', onclick: () => playTones(['normal', 'normal', 'normal'])},
                     PLAY + ' Play test sound');
    const go = el('button', {class: 'primary', onclick: () => {
      R.headphone.attempts.push({trials: [], correct: null});
      save();
      render();
    }}, n ? 'Try again' : 'Start the sound check');
    show(card(
      el('h1', {}, 'Sound check'),
      n ? el('p', {class: 'warn'}, 'The sound check was not passed. Please make sure that you are wearing ' +
             'headphones with both sides on your ears, and that the sound does not come from the computer\'s ' +
             'loudspeakers. Then try once more (attempt ' + (n + 1) + ' of ' + HP.attempts + ').') : null,
      el('p', {}, 'This test must be done with headphones (wired or wireless), not loudspeakers, in a quiet ' +
                  'place. It starts with a short check of your headphones.'),
      el('ol', {},
        el('li', {}, 'Put on your headphones.'),
        el('li', {}, 'Press "Play test sound" and set the volume of your computer to a comfortable level. You ' +
                     'can play it as often as you like.'),
        el('li', {}, 'In each of the ' + HP.trials + ' trials you will hear three tones. Choose the one that ' +
                     'is quietest. Each trial can be played only once.')),
      el('div', {class: 'clips'}, calib),
      actions(go)));
  }

  function renderHpTrial() {
    const R = state.results, attempts = R.headphone.attempts;
    const attempt = attempts[attempts.length - 1], i = attempt.trials.length;
    const trial = C.headphoneTrials(HP.trials, C.rng(state.session.raterId + ':headphone:' + (attempts.length - 1)))[i];
    let choice = null;
    const next = el('button', {class: 'primary', disabled: true, onclick: () => {
      attempt.trials.push({order: trial.order, answer: trial.answer, choice});
      if (attempt.trials.length >= HP.trials) {
        attempt.correct = attempt.trials.filter((t) => t.choice === t.answer).length;
        R.headphone.passed = attempt.correct >= HP.pass;
        if (R.headphone.passed) notice = 'Sound check passed. Thank you.';
      }
      save();
      render();
    }}, 'Next');
    const options = ['First', 'Second', 'Third'].map((label, k) => el('button', {
      class: 'rate', disabled: true, 'aria-pressed': 'false', onclick: () => {
        choice = k;
        options.forEach((o, j) => { o.classList.toggle('selected', j === k); o.setAttribute('aria-pressed', j === k); });
        next.disabled = false;
        next.focus();
      }}, el('span', {class: 'num'}, k + 1), el('span', {class: 'lab'}, label + ' tone')));
    const play = el('button', {class: 'play', onclick: () => {
      play.disabled = true;
      play.textContent = 'Playing…';
      playTones(trial.order, () => {
        play.textContent = CHECK + ' Played';
        options.forEach((o) => { o.disabled = false; });
      });
    }}, PLAY + ' Play');
    show(
      el('div', {class: 'progress'}, bar('Sound check: trial ' + (i + 1) + ' of ' + HP.trials, i / HP.trials)),
      card(
        el('p', {class: 'question'}, 'Which of the three tones was the quietest?'),
        el('p', {class: 'hint'}, 'You can play the tones only once.'),
        el('div', {class: 'clips'}, play),
        el('div', {class: 'scale'}, options),
        actions(next)));
    play.focus();
  }

  function renderHpFail() {
    show(card(
      el('h1', {}, 'Thank you for your time'),
      el('p', {}, 'This test has to be done over headphones, and the sound check was not passed, so the test ' +
                  'ends here. This usually depends on the equipment or the sound settings, not on you.'),
      el('p', {}, 'Please download the results file and send it to ' + contact() + ', so that the organiser ' +
                  'knows you took part.'),
      resultsButtons()));
  }

  // ── block introductions (they double as the breaks) ───────
  function partLabel(block) {
    const parts = ['mos', 'smos', 'cmos'].filter((b) => state.session[b].length);
    return 'Part ' + (parts.indexOf(block) + 1) + ' of ' + parts.length + ': ' + NAMES[block];
  }

  function renderIntro(block) {
    const S = state.session, n = S[block].length;
    const fullPlay = REQUIRE_FULL ? 'The rating buttons become active once the recording has played to the end.'
                                  : 'Please listen to each recording to the end before you rate it.';
    const mosText = [
      el('p', {}, 'You will hear short recordings of speech, one at a time. Some are recordings of real people; ' +
                  'others were produced by computer programs.'),
      el('p', {}, 'For each recording, rate how natural the speech sounds, that is, how much it sounds like a ' +
                  'real person speaking. Ignore what is being said. ' + fullPlay + ' You may play a recording ' +
                  'again before you rate it.'),
      scaleList('mos'),
    ];
    const breakText = ORDER.slice(0, ORDER.indexOf(block)).some((b) => S[b].length)
      ? el('p', {class: 'ok'}, 'This part is complete. Take a short break if you like; the test continues ' +
                               'when you press Start.') : null;
    let body, label = 'Start';
    if (block === 'practice') {
      body = [mosText, el('p', {}, 'We start with ' + n + ' practice recordings, so that you can get used to ' +
                                   'the scale. They do not count.')];
      label = 'Start practice';
    } else if (block === 'mos') {
      body = S.practice.length
        ? [el('p', {}, 'The practice is finished. The test itself starts now: ' + n + ' recordings, rated in ' +
                       'the same way.'), scaleList('mos')]
        : [mosText, el('p', {}, 'There are ' + n + ' recordings.')];
    } else if (block === 'smos') {
      body = [breakText,
        el('p', {}, 'In each trial you hear two recordings: a Reference and a Sample. Rate how similar the voice ' +
                    'in the Sample is to the voice in the Reference: could it be the same person speaking? ' +
                    'Ignore the words and the audio quality.'),
        el('p', {}, 'Play both recordings. ' + fullPlay.replace('the recording has', 'both have')),
        scaleList('smos'),
        el('p', {}, 'There are ' + n + ' trials.')];
    } else {
      body = [breakText,
        el('p', {}, 'In each trial you hear two versions, A and B, of the same sentence. Compare the overall ' +
                    'quality of B with the quality of A, from +3 (B is much better) to −3 (B is much worse). ' +
                    'If you hear no difference, choose 0 (About the same).'),
        el('p', {}, 'Play both recordings. ' + fullPlay.replace('the recording has', 'both have')),
        scaleList('cmos'),
        el('p', {}, 'There are ' + n + ' trials.')];
    }
    const start = el('button', {class: 'primary', onclick: () => {
      introSeen.add(block);
      render();
    }}, label);
    show(card(el('h1', {}, partLabel(block === 'practice' ? 'mos' : block)), body, actions(start)));
    preload([[block, 0]]);
    start.focus();
  }

  // ── rating trials ─────────────────────────────────────────
  function clipSpecs(block, trial) {
    if (block === 'smos') return [{role: 'ref', label: 'Reference', sid: trial.ref},
                                  {role: 'stim', label: 'Sample', sid: trial.stim}];
    if (block === 'cmos') return [{role: 'a', label: 'A', sid: trial.a}, {role: 'b', label: 'B', sid: trial.b}];
    return [{role: 'stim', label: 'Play', sid: trial.stim}];
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

  function renderTrial(block, index) {
    const S = state.session, R = state.results;
    const trial = S[block][index], kind = kindOf(block);
    const clips = clipSpecs(block, trial).map((c) => Object.assign(c, {
      el: audioFor(audioKey(block, trial, c.role), c.sid), plays: 0, full: false}));
    cur = {block, index, trial, kind, clips, rating: null, ratedAt: null, playing: null};

    for (const c of clips) {
      c.btn = el('button', {class: 'play', onclick: () => toggle(c)});
      c.badge = el('span', {class: 'badge'});
    }
    cur.rateBtns = SCALES[kind].map(([v, label]) => {
      const b = el('button', {class: 'rate', 'aria-pressed': 'false', onclick: () => choose(v)},
                   el('span', {class: 'num'}, fmt(kind, v)), el('span', {class: 'lab'}, label));
      b.dataset.v = v;
      return b;
    });
    cur.nextBtn = el('button', {class: 'primary', onclick: commit}, 'Next');
    cur.hint = el('p', {class: 'hint'}, clips.length > 1 ? 'Play both recordings to the end to enable the rating.'
                                                         : 'Play the recording to the end to enable the rating.');
    cur.errEl = el('p', {class: 'error', hidden: true});

    const total = ORDER.reduce((s, b) => s + S[b].length, 0);
    const done = ORDER.reduce((s, b) => s + R.responses[b].length, 0);
    const head = block === 'practice' ? 'Practice' : partLabel(block);
    show(
      el('div', {class: 'progress'},
        bar(head + ': trial ' + (index + 1) + ' of ' + S[block].length, index / S[block].length),
        bar('Overall: ' + Math.round(100 * done / total) + '% done', done / total, 'overall')),
      el('section', {class: 'card trial'},
        el('p', {class: 'question'}, QUESTIONS[kind]),
        C.isAttention(trial) && trial.instruction ? el('p', {class: 'instruction'}, trial.instruction) : null,
        el('div', {class: 'clips'}, clips.map((c) => el('div', {class: 'clip'}, c.btn, c.badge))),
        cur.errEl, cur.hint,
        el('div', {class: 'scale ' + kind}, cur.rateBtns),
        actions(cur.nextBtn)));

    for (const c of clips) {
      const a = c.el;
      a.onended = () => {
        c.full = true;
        if (cur && cur.playing === c) cur.playing = null;
        refresh();
      };
      a.onpause = () => {
        if (cur && cur.playing === c) { cur.playing = null; refresh(); }
      };
      a.onerror = () => audioError(c);
      if (a.error) audioError(c);
    }
    cur.shownAt = performance.now();
    refresh();
    preload([[block, index], nextPosition(block, index)]);
    clips[0].btn.focus();
  }

  function refresh() {
    if (!cur) return;
    const ready = !REQUIRE_FULL || cur.clips.every((c) => c.full);
    for (const c of cur.clips) {
      const playing = cur.playing === c;
      c.btn.textContent = playing ? STOP + ' Stop' : PLAY + ' ' + c.label;
      c.btn.classList.toggle('playing', playing);
      c.badge.textContent = c.full ? CHECK + ' played' : '';
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
      cur.playing = null;
      c.el.pause();
      rewind(c.el);
      refresh();
      return;
    }
    for (const o of cur.clips) {
      if (o !== c) { o.el.pause(); rewind(o.el); }
    }
    rewind(c.el);
    cur.playing = c;
    c.plays++;
    const p = c.el.play();
    if (p && p.catch) {
      p.catch((e) => {
        if (e.name === 'AbortError' || !cur || !cur.clips.includes(c)) return;
        c.plays--;
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
    cur.errEl.textContent = 'A recording could not be played. Please reload the page (your answers so far are ' +
                            'saved). If the problem continues, please contact ' + contact() + '.';
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
    rec.rt_ms = Math.round(cur.ratedAt - cur.shownAt);
    rec.t = new Date().toISOString();
    if (C.isAttention(trial)) rec.attention = trial.attention;
    R.responses[block].push(rec);
    R.block_ms[block] += Math.round(now - cur.shownAt);
    for (const c of clips) c.el.onended = c.el.onpause = c.el.onerror = null;
    cur = null;
    // finished = the last rating, so the background questions do not count towards the session length
    if (['background', 'finish'].includes(screen().name) && !R.finished) R.finished = rec.t;
    save();
    render();
  }

  // ── end screens and the results file ──────────────────────
  const pad = (n) => String(n).padStart(2, '0');
  const stamp = (d) => d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + '-' + pad(d.getHours()) +
                       pad(d.getMinutes());

  function resultsButtons() {
    const dlMsg = el('p', {class: 'hint'});
    const subMsg = el('p', {class: 'hint'});
    const dl = el('button', {class: 'primary', onclick: () => {
      const name = 'listening_' + state.results.rater_id + '_' + stamp(new Date()) + '.json';
      const blob = new Blob([JSON.stringify(state.results, null, 2)], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = el('a', {href: url, download: name});
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      dlMsg.textContent = 'Saved as ' + name + ' (usually in your Downloads folder). Please send this file to ' +
                          contact() + '.';
    }}, 'Download results');
    const buttons = [dl];
    if (typeof CFG.submit_url === 'string' && CFG.submit_url.trim()) {
      const sub = el('button', {class: 'secondary', onclick: () => {
        sub.disabled = true;
        subMsg.className = 'hint';
        subMsg.textContent = 'Sending…';
        fetch(CFG.submit_url.trim(), {method: 'POST', headers: {'Content-Type': 'application/json'},
                                      body: JSON.stringify(state.results)})
          .then((res) => {
            if (!res.ok) throw new Error('the server answered ' + res.status);
            subMsg.className = 'ok';
            subMsg.textContent = 'Your results were sent. Thank you!';
          })
          .catch((e) => {
            subMsg.className = 'error';
            subMsg.textContent = 'Sending failed (' + e.message + '). Please use "Download results" and send ' +
                                 'the file to ' + contact() + '.';
            sub.disabled = false;
          });
      }}, 'Submit');
      buttons.push(sub);
    }
    return [el('div', {class: 'actions left'}, buttons), dlMsg, subMsg];
  }

  function renderBackground() {
    const answers = {};
    const next = el('button', {class: 'primary', disabled: true, onclick: () => {
      state.results.background = Object.assign({}, answers);
      save();
      render();
    }}, 'Continue');
    const groups = C.BACKGROUND.map((q) => el('fieldset', {class: 'bg'},
      el('legend', {}, q.question),
      q.options.concat([C.NO_ANSWER]).map((opt) => {
        const input = el('input', {type: 'radio', name: 'bg-' + q.key, value: opt});
        input.addEventListener('change', () => {
          answers[q.key] = opt;
          next.disabled = C.checkBackground(answers).length > 0;
        });
        return el('label', {class: 'check'}, input, opt);
      })));
    show(card(
      el('h1', {}, 'Five short questions'),
      el('p', {}, 'All your ratings are saved. These questions only help us describe the group of listeners in the ' +
                  'paper. You can answer any of them with "' + C.NO_ANSWER + '".'),
      groups,
      actions(next)));
  }

  function renderFinish() {
    show(card(
      el('h1', {}, 'Thank you!'),
      el('p', {}, 'You have completed the listening test. Thank you for your help.'),
      el('p', {}, 'Please download your results file and send it to ' + contact() + '.'),
      resultsButtons()));
  }

  // ── start ─────────────────────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (!cur || cur.kind === 'cmos' || e.altKey || e.ctrlKey || e.metaKey || e.target.tagName === 'INPUT') return;
    if (/^[1-5]$/.test(e.key)) {
      e.preventDefault();
      choose(Number(e.key));
    }
  });

  function init() {
    D = window.LT_DATA;
    if (!C) return fatal('lt_core.js is missing next to index.html.');
    if (!D) return fatal('The test data (lt_data.js) were not found next to index.html. Please contact the organiser.');
    const problems = C.checkData(D);
    if (problems.length) {
      return fatal('The test data are invalid: ' + problems.slice(0, 5).join('; ') +
                   (problems.length > 5 ? ' (and ' + (problems.length - 5) + ' more)' : '') + '.');
    }
    CFG = D.config || {};
    const hp = CFG.headphone_check || {};
    HP = {trials: hp.trials ?? 6, pass: hp.pass ?? 5, attempts: hp.attempts ?? 2};
    REQUIRE_FULL = CFG.require_full_play !== false;
    renderConsent();
  }

  app = document.getElementById('app');
  if (new URLSearchParams(location.search).has('sample')) {
    const s = document.createElement('script');
    s.src = 'sample_lt_data.js';
    s.onload = init;
    s.onerror = () => fatal('sample_lt_data.js could not be loaded.');
    document.head.append(s);
  } else {
    init();
  }
})();
