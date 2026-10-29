/* Admin page of the listening-test server: status, package import, invite codes, the consent text, raters (by rater
   ID; codes are never listed), withdrawal, export. Every request carries the admin token as a bearer token; the token
   is kept in sessionStorage of this tab only. */
(() => {
  'use strict';
  const KEY = 'mos:admin-token';
  const $ = (id) => document.getElementById(id);
  let token = '';

  function getToken() {
    try { return sessionStorage.getItem(KEY) || ''; } catch (e) { return ''; }
  }

  function setToken(t) {
    token = t;
    try {
      if (t) sessionStorage.setItem(KEY, t);
      else sessionStorage.removeItem(KEY);
    } catch (e) { /* kept in memory only */ }
  }

  async function call(method, path, body, raw) {
    const opt = {method, headers: {Authorization: 'Bearer ' + token}};
    if (body !== undefined) {
      opt.headers['Content-Type'] = 'application/json';
      opt.body = JSON.stringify(body);
    }
    const res = await fetch(path, opt);
    if (!res.ok) {
      let msg = 'HTTP ' + res.status;
      try {
        const d = await res.json();
        msg = (d.error && d.error.message) || msg;
        if (d.error && d.error.details) msg += '\n' + d.error.details.join('\n');
      } catch (e) { /* not JSON */ }
      const err = new Error(msg);
      err.status = res.status;
      throw err;
    }
    return raw ? res : res.json();
  }

  function message(text, cls) {
    const m = $('msg');
    m.className = cls || 'ok';
    m.textContent = text;
    m.hidden = !text;
  }

  function fail(e) {
    if (e.status === 401) return signOut('The token was not accepted.');
    message(e.message, 'error');
  }

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text != null) e.textContent = String(text);
    if (cls) e.className = cls;
    return e;
  }

  function download(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  }

  // ── sections ──────────────────────────────────────────────
  async function loadStatus() {
    const s = await call('GET', '/api/admin/status');
    const dl = $('status');
    dl.replaceChildren();
    const p = s.package;
    const rows = [
      ['Data directory', s.data_dir],
      ['Package', p ? p.id + (p.sample ? ' (TEST DATA: journal sample)' : p.dry_run ? ' (TEST DATA: dry-run build)' : '')
        : 'none imported'],
      ['Imported', p ? p.imported_at : '–'],
      ['Trials', p ? Object.entries(p.trials).map(([b, n]) => b + ' ' + n).join(', ') : '–'],
      ['Halves', p ? String(p.halves) : '–'],
      ['Consent text', s.consent.set ? (s.consent.problems.length ? 'refused: ' + s.consent.problems.join('; ')
        : 'sha256 ' + s.consent.sha256) : (p && p.sample ? 'none (the sample text is shown)' : 'none (raters cannot start)')],
      ['Raters', s.raters.total + ' (' + s.raters.invite + ' invited, ' + s.raters.open + ' registered), ' +
        s.raters.disabled + ' disabled'],
      ['Codes issued', s.codes.issued + ' of at most ' + s.codes.max + ' (both routes, withdrawn ones included); ' +
        s.codes.left + ' left' + (s.codes.left ? '' : ': no invite codes and no registration until the cap is raised')],
      ['Open for raters', String(s.open)],
    ];
    for (const [k, v] of rows) dl.append(el('dt', k), el('dd', v));
    $('openreg').checked = !!s.open_registration;
    $('contact').value = s.contact || '';
    $('maxcodes').value = s.codes.max;
  }

  async function loadRaters() {
    const r = await call('GET', '/api/admin/raters');
    const table = $('raters');
    table.replaceChildren();
    const head = ['rater', 'route', 'half', 'note', 'created', 'first used', 'consent', 'lang', 'browser', 'headphone',
                  'practice', 'mos', 'smos', 'cmos', 'finished', 'completed', 'last seen', ''];
    const tr = el('tr');
    head.forEach((h) => tr.append(el('th', h)));
    table.append(tr);
    let done = 0;
    for (const x of r.raters) {
      if (x.completed_at) done++;
      const row = el('tr', null, x.disabled ? 'disabled' : '');
      const hp = x.headphone ? (x.headphone.passed ? 'passed' : x.headphone.failed ? 'failed' : x.headphone.attempts
        ? 'attempt ' + x.headphone.attempts : '–') : '–';
      const prog = (b) => (x.progress && x.progress[b] ? x.progress[b].answered + '/' + x.progress[b].total : '–');
      const cells = [x.rater_id, x.route, x.half, x.note || '', (x.created_at || '').slice(0, 16),
                     (x.first_used || '').slice(0, 16), (x.consent_at || '').slice(0, 16) || '–', x.consent_lang || '',
                     x.browser || '', hp, prog('practice'), prog('mos'), prog('smos'), prog('cmos'),
                     (x.finished || '').slice(0, 16), (x.completed_at || '').slice(0, 16), (x.last_seen || '').slice(0, 16)];
      cells.forEach((c, i) => row.append(el('td', c, i >= 10 && i <= 13 ? 'num' : '')));
      const td = el('td');
      const tog = el('button', x.disabled ? 'Enable' : 'Disable');
      tog.addEventListener('click', async () => {
        try {
          await call('POST', '/api/admin/raters/disable', {rater_id: x.rater_id, disabled: !x.disabled});
          await loadRaters();
        } catch (e) { fail(e); }
      });
      td.append(tog);
      if (x.consent_at) {
        const js = el('button', 'JSON');
        js.addEventListener('click', async () => {
          try {
            const res = await call('GET', '/api/admin/export/rater?rater=' + encodeURIComponent(x.rater_id), undefined, true);
            download(await res.blob(), 'listening_' + x.rater_id + '.json');
          } catch (e) { fail(e); }
        });
        td.append(' ', js);
      }
      row.append(td);
      table.append(row);
    }
    const h = r.halves, per = (k) => 'A ' + h.A[k] + ', B ' + h.B[k];
    $('raters-sum').textContent = r.raters.length + ' raters, ' + done + ' completed sessions in package ' +
                                  (r.package_id || '–') + '. Per half (disabled raters not counted): codes ' +
                                  per('issued') + '; consented ' + per('consented') + '; all ratings given ' +
                                  per('finished') + '; completed ' + per('completed') + '; sound check failed ' +
                                  per('headphone_failed') + '.';
  }

  async function refreshAll() {
    try {
      await loadStatus();
      await loadRaters();
    } catch (e) { fail(e); }
  }

  function signOut(why) {
    setToken('');
    $('main').hidden = true;
    $('signout').hidden = true;
    $('signin').hidden = false;
    $('signin-msg').hidden = !why;
    $('signin-msg').textContent = why || '';
  }

  async function signIn(t) {
    setToken(t);
    try {
      await loadStatus();
    } catch (e) {
      return signOut(e.status === 401 ? 'The token was not accepted.' : e.message);
    }
    $('signin').hidden = true;
    $('main').hidden = false;
    $('signout').hidden = false;
    message('');
    loadRaters().catch(fail);
  }

  // ── events ────────────────────────────────────────────────
  $('signin-btn').addEventListener('click', () => signIn($('token').value.trim()));
  $('token').addEventListener('keydown', (e) => { if (e.key === 'Enter') signIn($('token').value.trim()); });
  $('signout').addEventListener('click', () => signOut(''));
  $('refresh').addEventListener('click', refreshAll);

  $('openreg').addEventListener('change', async () => {
    try {
      await call('POST', '/api/admin/settings', {open_registration: $('openreg').checked});
      message('Open registration ' + ($('openreg').checked ? 'on' : 'off') + '.');
    } catch (e) { fail(e); }
    loadStatus().catch(fail);
  });

  $('contact-save').addEventListener('click', async () => {
    try {
      await call('POST', '/api/admin/settings', {contact: $('contact').value});
      message('Contact saved.');
    } catch (e) { fail(e); }
  });

  $('maxcodes-save').addEventListener('click', async () => {
    try {
      const r = await call('POST', '/api/admin/settings', {max_codes: Number($('maxcodes').value)});
      message('Code cap set to ' + r.max_codes + '.');
    } catch (e) { fail(e); }
    loadStatus().catch(fail);
  });

  $('import').addEventListener('click', async () => {
    $('import-out').textContent = 'Importing…';
    try {
      const r = await call('POST', '/api/admin/package/import', {force: $('force').checked});
      $('import-out').textContent = JSON.stringify(r, null, 1);
      $('force').checked = false;
      await refreshAll();
    } catch (e) {
      $('import-out').textContent = '';
      fail(e);
    }
  });

  $('create').addEventListener('click', async () => {
    try {
      const r = await call('POST', '/api/admin/codes', {n: Number($('n').value), prefix: $('prefix').value.trim(),
                                                        note: $('note').value.trim()});
      $('codes').value = r.codes.map((c) => c.code + '\t' + c.rater_id + '\thalf ' + c.half).join('\n');
      message(r.codes.length + ' codes created.');
      await refreshAll();
    } catch (e) { fail(e); }
  });

  $('consent-load').addEventListener('click', async () => {
    try {
      $('consent-text').value = (await call('GET', '/api/admin/consent')).html || '';
    } catch (e) { fail(e); }
  });

  $('consent-save').addEventListener('click', async () => {
    try {
      const r = await call('POST', '/api/admin/consent', {html: $('consent-text').value});
      message('Consent text saved, sha256 ' + r.sha256 + '.');
      await loadStatus();
    } catch (e) { fail(e); }
  });

  let found = null;
  $('wd-find').addEventListener('click', async () => {
    found = null;
    $('wd-delete').hidden = true;
    try {
      const r = await call('POST', '/api/admin/raters/find', {code: $('wd-code').value});
      found = r.rater_id;
      $('wd-out').textContent = r.rater_id + ' (' + r.route + ', half ' + r.half + (r.disabled ? ', disabled' : '') + ')';
      $('wd-delete').textContent = 'Delete ' + r.rater_id;
      $('wd-delete').hidden = false;
    } catch (e) {
      $('wd-out').textContent = e.status === 404 ? 'No rater has this code.' : '';
      if (e.status !== 404) fail(e);
    }
  });

  $('wd-delete').addEventListener('click', async () => {
    if (!found || !window.confirm('Delete ' + found + ' and every answer? This cannot be undone.')) return;
    try {
      const r = await call('POST', '/api/admin/raters/delete', {rater_id: found});
      message(r.rater_id + ' deleted; ' + r.export_files + ' export files and ' + r.backup_files + ' backups changed' +
              (r.wal_cleared ? '.' : '. The write-ahead log could not be emptied yet: delete again or restart the server.'),
              r.wal_cleared ? 'ok' : 'error');
      found = null;
      $('wd-code').value = '';
      $('wd-out').textContent = '';
      $('wd-delete').hidden = true;
      await refreshAll();
    } catch (e) { fail(e); }
  });

  $('export').addEventListener('click', async () => {
    try {
      const r = await call('POST', '/api/admin/export', {completed_only: $('completed-only').checked});
      $('export-out').textContent = r.raters + ' result files and answers.csv written to\n' + r.dir;
    } catch (e) { fail(e); }
  });

  $('csv').addEventListener('click', async () => {
    try {
      const q = $('completed-only').checked ? '?completed_only=1' : '';
      const res = await call('GET', '/api/admin/export/answers.csv' + q, undefined, true);
      download(await res.blob(), 'answers.csv');
    } catch (e) { fail(e); }
  });

  const saved = getToken();
  if (saved) signIn(saved);
})();
