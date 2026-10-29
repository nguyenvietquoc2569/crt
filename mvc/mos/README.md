# MOS / SMOS listening test server

Web app for the MOS (naturalness, 5-point ACR, ITU-T P.800/P.808) and SMOS (similarity to the target speaker, 5 points)
listening test of MimiFlow-VC under `mvc/LISTENING_ADDENDUM_mvc_v1.md` (§5 scales and wording, §6 session, §7 raters,
§11 website and data handling). It reuses the journal's tested page (`benchmark/listening/page/`): `lt_core.js` is
copied with four changes (background question 6, the half given by the server, no item on two consecutive main trials,
the break inside a block), and `lt_app.js` is adapted so that trials come from the server and every answer is stored
there at once. Bun + bun:sqlite, no dependencies. The server binds to 127.0.0.1 only; the public address
(https://mos.betaschool.edu.vn) is a Cloudflare tunnel in front of it.

The addendum must be frozen before any stimulus is built (`mvc/ANALYSIS_PLAN_mvc_v1.md` §6.4). This server only
collects answers; it never reads the condition key.

| Path | Role |
|---|---|
| `server/main.ts` | entry point (environment below), graceful shutdown on SIGINT/SIGTERM |
| `server/app.ts` | HTTP: rater API, admin API, stimuli, static pages, the host check, security headers, rate limits, the code cap, start checks, daily backup, withdrawal |
| `server/core.ts` | LT_DATA checks and items, sessions and plans, the results file, codes and their HMAC, the address key of the limits (IPv6 by /64), browser family, CSV, sample audio |
| `server/db.ts` | SQLite schema and migrations (WAL, secure_delete), the withdrawal from the database, its log and its copies |
| `server/*.test.ts`, `server/test_util.ts`, `server/fake_page.ts` | `bun test`; `fake_page.ts` runs the rater page against the server in a minimal fake browser |
| `public/` | rater page (`index.html`, `i18n.js`, `lt_core.js`, `lt_app.js`, `style.css`), admin page (`admin.html`, `admin.js`), the journal's `sample_lt_data.js`, node tests |
| `consent_template_v1.html` | the consent text with the placeholders [CONTACT_NAME], [CONTACT_EMAIL], [ETHICS_NOTE], [RETENTION_DATE]; the filled copy lives only in the data directory |

## Run

```
cd mvc/mos
bun run start                 # data in ~/mos_data, http://127.0.0.1:8790
MOS_SAMPLE=1 bun run start    # no package yet: the journal's sample data with synthetic tones, shown as TEST DATA
bun run dev                   # as MOS_SAMPLE=1, with the cookie allowed over plain http
bun run test                  # bun test ./server && node --test public/lt_core.test.js public/i18n.test.js
```

The builder's tests: `.venv-stats/bin/python -m pytest -q mvc/mos/tests` from the repository root.
No launchd job is used (addendum §11.1): the server (`bun run start`) and cloudflared run in Terminal tabs on the Mac,
and a stopped one is restarted by hand.

| Variable | Default | Meaning |
|---|---|---|
| `MOS_DATA_DIR` | `~/mos_data` | all runtime data (below); set to mode 700 at every start |
| `MOS_PORT` | `8790` | port on 127.0.0.1 |
| `MOS_HOSTS` | `mos.betaschool.edu.vn` | host names served besides 127.0.0.1, localhost and [::1] (comma-separated, any port); a request with another `Host` gets 421 |
| `MOS_SAMPLE` | off | `1`: while no package is imported, serve the sample data (TEST DATA banner). Dropped at the next start without it |
| `MOS_COOKIE_SECURE` | on | `0` only for testing over plain http in Safari (Chrome and Firefox accept Secure cookies on 127.0.0.1) |
| `MOS_TRUST_PROXY` | on | take the client address of the limits from `CF-Connecting-IP` (cloudflared); kept in memory only |
| `MOS_LOGIN_LIMIT` | `100` | failed logins per IPv4 address or IPv6 /64 per 10 minutes (successful ones do not count, so a lab or campus behind one address is fine); all logins and registrations of one address are capped at 10 times this |
| `MOS_REGISTER_LIMIT` | `30` | open registrations per 10 minutes from all clients together |

## Data directory (never in the repository)

```
$MOS_DATA_DIR/                      mode 700
  mos.sqlite (+ -wal, -shm)         raters (rater ID, HMAC of the code, route, half), rating sessions (from the consent
                                    on), answers, sign-ins, settings; no IP address and no access code
  admin_token                       admin bearer token, mode 600, created on the first start; read it from the file
  server_secret                     key of the code HMAC, mode 600, created on the first start; never printed or served
  consent_v1.html                   the filled consent_template_v1.html (Vietnamese and English), shown as it is
  package/lt_data.json              LT_DATA version 1 (or the journal builder's lt_data.js)
  package/stimuli/<sid>.flac        the clips (flac, wav, mp3, ogg, opus, m4a)
  private/                          key.json, items.json, cache/: written by build_stimuli.py; the server never opens it
  export/<package>_<time>/          results/listening_<rater ID>.json and answers.csv
  backup/mos-<YYYYMMDD>.sqlite      the daily copy of the database (mode 600), made at start and once a day
```

The server does not start while `package/` holds a file named `key*.json` (two levels deep), and the import refuses any
file with "key" in its name at the top of `package/`. Stimulus ids are 6-64 and trial and item ids 1-64 letters,
digits, `-` or `_`, starting with a letter or digit. Durations are rounded to 0.1 s at import.

## Admin steps

Open `/admin` and sign in with the content of `admin_token` (kept in this browser tab only).

For the real collection, follow addendum §12 steps 7-9 for the order (fresh database; on 127.0.0.1, open registration
off and the cap at 43; then the consent, the import, the footer contact and the pilot), not the numbering below.

1. Build the stimuli into `$MOS_DATA_DIR/package/` and the key into `$MOS_DATA_DIR/private/`. Press **Import package**.
   The import checks the data (ids, stimuli, attention values, duplicate tids, the items) and every audio file.
   Re-importing the same content changes nothing. Replacing a package that already has answers needs "replace": its
   answers stay in the database and remain exportable (`?package=<id>`). A builder `dry_run` package is shown to
   raters as TEST DATA.
2. Fill `consent_template_v1.html` (contact Quoc Nguyen, UIT, VNU-HCM, nvquoc@uit.edu.vn; the ethics note; the
   retention date), paste it and press **Save**, or write it to `$MOS_DATA_DIR/consent_v1.html`. A text that still
   holds a placeholder, a script, an event handler (also after "/" or a quote, as in `<svg/onload=...>`), a script URL
   (also behind character references, as in `java&#115;cript:`), a frame or a form is refused, also when written to
   the file by hand (the test then shows as not open). Inline styles are not applied. The sha256 of the text each
   rater agreed to is stored. If the text changes while a rater has it on screen, the rater is shown the new text and
   asked to agree again. Set the contact shown in the page footer.
3. **Create codes** (1-500, optional prefix such as `VC`, optional note). Codes are 12 Crockford base32 characters
   (60 bits), e.g. `VC-7K3M9QX2TA4B`. Each new code gets the next rater ID (R001, R002, ...) and a half (below). The
   codes are shown once: the server keeps only their HMAC. Copy them, send one code per rater by private message, and
   keep the list of who received which code outside the data directory and the repository.
4. **The code cap** (addendum §7.1): the server issues at most 40 codes in all by default, by both routes together
   (addendum §7.1: 40 plus 3 pilot codes, so the cap is set to 43 before the pilot). Every code ever issued counts,
   disabled and withdrawn ones included (test and pilot codes too). The cap is a setting (0-10000; each change is
   logged). When it is reached, **Create codes** is refused (409 `code_cap`) and the page offers no
   registration. The status shows the codes issued, the cap and the codes left.
5. **Open registration** is the second access route. It is on by default, also in a fresh database: the page offers
   "I have no code: start a new session", which issues a code, shows it once, and starts. It is limited to 30
   registrations per 10 minutes from all clients together and stops at the code cap. In the real collection it is off
   from before the consent is saved until invitations go out, and off again from the close (addendum §7.1, §12 steps
   7-9). Each rater's route (`invite` or `open`) is stored and exported.
6. Watch **Raters**: rater ID, route, half, first use, consent (time and page language), browser, sound check,
   answered/total per block, finished (time of the last rating), completed, last seen; and per half the codes issued,
   consented, fully rated, completed and sound checks failed (disabled raters not counted). **Disable** ends a
   rater's sign-ins and blocks the code.
7. **Withdrawal**: enter the code a rater sends, press **Find**, then **Delete**. The rater, every session, answer and
   sign-in are deleted from the database (rows overwritten by secure_delete, and the write-ahead log emptied at once),
   from the export folders (their results file and their rows of answers.csv) and from every daily copy in `backup/`
   (a copy that cannot be cleaned is deleted). The answer says how many export files and copies changed. Copies of the
   database taken off the Mac are not reached: delete the rater there by hand. The rater ID is not given out again.
8. **Export**: "Write export folder" writes one `listening_<rater ID>.json` per rater who consented, in the journal
   page's results format, plus `answers.csv` (one row per answer, opaque ids only). Optionally only completed sessions;
   a session counts as completed only after the background questions and the end screen, so untick "completed
   sessions only" for the pilot export and the final export (addendum §6.3, §12 step 9).
   Only the screening and analysis script of addendum §9.5 opens an export (addendum §12): do not run
   `benchmark/listening_analysis.py` or any other analysis on it, and do not open the key. Save the raters list and the
   status in the export folder as `raters.json` and `status.json` (addendum §12 step 9); the admin page shows them but
   has no download, so on the Mac run, with the folder the export reported:
   `(umask 077; T=$(cat ~/mos_data/admin_token); for x in raters status; do curl -s -H "Authorization: Bearer $T" http://127.0.0.1:8790/api/admin/$x > <dir>/$x.json; done)`.
   A withdrawal does not change these two files, so remove the rater's entry from `raters.json` by hand.

## Halves (addendum §6.2, §7.1)

The half is set when a code is issued or a rater registers, and stored with the rater: it is never derived from the
code or the rater ID, and a rater keeps it in every package. Each new code goes to the half with fewer consented raters
who can still complete in the current package (disabled raters and raters who failed the sound check not counted),
counting the codes of the same request as if they had consented; on a tie, to the half with fewer codes issued (not
disabled); then to A. It is decided by these counts only, never by a rating.

## Trial order (addendum §6.1)

Per rater and block, the main trials (gold and `gold_src` included; the page and the server never see the kinds) are
shuffled with mulberry32 seeded by FNV-1a of `<rater ID>:<block>`, drawing again from the same stream until no item is
on two consecutive trials (a uniform draw among such orders; a repair moves trials only if 10,000 draws fail). The
attention trials then go into the gaps: never in the first 3 positions, never next to each other. The item of a main
MOS or SMOS trial is its opaque `item` field when the builder gives one; otherwise the SMOS trial's `ref` (the clips of
one item share one reference) and, for MOS, the reference of the SMOS trial that plays the same clip. The import
refuses a main trial whose item cannot be found and an item with more than half of a block's main trials in a half.
Item ids never reach the browser.

## What raters do (addendum §6.1)

1. Open https://mos.betaschool.edu.vn with headphones in a quiet place, on a computer or a phone. The page is in
   Vietnamese, with an English switch on every screen (a trial keeps its plays and rating when the language changes).
2. Enter the access code (12 characters, maybe after a prefix and "-"), or start a new session with open registration
   and write down the code shown once. If registration is closed, the page says so.
3. Read the information and tick "Tôi đồng ý" (I agree). Until then the server stores only that the code was used.
4. Sound check (Woods et al., 2017): 6 trials, pick the quietest of three tones, pass at 5 of 6, 2 attempts. Two
   failures end the session.
5. MOS: instructions, 3 practice trials, then the MOS trials with a break screen after half of them; a break; SMOS:
   instructions, then the SMOS trials with a break screen after half of them. Each clip must play to the end before
   the rating buttons work (if the package sets `require_full_play`; the server then also refuses a rating with an
   unplayed clip). Attention trials say which rating to choose.
6. Six background questions (each with "Prefer not to say"), then the thanks.

Every answer is sent at once with the page language. If the network fails, answers wait in the browser and are sent
again. A rater can close the page and continue later on any device with the same code: the test resumes at the first
unanswered trial (the same trials, in the same order). The sign-in lasts until the browser is closed (at most 24 hours).
"Use another code" signs out and removes the page's local copy of the rater's answers, unless some still wait to be
sent (they are sent once the code is entered again). At the end the sign-in ends on every device and the page removes
its local copy, so the next person at a shared computer starts at the code screen. If the page and the server keep
disagreeing (more than 3 reloads of the server state in 30 s), the page stops and asks the rater to reload.

## What is stored (addendum §11.3)

Per rater: the HMAC of the code, the rater ID, the half, the route, the time of first and last use. From the consent on:
the time of the consent, its page language and the sha256 of the text; the browser family and major version (e.g.
"Chrome 130"); the sound-check trials and result; every answer (trial id, stimulus ids, rating, plays per clip, seconds
played, response time, time of answer, page language); the time of the last rating and of the end; the six background
answers. Not stored: name, e-mail, IP address, the user-agent string, location, the code itself. The limits keep the
client address (an IPv4 address, or the /64 of an IPv6 address) in memory only, for their 10-minute window. The server
keeps no access log; it logs errors with the method and path only (paths never hold a code), changes of the code cap
and withdrawals by rater ID.

The results file is the journal page's (version 1), keyed by the rater ID, with: `route`, `completed` (the end),
`consent` (`at`, `lang`, `sha256`), `lang` in every response; `started` is the time of the consent, and `user_agent`
holds the browser family. `answers.csv` columns: `package_id, rater_id, route, parity, half, block, pos, tid, stim, ref,
a, b, rating, attention, lang, plays, plays_ref, plays_stim, plays_a, plays_b, played_s, rt_ms, t_ms, t`.

## API (JSON, errors as `{"error": {"code", "message"}}`)

Every request whose `Host` is not one of `MOS_HOSTS` or a loopback address gets 421 `bad_host`.

Rater (session cookie `mos_sid`: 32 random bytes, HttpOnly, Secure, SameSite=Strict, no Max-Age; the sign-in expires on
the server after 24 h): `GET /api/info` (`open_registration` is false when registration is off or the cap is reached);
`GET /api/consent` → `{html, sha256}`; `POST /api/login {code}`; `POST /api/register {}` (open registration; the only
answer that holds a code; 403 `closed` when it is off or at the cap); `POST /api/logout {}`; `GET /api/state` (the
results so far; before the consent an empty skeleton with the rater's half); `POST /api/consent {lang, sha256}` (starts
the rating session; 409 `consent_changed` when the text has changed); `GET /api/plan` (the rater's trials and stimuli,
opaque ids only; after the consent); `POST /api/headphone {attempt, trial, choice}`; `POST /api/answer {tid, block,
value, lang, plays, played_s, rt_ms, t_ms}` (400 `not_played` when the package asks for full plays and a clip has 0
plays); `POST /api/background {age, first_language, english, hearing, device, previous_test}`; `POST /api/finish {}`
(records the end, ends every sign-in of the rater and clears the cookie).
Everything after the login needs the consent (409 `consent_required`). Answers are taken only for the rater's next
trial; a repeat of an answered trial returns 200 with `duplicate: true` and keeps the first answer. `rt_ms` and `t_ms`
are non-negative integers, stored capped at 24 h. Bodies are limited to 16 KB, also when sent chunked. Login and
registration answer 429 with `Retry-After` (seconds left) over the limits. Codes travel only in POST bodies.
Stimuli: `GET /s/<sid>.<ext>` (signed in, after the consent, only that rater's clips, Range requests, `private` long
cache).
Admin (`Authorization: Bearer <admin_token>`): `GET /api/admin/status` (with `codes: {issued, max, left}`),
`GET /api/admin/raters[?package=]` (with `halves`), `POST /api/admin/codes {n, prefix?, note?}` →
`{codes: [{code, rater_id, half}]}` (409 `code_cap` when `n` is over the codes left),
`POST /api/admin/raters/disable {rater_id, disabled}`, `POST /api/admin/raters/find {code}`,
`POST /api/admin/raters/delete {rater_id} | {code}` → `{rater_id, export_files, backup_files, wal_cleared}`,
`POST /api/admin/package/import {force?}`, `GET|POST /api/admin/consent {html}`,
`POST /api/admin/settings {open_registration?, contact?, max_codes?}`, `POST /api/admin/export {completed_only?, package?}`,
`GET /api/admin/export/rater?rater=R001[&package=]`, `GET /api/admin/export/answers.csv[?completed_only=1][&package=]`.

## Upgrading from the first version of this server

The first version kept its codes in plain text, with 10 characters (50 bits). Version 2 accepts only 12-character
codes and does not carry the first version's raters over: on a first-version database that holds any code the server
does not start and says why, leaving the file as it was. Stop the server, move `mos.sqlite`, `mos.sqlite-wal` and
`mos.sqlite-shm` out of the data directory (or delete them), and start again: an empty database is created, rater IDs
start at R001 and no old code counts against the cap. A first-version database without codes is converted and keeps
its packages and settings. Also: delete the export folders and database copies the first version made (they name
raters by code); save the filled template as `consent_v1.html` (`consent_vi.html` and `consent_en.html` are no longer
read); and import the package again (a package imported by the first version has no item ids).

## Deployment notes

- The tunnel points at `http://127.0.0.1:8790` and must pass the original `Host` header (the default; do not set
  `httpHostHeader`): requests for any host name other than `MOS_HOSTS` get 421, and POSTs whose `Origin` does not match
  `Host` get 403. After the tunnel is up, test a POST with an `Origin` header through it.
- Turn off Cloudflare Rocket Loader and any injected analytics for this hostname: the strict CSP blocks them.
- Put Cloudflare Access in front of `/admin*` and `/api/admin/*`; otherwise the admin token is the only protection.
- cloudflared runs at its default log level (info), which records no client addresses (checked 30 Oct 2026, addendum
  §11.3); do not raise it to debug during collection.
- Test on a separate data directory: every code issued on the live database counts against the cap.
- Keep the Mac awake during data collection. The daily copy goes to `$MOS_DATA_DIR/backup/`; copy it off the Mac if
  needed, and delete the copies with the database at the retention date.
