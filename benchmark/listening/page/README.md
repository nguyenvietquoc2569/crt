# Listening-test page

Self-contained web page for the listening test of `benchmark/ANALYSIS_PLAN_v1.md` §5.4: MOS (naturalness, 5-point
ACR, ITU-T P.808), SMOS (similarity to a target reference, 5-point) and CMOS (CCR −3..+3, ITU-T P.800), after a
headphone check. Plain HTML/CSS/JS, no network access and no libraries; it runs from a local folder (`file://`)
or a static web host. The page only collects answers. Screening (gold items, attention items, hidden identical
pairs, headphone check) is done later by the analysis script.

| File | Role |
|---|---|
| `index.html` | page shell; loads `style.css`, `lt_data.js`, `lt_core.js`, `lt_app.js` (classic scripts, no modules) |
| `lt_core.js` | pure logic: seeded randomisation, session layout, results skeleton, headphone-check tones |
| `lt_app.js` | screens, audio, autosave/resume, results download and submit |
| `style.css` | layout and theme |
| `sample_lt_data.js` | fake data for a visual check (`index.html?sample`) |
| `lt_core.test.js`, `package.json` | Node tests; `package.json` only lets `node --test <this folder>` find them |

## Assembling the package for raters

`benchmark/listening_stimuli.py build --plan <plan.json> --out <dir> --key <key.json>` writes `lt_data.js`
(defining `window.LT_DATA`) and the audio files under `stimuli/` into `<dir>`, and the key (what every trial and
clip is) to `<key.json>`, which it refuses to put inside `<dir>`: the key reveals the conditions, so it never
reaches raters (the organiser keeps it for `listening_analysis.py`). Copy the four page files into `<dir>`. The rater
package is then one folder:

```
listening_test/
  index.html  style.css  lt_core.js  lt_app.js     <- from this folder
  lt_data.js                                       <- from listening_stimuli.py
  stimuli/<sid>.flac ...                           <- from listening_stimuli.py
```

`sample_lt_data.js`, the tests, `package.json` and this README are not needed (they do no harm if included).
Zip the folder and send it, or upload it unchanged to a static host. Before sending it, open `index.html` from the
folder and go through the consent, rater-ID and sound-check screens and the first trial. A missing or invalid
`lt_data.js` (unknown stimulus ids, duplicate tids) stops the page with a message.

`LT_DATA` (version 1): `config` {`title`, `consent_html`, `contact`, `halves`, `require_full_play`,
`headphone_check` {`trials`, `pass`, `attempts`}, `submit_url`}, `stimuli` {sid: {`src`, `dur`}}, `practice`
[{`tid`, `stim`}], `mos` [{`tid`, `stim`, `half`?}], `smos` [{`tid`, `stim`, `ref`, `half`?}], `cmos` [{`tid`,
`x`, `y`, `group`?}]. Any trial with `attention` (the required rating) and `instruction` is an attention item.
`group` is an opaque id per comparison (CRT-AL vs FAISS, CRT-AL vs CRT τ = 0.4, the hidden identical pairs).

## What raters do

1. Unzip the folder (or open the link) and open `index.html` in a current Chrome, Firefox or Safari on a
   computer. Use headphones in a quiet room.
2. Read the information and tick "I agree", then enter the rater ID given by the organiser (2–20 letters,
   digits, `-` or `_`; for example `R07`).
3. Sound check: set a comfortable volume with the test sound, then pick the quietest of three tones in 6 trials.
   Raters who fail twice are thanked and stopped; they still download and send their file.
4. Practice (3 recordings), then the MOS, SMOS and CMOS parts, with a break screen between parts. Each clip must
   be played to the end before the rating buttons work (`require_full_play`); replays are allowed.
5. At the end, press **Download results** and send the file `listening_<raterId>_<YYYYMMDD-HHMM>.json` to the
   contact. If `config.submit_url` is set, **Submit** also POSTs the same JSON there.

Answers are saved in the browser after every response (localStorage key `lt:<raterId>`). A rater who closes the
page reopens the same `index.html` in the same browser and enters the same ID, and **Resume** continues at the
first unanswered trial (that trial starts again). If the saved session belongs to different test data, the old
one is kept under `lt-backup:<raterId>:<time>` and a new session starts.

## Randomisation (deterministic per rater ID)

- Parity: the trailing digits of the ID mod 2 (`R07` → 1, `R10` → 0); IDs without trailing digits use the
  32-bit FNV-1a hash of the ID (UTF-8 bytes) mod 2.
- `halves: true` gives parity 0 half `A` and parity 1 half `B` of the MOS/SMOS items that carry a `half`. Items
  without `half` and all CMOS pairs go to every rater.
- Each block's main trials are shuffled with mulberry32 seeded by FNV-1a(`<id>:<block>`). Attention trials
  are inserted with the seed `<id>:<block>:attention`, never in the first 3 positions and never next to each
  other (unless the block is too short for that). Practice keeps its order.
- CMOS: within each `group`, the k-th non-attention pair shows x as A when (k + parity) is even. So each rater gets
  x first in exactly half the pairs of every comparison (16 of the 32 CRT-AL vs FAISS pairs, 16 of 32 vs CRT
  τ = 0.4, 4 of the 8 hidden pairs), and the two parities are complementary. Pairs without `group` share one
  counter. Attention pairs always show x as A.
- Sound check: `headphoneTrials(6, rng('<id>:headphone:<attempt>'))`. The tones follow Woods et al. (2017): 200 Hz,
  1 s, 100 ms ramps; one tone is 6 dB quieter and one has its right channel inverted (antiphase). This page adds a
  0.3 s lead-in and 0.5 s gaps. Each trial plays once. Pass at ≥ 5 of 6 correct, at most 2 attempts.

## Results file

`{version, rater_id, parity, half, started, finished, user_agent, config_title, headphone {attempts [{trials
[{order, answer, choice}], correct}], passed}, responses {practice, mos, smos, cmos}, block_ms}`.

- `practice`/`mos`: `{tid, stim, rating, plays, rt_ms, t, attention?}`; `smos`: `{tid, stim, ref, rating,
  plays_ref, plays_stim, rt_ms, t, attention?}`; `cmos`: `{tid, a, b, rating, plays_a, plays_b, rt_ms, t,
  attention?}`. The CMOS `rating` is the quality of B relative to A, so the preference for y is `rating` when
  `a` is x and `−rating` when `a` is y.
- `plays*` counts started plays (stopped ones included). `rt_ms` runs from the trial screen being shown to the
  last rating click, and `t` is the time Next was pressed. `attention` is the rating the item asked for.
- `block_ms` is the time spent on the block's trial screens (breaks and instruction screens excluded; 0 for a
  block not reached). `finished` stays `null` unless the CMOS block was completed.
- The answers are in presentation order.

## Submit endpoint (optional)

`submit_url` receives `POST` with `Content-Type: application/json`. A page opened from `file://` sends
`Origin: null`, so the server must answer the CORS preflight and the POST with a matching
`Access-Control-Allow-Origin`. Download always remains available.

## Tests

```
node --test benchmark/listening/page/
```

They cover the seeded functions, the session layout, the results schema, the headphone-check signal, the sample
data and the script order in `index.html`. The UI was checked in a Chromium browser with the sample data and with
real FLAC files.
