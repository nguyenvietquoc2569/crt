// A small browser for the page tests: the rater page (public/i18n.js, lt_core.js, lt_app.js) runs in a vm context
// with a minimal DOM, localStorage, a cookie jar and fetch against a running test server, and with Audio and
// AudioContext that "play" at once. A Browser (storage and cookies) outlives a page, like a browser profile across
// reloads. Timers of a second or more (the page's retry) run 100 times faster. No dependencies.
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { createContext, runInContext } from 'node:vm';
import { PUBLIC_DIR } from './core';

export const EN = require('../public/i18n.js').STRINGS.en as Record<string, string>;
export const VI = require('../public/i18n.js').STRINGS.vi as Record<string, string>;

export class FakeNode {
  parent: FakeElement | null = null;
  get textContent(): string { return ''; }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
}

class FakeText extends FakeNode {
  constructor(public data: string) { super(); }
  get textContent() { return this.data; }
}

const REFLECTED = ['hidden', 'disabled', 'checked'];

export class FakeElement extends FakeNode {
  children: FakeNode[] = [];
  attrs: Record<string, string> = {};
  listeners: Record<string, ((e: unknown) => void)[]> = {};
  className = '';
  hidden = false;
  disabled = false;
  checked = false;
  value = '';
  type = '';
  name = '';
  dataset: Record<string, unknown> = {};
  style: Record<string, string> = {};
  private html: string | null = null;
  classList = {
    contains: (c: string) => this.className.split(/\s+/).includes(c),
    add: (c: string) => this.classList.toggle(c, true),
    remove: (c: string) => this.classList.toggle(c, false),
    toggle: (c: string, on?: boolean) => {
      const list = this.className.split(/\s+/).filter((x) => x && x !== c);
      const want = on ?? !this.classList.contains(c);
      this.className = (want ? [...list, c] : list).join(' ');
      return want;
    },
  };

  constructor(public tagName: string) { super(); }

  setAttribute(k: string, v: unknown) {
    this.attrs[k] = String(v);
    if (REFLECTED.includes(k)) (this as any)[k] = true;
    else if (k === 'type' || k === 'value' || k === 'name') (this as any)[k] = String(v);
  }

  getAttribute(k: string) { return k in this.attrs ? this.attrs[k] : null; }

  removeAttribute(k: string) {
    delete this.attrs[k];
    if (REFLECTED.includes(k)) (this as any)[k] = false;
  }

  set innerHTML(v: string) {
    this.html = String(v);
    this.children = [];
  }

  get innerHTML() { return this.html ?? ''; }

  get textContent(): string {
    if (this.html !== null) return this.html.replace(/<[^>]*>/g, '');
    return this.children.map((c) => c.textContent).join('');
  }

  // The text a rater sees: without hidden elements.
  get shownText(): string {
    if (this.hidden) return '';
    if (this.html !== null) return this.html.replace(/<[^>]*>/g, '');
    return this.children.map((c) => (c instanceof FakeElement ? c.shownText : c.textContent)).join('');
  }

  set textContent(v: string) {
    this.html = null;
    this.children = [];
    this.append(String(v));
  }

  append(...nodes: unknown[]) {
    for (const n of nodes) {
      const node = n instanceof FakeNode ? n : new FakeText(String(n));
      node.remove();
      node.parent = this;
      this.children.push(node);
    }
  }

  replaceChildren(...nodes: unknown[]) {
    for (const c of this.children) c.parent = null;
    this.children = [];
    this.html = null;
    this.append(...nodes);
  }

  addEventListener(type: string, fn: (e: unknown) => void) {
    (this.listeners[type] ??= []).push(fn);
  }

  dispatch(type: string, extra: Record<string, unknown> = {}) {
    for (const fn of this.listeners[type] ?? []) fn({type, target: this, preventDefault() {}, ...extra});
  }

  focus() {}

  // As in a browser: a disabled control ignores clicks; a checkbox or radio changes, then fires "change".
  click() {
    if (this.disabled) return;
    const box = this.tagName === 'INPUT' && (this.type === 'checkbox' || this.type === 'radio');
    if (box) this.checked = this.type === 'radio' ? true : !this.checked;
    this.dispatch('click');
    if (box) this.dispatch('change');
  }

  all(): FakeElement[] {
    const out: FakeElement[] = [];
    for (const c of this.children) {
      if (c instanceof FakeElement) out.push(c, ...c.all());
    }
    return out;
  }
}

class FakeStorage {
  map = new Map<string, string>();
  getItem(k: string) { return this.map.has(k) ? this.map.get(k)! : null; }
  setItem(k: string, v: unknown) { this.map.set(k, String(v)); }
  removeItem(k: string) { this.map.delete(k); }
  key(i: number) { return [...this.map.keys()][i] ?? null; }
  get length() { return this.map.size; }
}

// A clip "plays" for a millisecond, then ends.
class FakeAudio {
  src = '';
  preload = '';
  paused = true;
  currentTime = 0;
  error = null;
  onended: (() => void) | null = null;
  onpause: (() => void) | null = null;
  onerror: (() => void) | null = null;

  play() {
    this.paused = false;
    setTimeout(() => {
      if (this.paused) return;
      this.paused = true;
      this.onended?.();
    }, 1);
    return Promise.resolve();
  }

  pause() {
    if (this.paused) return;
    this.paused = true;
    this.onpause?.();
  }

  load() {}
  removeAttribute(k: string) { if (k === 'src') this.src = ''; }
}

class FakeAudioContext {
  state = 'running';
  sampleRate = 8000;
  destination = {};
  resume() { return Promise.resolve(); }

  createBuffer(channels: number, length: number) {
    const data = Array.from({length: channels}, () => new Float32Array(length));
    return {getChannelData: (i: number) => data[i]};
  }

  createBufferSource() {
    const src: any = {buffer: null, onended: null, connect() {}, stop() {},
                      start() { setTimeout(() => src.onended?.(), 1); }};
    return src;
  }
}

export class Browser {
  storage = new FakeStorage();
  cookies = new Map<string, string>();

  constructor(lang = 'en') {
    this.storage.setItem('mos:lang', lang);
  }

  cookieHeader() {
    return [...this.cookies].map(([k, v]) => `${k}=${v}`).join('; ');
  }
}

// A fake answer, null (the server answers), or 'lose': the server gets the request, the page a network error.
type Intercept = (method: string, path: string, body: unknown) => Response | 'lose' | null;

export class Page {
  app = new FakeElement('MAIN');
  requests: string[] = [];
  clock = 0;               // ms added to performance.now()
  intercept: Intercept | null = null;
  maxRequests = 500;       // a looping page fails the test instead of hanging it

  constructor(public base: string, public browser: Browser) {}

  open() {
    const body = new FakeElement('BODY');
    body.append(this.app);
    const document = {
      createElement: (tag: string) => new FakeElement(tag.toUpperCase()),
      getElementById: (id: string) => (id === 'app' ? this.app : null),
      body, documentElement: {lang: ''}, title: '', addEventListener() {},
    };
    const ctx: any = {
      document, localStorage: this.browser.storage, Node: FakeNode, Audio: FakeAudio, AudioContext: FakeAudioContext,
      performance: {now: () => performance.now() + this.clock}, fetch: (p: string, o: any) => this.fetch(p, o),
      setTimeout: (fn: () => void, ms = 0) => setTimeout(fn, ms >= 1000 ? ms / 100 : ms), clearTimeout,
      scrollTo() {}, URL, Blob, TextEncoder, TextDecoder, console,
    };
    ctx.window = ctx;
    createContext(ctx);
    for (const f of ['i18n.js', 'lt_core.js', 'lt_app.js']) {
      runInContext(readFileSync(join(PUBLIC_DIR, f), 'utf8'), ctx, {filename: f});
    }
    return this;
  }

  private async fetch(path: string, opt: any = {}) {
    const method = opt.method || 'GET';
    this.requests.push(`${method} ${path}`);
    if (this.requests.length > this.maxRequests) throw new TypeError('the page sent too many requests');
    const fake = this.intercept?.(method, path, opt.body);
    if (fake instanceof Response) return fake;
    const headers: Record<string, string> = {...(opt.headers || {})};
    if (this.browser.cookies.size) headers.Cookie = this.browser.cookieHeader();
    if (method === 'POST') headers.Origin = this.base;
    const res = await fetch(this.base + path, {method, headers, body: opt.body});
    if (fake === 'lose') throw new TypeError('the answer was lost');
    const set = res.headers.get('set-cookie');
    if (set) {
      const pair = set.split(';')[0];
      const i = pair.indexOf('=');
      const [k, v] = [pair.slice(0, i).trim(), pair.slice(i + 1).trim()];
      if (!v || /;\s*max-age=0\b/i.test(set)) this.browser.cookies.delete(k);
      else this.browser.cookies.set(k, v);
    }
    return res;
  }

  text() { return this.app.shownText; }
  all(tag?: string) { return this.app.all().filter((e) => !tag || e.tagName === tag); }
  buttons() { return this.all('BUTTON'); }
  button(label: string) { return this.buttons().find((b) => b.textContent.trim() === label) ?? null; }
  count(prefix: string) { return this.requests.filter((r) => r.startsWith(prefix)).length; }

  async waitFor<T>(cond: () => T, what: string, ms = 4000): Promise<NonNullable<T>> {
    const end = Date.now() + ms;
    for (;;) {
      const v = cond();
      if (v) return v as NonNullable<T>;
      if (Date.now() > end) throw new Error(`timed out waiting for ${what}; the page shows: ${this.text().slice(0, 600)}`);
      await new Promise((r) => setTimeout(r, 2));
    }
  }

  async click(label: string) {
    const b = await this.waitFor(() => {
      const x = this.button(label);
      return x && !x.disabled ? x : null;
    }, `an enabled button "${label}"`);
    b.click();
  }

  // The main button of the screen (Next, Start, Continue), whatever the language.
  async clickPrimary() {
    const b = await this.waitFor(() => this.buttons().find((x) => x.classList.contains('primary') && !x.disabled),
                                 'an enabled main button');
    b.click();
  }

  // The language switch of the header.
  switchLanguage() {
    const b = this.buttons().find((x) => x.textContent === EN.langName || x.textContent === VI.langName);
    if (!b) throw new Error('no language switch on this screen');
    b.click();
  }
}

// ── the steps of a rater ─────────────────────────────────────

export async function agree(page: Page) {
  const box = await page.waitFor(() => page.all('INPUT').find((e) => e.type === 'checkbox'), 'the consent checkbox');
  box.click();
  await page.click(EN.continue);
}

export async function enterCode(page: Page, code: string) {
  const input = await page.waitFor(() => page.all('INPUT').find((e) => e.type === 'text'), 'the code field');
  input.value = code;
  input.dispatch('input');
  await page.click(EN.continue);
}

// One attempt of the sound check; answers(i) is the option chosen in trial i; during(i) runs once the tones of trial i
// have played.
export async function soundCheck(page: Page, answers: (i: number) => number, during?: (i: number) => void) {
  await page.waitFor(() => page.button(EN.hpStart) || page.button(EN.tryAgain), 'the sound-check start');
  (page.button(EN.hpStart) || page.button(EN.tryAgain))!.click();
  for (let i = 0; i < 6; i++) {
    const progress = EN.hpProgress.replace('{i}', String(i + 1)).replace('{n}', '6');
    await page.waitFor(() => page.text().includes(progress), progress);
    const play = await page.waitFor(() => page.buttons().find((b) => b.classList.contains('play') && !b.disabled),
                                    'the play button');
    play.click();
    const options = await page.waitFor(() => {
      const o = page.buttons().filter((b) => b.classList.contains('rate'));
      return o.length === 3 && o.every((b) => !b.disabled) ? o : null;
    }, 'the three tone options');
    during?.(i);
    page.buttons().filter((b) => b.classList.contains('rate'))[answers(i)].click();
    await page.click(EN.next);
  }
}

// Rates n trials, in either language, through the block introductions and the break screens on the way; before(k)
// runs before the rating of trial k, once its clips have played. Returns the trials k before which a break was shown.
export async function rate(page: Page, n: number, value = 4, before?: (k: number) => void) {
  const breaks: number[] = [];
  const clips = () => page.buttons().filter((b) => b.classList.contains('play'));
  for (let k = 0; k < n; k++) {
    for (;;) {
      await page.waitFor(() => clips().length || page.buttons().some((b) => b.classList.contains('primary')), `trial ${k}`);
      if (clips().length) break;
      if (page.text().includes(EN.midBreakText) || page.text().includes(VI.midBreakText)) breaks.push(k);
      await page.clickPrimary();
    }
    for (const c of clips()) {
      c.click();
      await page.waitFor(() => !c.classList.contains('playing'), 'the clip to end');
    }
    const choice = () => page.buttons().find((b) => b.classList.contains('rate') && !b.disabled &&
                                                   Number(b.dataset.v) === value);
    await page.waitFor(choice, 'the rating buttons');
    before?.(k);
    (await page.waitFor(choice, 'the rating buttons')).click();
    await page.clickPrimary();
  }
  return breaks;
}

// Answers every background question with its first option, in either language; returns how many there were.
export async function background(page: Page) {
  await page.waitFor(() => page.all('INPUT').some((e) => e.type === 'radio'), 'the background questions');
  const groups = new Map<string, FakeElement>();
  for (const r of page.all('INPUT').filter((e) => e.type === 'radio')) if (!groups.has(r.name)) groups.set(r.name, r);
  for (const r of groups.values()) r.click();
  await page.clickPrimary();
  return groups.size;
}
