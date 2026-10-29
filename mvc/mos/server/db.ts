// SQLite store of the listening-test server (bun:sqlite, WAL). Migrations run in order on start; PRAGMA user_version
// holds how many have run. No IP address and no access code is stored: a rater is a rater ID (R001, ... in issue
// order, never reused) with the HMAC of the code (addendum §11.2). A rating session exists only from the consent on
// (§11.5). Deleted rows are overwritten (secure_delete), and a withdrawal also empties the write-ahead log.
import { Database } from 'bun:sqlite';
import type { AnswerRow, Block, Half, Headphone, LtData, Session, SessionRow } from './core';
import { raterIdOf } from './core';

type Migration = string | ((db: Database) => void);

export const MIGRATIONS: Migration[] = [
  `CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
   CREATE TABLE packages (id TEXT PRIMARY KEY, imported_at TEXT NOT NULL, sample INTEGER NOT NULL DEFAULT 0,
                          lt_data TEXT NOT NULL);
   CREATE TABLE raters (code TEXT PRIMARY KEY, created_at TEXT NOT NULL, origin TEXT NOT NULL, note TEXT,
                        disabled INTEGER NOT NULL DEFAULT 0, last_seen TEXT);
   CREATE TABLE sessions (code TEXT NOT NULL REFERENCES raters(code), package_id TEXT NOT NULL REFERENCES packages(id),
                          session_json TEXT NOT NULL, started TEXT NOT NULL, user_agent TEXT, consent_at TEXT,
                          consent_lang TEXT, consent_sha TEXT, headphone_json TEXT NOT NULL, background_json TEXT,
                          finished TEXT, completed_at TEXT, PRIMARY KEY (code, package_id));
   CREATE TABLE answers (code TEXT NOT NULL, package_id TEXT NOT NULL, block TEXT NOT NULL, pos INTEGER NOT NULL,
                         tid TEXT NOT NULL, rating INTEGER NOT NULL, plays_json TEXT NOT NULL, played_s REAL,
                         rt_ms INTEGER, t_ms INTEGER, t TEXT NOT NULL, PRIMARY KEY (code, package_id, tid),
                         UNIQUE (code, package_id, block, pos),
                         FOREIGN KEY (code, package_id) REFERENCES sessions(code, package_id));
   CREATE TABLE auth (token_hash TEXT PRIMARY KEY, code TEXT NOT NULL REFERENCES raters(code), created_at TEXT NOT NULL,
                      expires_at TEXT NOT NULL);
   CREATE INDEX auth_code ON auth(code);`,
  // Version 2: codes only as their HMAC, rater IDs, the stored half and route, sessions from the consent on, the
  // browser family instead of the user agent, the page language of every answer. The first version's codes have 50
  // bits (§11.2 asks for 60) and are kept in plain text, so its raters are not carried over: a first-version database
  // that holds any code is refused, and the server does not start until it is moved out of the data directory. One
  // without codes keeps its packages and settings.
  (db) => {
    const n = (db.query('SELECT COUNT(*) AS n FROM raters').get() as {n: number}).n;
    if (n) {
      throw new Error(`${db.filename} is a database of the first version of this server with ${n} access codes of ` +
                      '50 bits, which version 2 does not accept. Stop the server, move mos.sqlite, mos.sqlite-wal and ' +
                      'mos.sqlite-shm out of the data directory (or delete them), and start again with an empty database');
    }
    db.exec(`DROP INDEX auth_code; DROP TABLE auth; DROP TABLE answers; DROP TABLE sessions; DROP TABLE raters;
      CREATE TABLE raters (rater_id TEXT PRIMARY KEY, code_hmac TEXT NOT NULL UNIQUE,
                           route TEXT NOT NULL CHECK (route IN ('invite', 'open')),
                           half TEXT NOT NULL CHECK (half IN ('A', 'B')), note TEXT, created_at TEXT NOT NULL,
                           disabled INTEGER NOT NULL DEFAULT 0, first_used TEXT, last_seen TEXT);
      CREATE TABLE sessions (rater_id TEXT NOT NULL REFERENCES raters(rater_id) ON DELETE CASCADE,
                             package_id TEXT NOT NULL REFERENCES packages(id), session_json TEXT NOT NULL, browser TEXT,
                             consent_at TEXT NOT NULL, consent_lang TEXT NOT NULL, consent_sha TEXT NOT NULL,
                             headphone_json TEXT NOT NULL, background_json TEXT, finished TEXT, completed_at TEXT,
                             PRIMARY KEY (rater_id, package_id));
      CREATE TABLE answers (rater_id TEXT NOT NULL, package_id TEXT NOT NULL, block TEXT NOT NULL, pos INTEGER NOT NULL,
                            tid TEXT NOT NULL, rating INTEGER NOT NULL, lang TEXT, plays_json TEXT NOT NULL,
                            played_s REAL, rt_ms INTEGER, t_ms INTEGER, t TEXT NOT NULL,
                            PRIMARY KEY (rater_id, package_id, tid), UNIQUE (rater_id, package_id, block, pos),
                            FOREIGN KEY (rater_id, package_id) REFERENCES sessions(rater_id, package_id)
                              ON DELETE CASCADE);
      CREATE TABLE auth (token_hash TEXT PRIMARY KEY, rater_id TEXT NOT NULL REFERENCES raters(rater_id) ON DELETE CASCADE,
                         created_at TEXT NOT NULL, expires_at TEXT NOT NULL);
      CREATE INDEX auth_rater ON auth(rater_id);`);
  },
];

export const SCHEMA_VERSION = MIGRATIONS.length;

export type SessionMeta = SessionRow & {half: Half};
export type RaterRow = {rater_id: string; code_hmac: string; route: string; half: Half; note: string | null;
                        created_at: string; disabled: number; first_used: string | null; last_seen: string | null};

export class Store {
  db: Database;

  constructor(path: string) {
    this.db = new Database(path, {create: true, strict: true});
    this.db.exec('PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; PRAGMA busy_timeout = 5000; ' +
                 'PRAGMA secure_delete = ON;');
    try {
      this.migrate();
    } catch (e) {
      this.db.close();
      throw e;
    }
  }

  migrate() {
    const v = (this.db.query('PRAGMA user_version').get() as {user_version: number}).user_version;
    for (let i = v; i < MIGRATIONS.length; i++) {
      const m = MIGRATIONS[i];
      this.db.exec('PRAGMA foreign_keys = OFF');   // tables are rebuilt; checked again below
      try {
        this.db.transaction(() => {
          if (typeof m === 'string') this.db.exec(m);
          else m(this.db);
          const bad = this.db.query('PRAGMA foreign_key_check').all();
          if (bad.length) throw new Error(`migration ${i + 1}: ${bad.length} foreign key problems`);
          this.db.exec(`PRAGMA user_version = ${i + 1}`);
        })();
      } finally {
        this.db.exec('PRAGMA foreign_keys = ON');
      }
    }
  }

  close() {
    this.db.close();
  }

  // Copies the write-ahead log into the database and truncates it to 0 bytes, so that deleted rows (overwritten in
  // the database by secure_delete) leave no copy in the log. False when a reader kept part of the log.
  checkpoint(): boolean {
    const r = this.db.query('PRAGMA wal_checkpoint(TRUNCATE)').get() as {busy: number; log: number};
    return r.busy === 0 && r.log === 0;
  }

  // A copy of the database (VACUUM INTO), for the daily backup.
  backupTo(path: string) {
    this.db.query('VACUUM INTO ?').run(path);
  }

  // ── settings ───────────────────────────────────────────────
  get(key: string): string | null {
    const r = this.db.query('SELECT value FROM meta WHERE key = ?').get(key) as {value: string} | null;
    return r ? r.value : null;
  }

  set(key: string, value: string | null) {
    if (value === null) this.db.query('DELETE FROM meta WHERE key = ?').run(key);
    else this.db.query('INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value')
      .run(key, value);
  }

  // ── packages ───────────────────────────────────────────────
  addPackage(id: string, data: LtData, sample: boolean, now: string) {
    this.db.query('INSERT OR IGNORE INTO packages (id, imported_at, sample, lt_data) VALUES (?, ?, ?, ?)')
      .run(id, now, sample ? 1 : 0, JSON.stringify(data));
  }

  getPackage(id: string): {id: string; imported_at: string; sample: boolean; data: LtData} | null {
    const r = this.db.query('SELECT * FROM packages WHERE id = ?').get(id) as any;
    return r ? {id: r.id, imported_at: r.imported_at, sample: !!r.sample, data: JSON.parse(r.lt_data)} : null;
  }

  answerCount(packageId: string): number {
    return (this.db.query('SELECT COUNT(*) AS n FROM answers WHERE package_id = ?').get(packageId) as {n: number}).n;
  }

  // ── raters and login sessions ──────────────────────────────
  // A new rater with the next rater ID; false when the HMAC is taken (the caller draws another code).
  addRater(codeHmac: string, route: string, half: Half, note: string | null, now: string): string | null {
    return this.db.transaction(() => {
      if (this.db.query('SELECT 1 FROM raters WHERE code_hmac = ?').get(codeHmac)) return null;
      const seq = Number(this.get('rater_seq') ?? 0) + 1;
      const id = raterIdOf(seq);
      this.db.query(`INSERT INTO raters (rater_id, code_hmac, route, half, note, created_at)
                     VALUES (?, ?, ?, ?, ?, ?)`).run(id, codeHmac, route, half, note, now);
      this.set('rater_seq', String(seq));
      return id;
    })();
  }

  rater(raterId: string): RaterRow | null {
    return this.db.query('SELECT * FROM raters WHERE rater_id = ?').get(raterId) as RaterRow | null;
  }

  raterByHmac(codeHmac: string): RaterRow | null {
    return this.db.query('SELECT * FROM raters WHERE code_hmac = ?').get(codeHmac) as RaterRow | null;
  }

  raters(): RaterRow[] {
    return this.db.query('SELECT * FROM raters ORDER BY rowid').all() as RaterRow[];
  }

  setDisabled(raterId: string, disabled: boolean): boolean {
    const n = this.db.query('UPDATE raters SET disabled = ? WHERE rater_id = ?').run(disabled ? 1 : 0, raterId).changes;
    if (disabled) this.dropAuthOf(raterId);
    return n === 1;
  }

  // Withdrawal: the rater, every session, answer and sign-in (cascade), overwritten in the database and gone from the
  // write-ahead log. The rater ID is not given out again. walCleared: false when the log could not be truncated.
  deleteRater(raterId: string): {deleted: boolean; walCleared: boolean} {
    const deleted = this.db.query('DELETE FROM raters WHERE rater_id = ?').run(raterId).changes === 1;
    return {deleted, walCleared: this.checkpoint()};
  }

  // How many codes were ever issued, by both routes (withdrawn and disabled ones included).
  codesIssued(): number {
    return Number(this.get('rater_seq') ?? 0);
  }

  // For the half of the next code: per half, the codes issued that are not disabled, and the sound-check state of
  // every consented rater who is not disabled in the package.
  halfCounts(packageId: string | null) {
    const issued: Record<Half, number> = {A: 0, B: 0};
    for (const r of this.db.query('SELECT half, COUNT(*) AS n FROM raters WHERE disabled = 0 GROUP BY half')
      .all() as {half: Half; n: number}[]) issued[r.half] = r.n;
    const sessions = packageId === null ? [] : (this.db.query(`SELECT r.half, s.headphone_json FROM sessions s
        JOIN raters r ON r.rater_id = s.rater_id WHERE s.package_id = ? AND r.disabled = 0`)
      .all(packageId) as {half: Half; headphone_json: string}[])
      .map((r) => ({half: r.half, headphone: JSON.parse(r.headphone_json) as Headphone}));
    return {issued, sessions};
  }

  // That the code was used: the only thing stored before the consent.
  touch(raterId: string, now: string) {
    this.db.query('UPDATE raters SET last_seen = ?, first_used = COALESCE(first_used, ?) WHERE rater_id = ?')
      .run(now, now, raterId);
  }

  addAuth(tokenHash: string, raterId: string, now: string, expires: string) {
    this.db.query('DELETE FROM auth WHERE expires_at < ?').run(now);
    this.db.query('INSERT INTO auth (token_hash, rater_id, created_at, expires_at) VALUES (?, ?, ?, ?)')
      .run(tokenHash, raterId, now, expires);
  }

  authRater(tokenHash: string, now: string): string | null {
    const r = this.db.query(`SELECT a.rater_id FROM auth a JOIN raters r ON r.rater_id = a.rater_id
                             WHERE a.token_hash = ? AND a.expires_at > ? AND r.disabled = 0`).get(tokenHash, now) as any;
    return r ? r.rater_id : null;
  }

  dropAuth(tokenHash: string) {
    this.db.query('DELETE FROM auth WHERE token_hash = ?').run(tokenHash);
  }

  dropAuthOf(raterId: string) {
    this.db.query('DELETE FROM auth WHERE rater_id = ?').run(raterId);
  }

  // ── rating sessions (from the consent on) ──────────────────
  addSession(raterId: string, packageId: string, s: Session, browser: string | null, lang: string, sha: string,
             now: string) {
    this.db.query(`INSERT OR IGNORE INTO sessions (rater_id, package_id, session_json, browser, consent_at, consent_lang,
                                                   consent_sha, headphone_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
      .run(raterId, packageId, JSON.stringify(s), browser, now, lang, sha, JSON.stringify({attempts: [], passed: false}));
  }

  session(raterId: string, packageId: string): SessionMeta | null {
    const r = this.db.query(`SELECT s.*, r.route, r.half FROM sessions s JOIN raters r ON r.rater_id = s.rater_id
                             WHERE s.rater_id = ? AND s.package_id = ?`).get(raterId, packageId) as any;
    return r ? this.sessionFrom(r) : null;
  }

  sessions(packageId: string): SessionMeta[] {
    return (this.db.query(`SELECT s.*, r.route, r.half FROM sessions s JOIN raters r ON r.rater_id = s.rater_id
                           WHERE s.package_id = ? ORDER BY r.rowid`).all(packageId) as any[])
      .map((r) => this.sessionFrom(r));
  }

  private sessionFrom(r: any): SessionMeta {
    return {rater_id: r.rater_id, route: r.route, half: r.half, package_id: r.package_id,
            session: JSON.parse(r.session_json), browser: r.browser, consent_at: r.consent_at,
            consent_lang: r.consent_lang, consent_sha: r.consent_sha,
            headphone: JSON.parse(r.headphone_json) as Headphone,
            background: r.background_json ? JSON.parse(r.background_json) : null, finished: r.finished,
            completed_at: r.completed_at};
  }

  setHeadphone(raterId: string, packageId: string, hp: Headphone) {
    this.db.query('UPDATE sessions SET headphone_json = ? WHERE rater_id = ? AND package_id = ?')
      .run(JSON.stringify(hp), raterId, packageId);
  }

  setBackground(raterId: string, packageId: string, bg: Record<string, string>) {
    this.db.query(`UPDATE sessions SET background_json = ? WHERE rater_id = ? AND package_id = ?
                   AND background_json IS NULL`).run(JSON.stringify(bg), raterId, packageId);
  }

  setCompleted(raterId: string, packageId: string, now: string) {
    this.db.query('UPDATE sessions SET completed_at = ? WHERE rater_id = ? AND package_id = ? AND completed_at IS NULL')
      .run(now, raterId, packageId);
  }

  // Stores the answer and, when it is the last trial, the session's finished time, in one transaction.
  addAnswer(raterId: string, packageId: string, a: AnswerRow, last: boolean): boolean {
    return this.db.transaction(() => {
      const n = this.db.query(`INSERT OR IGNORE INTO answers (rater_id, package_id, block, pos, tid, rating, lang,
                                 plays_json, played_s, rt_ms, t_ms, t) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
        .run(raterId, packageId, a.block, a.pos, a.tid, a.rating, a.lang, JSON.stringify(a.plays), a.played_s, a.rt_ms,
             a.t_ms, a.t).changes;
      if (n === 1 && last) {
        this.db.query('UPDATE sessions SET finished = ? WHERE rater_id = ? AND package_id = ? AND finished IS NULL')
          .run(a.t, raterId, packageId);
      }
      return n === 1;
    })();
  }

  private answerFrom(r: any): AnswerRow {
    return {block: r.block as Block, pos: r.pos, tid: r.tid, rating: r.rating, lang: r.lang,
            plays: JSON.parse(r.plays_json), played_s: r.played_s, rt_ms: r.rt_ms, t_ms: r.t_ms, t: r.t};
  }

  answers(raterId: string, packageId: string): AnswerRow[] {
    return (this.db.query('SELECT * FROM answers WHERE rater_id = ? AND package_id = ?').all(raterId, packageId) as any[])
      .map((r) => this.answerFrom(r));
  }

  answer(raterId: string, packageId: string, tid: string): AnswerRow | null {
    const r = this.db.query('SELECT * FROM answers WHERE rater_id = ? AND package_id = ? AND tid = ?')
      .get(raterId, packageId, tid) as any;
    return r ? this.answerFrom(r) : null;
  }

  answerCounts(raterId: string, packageId: string): Record<string, number> {
    const rows = this.db.query(`SELECT block, COUNT(*) AS n FROM answers WHERE rater_id = ? AND package_id = ?
                                GROUP BY block`).all(raterId, packageId) as {block: string; n: number}[];
    return Object.fromEntries(rows.map((r) => [r.block, r.n]));
  }
}

// Removes a withdrawn rater from a copy of the database (a daily backup): the rater, sessions, answers and sign-ins,
// then VACUUM, so that the file keeps no trace of them. The rollback journal stays in memory. Returns whether the
// rater was in the copy.
export function scrubCopy(path: string, raterId: string): boolean {
  const db = new Database(path, {strict: true});
  try {
    db.exec('PRAGMA journal_mode = MEMORY; PRAGMA temp_store = MEMORY; PRAGMA foreign_keys = ON; ' +
            'PRAGMA secure_delete = ON;');
    const n = db.query('DELETE FROM raters WHERE rater_id = ?').run(raterId).changes;
    if (n) db.exec('VACUUM');
    return n > 0;
  } finally {
    db.close();
  }
}
