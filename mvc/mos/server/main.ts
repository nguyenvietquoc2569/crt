// Entry point: bun server/main.ts (from mvc/mos). Environment: MOS_DATA_DIR (default ~/mos_data), MOS_PORT (default
// 8790), MOS_HOSTS (host names served besides the loopback addresses, comma-separated, default
// mos.betaschool.edu.vn), MOS_SAMPLE=1 (journal sample data while no package is imported), MOS_COOKIE_SECURE=0
// (plain-http testing only), MOS_TRUST_PROXY=0 (do not take the client IP from CF-Connecting-IP), MOS_LOGIN_LIMIT
// (failed logins per IP or IPv6 /64 and 10 minutes, default 100; all logins and registrations of one are capped at 10
// times that), MOS_REGISTER_LIMIT (open registrations per 10 minutes from all clients together, default 30).
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';
import { startServer } from './app';

const env = process.env;
process.umask(0o077);   // the database, the token, the secret, the consent text, exports and backups: owner only
const dataDir = resolve(env.MOS_DATA_DIR || join(homedir(), 'mos_data'));
const port = Number(env.MOS_PORT || 8790);
if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error(`MOS_PORT ${env.MOS_PORT} is not a port`);

const {server, stop} = startServer({
  dataDir, port, hostname: '127.0.0.1',
  sample: env.MOS_SAMPLE === '1',
  cookieSecure: env.MOS_COOKIE_SECURE !== '0',
  trustProxy: env.MOS_TRUST_PROXY !== '0',
  loginLimit: env.MOS_LOGIN_LIMIT ? Number(env.MOS_LOGIN_LIMIT) : undefined,
  registerLimit: env.MOS_REGISTER_LIMIT ? Number(env.MOS_REGISTER_LIMIT) : undefined,
  hosts: (env.MOS_HOSTS ?? 'mos.betaschool.edu.vn').split(','),
});
console.log(`listening test on http://127.0.0.1:${server.port} (data ${dataDir}${env.MOS_SAMPLE === '1' ? ', sample mode' : ''})`);

let stopping = false;
for (const sig of ['SIGINT', 'SIGTERM'] as const) {
  process.on(sig, async () => {
    if (stopping) return;
    stopping = true;
    console.log(`${sig}: stopping`);
    await stop();
    process.exit(0);
  });
}
