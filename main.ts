// xStocks movers forecast: 5 session days of 9 tokenized Nasdaq names, each day's order by return and each name's
// return in 42 bins. Node 22, standard library only, deterministic (seeded draws, fixed iteration order).
//
// Model (each piece backtested against the house baseline on two years of daily history, see README.md):
//   - vol: a prior daily vol per name from long daily history (EWMA blended with the 1-year level, embedded below),
//     moved toward the history's own hourly realized variance through a market-wide ratio and a smaller per-name one;
//   - tails: Student-t marginals (exact CDF for the bins), a shared chi-square scale across names in the simulation;
//   - order: Monte Carlo of a multivariate t with a shrunk correlation matrix from daily history (Cholesky), so a
//     market-wide day moves every name together and only the dispersion decides the places;
//   - day 1: centred on most of the move already seen from the previous close's TWAP to the history's last bar,
//     with only the variance still to come.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

export type Bar = { t: number; o: number; c: number };
export type Prior = { vol: number; lr: number };
export type Params = {
  nu: number;          // Student-t degrees of freedom for each day's return
  gMarket: number;     // exponent on the market-wide realized/prior variance ratio
  gName: number;       // exponent on the name's own realized/market ratio
  day1Carry: number;   // share of the after-close move carried into day 1's mean
  hourShare: number;   // share of a day's variance one after-hours hour holds
  revert: number;      // per-day pull of a later day's variance back to the long-run level (0 none, 1 all)
  weekend: number;     // variance multiplier for a day whose previous session is more than 1 calendar day back
  draws: number;       // Monte Carlo draws for the order
  seed: number;
};

export const PARAMS: Params = {
  nu: 6, gMarket: 0.3, gName: 0.1, day1Carry: 0.9, hourShare: 0.03, revert: 0.05, weekend: 1.1,
  draws: 400000, seed: 20261110,
};

// Prior daily vol per xStock from the underlying shares' daily closes to 2026-11-06: vol is 0.4 x the EWMA variance
// (lambda 0.97) plus 0.6 x the trailing 250-day variance, lr the 250-day level alone (SPCX: since its 2026 listing).
export const NAMES = ['AAPLX', 'AMZNX', 'GOOGLX', 'METAX', 'MSFTX', 'MSTRX', 'NVDAX', 'SPCXX', 'TSLAX'];
export const PRIORS: Record<string, Prior> = {
  AAPLX: { vol: 0.0159, lr: 0.0157 },
  AMZNX: { vol: 0.0223, lr: 0.0214 },
  GOOGLX: { vol: 0.0234, lr: 0.0213 },
  METAX: { vol: 0.0267, lr: 0.0260 },
  MSFTX: { vol: 0.0215, lr: 0.0213 },
  MSTRX: { vol: 0.0517, lr: 0.0518 },
  NVDAX: { vol: 0.0254, lr: 0.0242 },
  SPCXX: { vol: 0.0506, lr: 0.0519 },
  TSLAX: { vol: 0.0303, lr: 0.0288 },
};
// Daily return correlations over the last 120 sessions, shrunk 30% toward their mean, in NAMES order.
export const CORR: number[][] = [
  [1.00, 0.04, 0.05, 0.11, 0.20, 0.17, 0.00, 0.14, 0.11],
  [0.04, 1.00, 0.54, 0.31, 0.42, 0.24, 0.21, 0.22, 0.29],
  [0.05, 0.54, 1.00, 0.29, 0.27, 0.28, 0.21, 0.28, 0.28],
  [0.11, 0.31, 0.29, 1.00, 0.17, 0.25, 0.20, 0.18, 0.31],
  [0.20, 0.42, 0.27, 0.17, 1.00, 0.30, 0.20, 0.19, 0.19],
  [0.17, 0.24, 0.28, 0.25, 0.30, 1.00, 0.33, 0.24, 0.35],
  [0.00, 0.21, 0.21, 0.20, 0.20, 0.33, 1.00, 0.30, 0.38],
  [0.14, 0.22, 0.28, 0.18, 0.19, 0.24, 0.30, 1.00, 0.30],
  [0.11, 0.29, 0.28, 0.31, 0.19, 0.35, 0.38, 0.30, 1.00],
];
const DEFAULT_PRIOR: Prior = { vol: 0.03, lr: 0.03 };
const DEFAULT_CORR = 0.25;
const LOG9 = Math.log(9);

// ---------- random numbers (sfc32, Box-Muller, Marsaglia-Tsang gamma) ----------
export function rng(seed: number): () => number {
  let a = 0x9e3779b9 ^ seed, b = 0x243f6a88, c = 0xb7e15162, d = seed | 0;
  const next = (): number => {
    a |= 0; b |= 0; c |= 0; d |= 0;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0;
    a = b ^ (b >>> 9);
    b = (c + (c << 3)) | 0;
    c = (c << 21) | (c >>> 11);
    c = (c + t) | 0;
    return (t >>> 0) / 4294967296;
  };
  for (let i = 0; i < 20; i++) next();
  return next;
}

export function normalSource(u: () => number): () => number {
  let spare = NaN;
  return () => {
    if (!Number.isNaN(spare)) { const s = spare; spare = NaN; return s; }
    let x = u();
    while (x <= 1e-300) x = u();
    const r = Math.sqrt(-2 * Math.log(x)), th = 2 * Math.PI * u();
    spare = r * Math.sin(th);
    return r * Math.cos(th);
  };
}

export function gammaDraw(shape: number, u: () => number, n: () => number): number {
  if (shape < 1) return gammaDraw(shape + 1, u, n) * Math.pow(Math.max(u(), 1e-300), 1 / shape);
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = n(), v = 1 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    const w = u();
    if (w < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(Math.max(w, 1e-300)) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

// ---------- distributions ----------
function logGamma(x: number): number {
  const g = [76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155,
    0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x, tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (const k of g) ser += k / ++y;
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

function betaContinuedFraction(a: number, b: number, x: number): number {
  const tiny = 1e-300;
  let c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < tiny) d = tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 300; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d; if (Math.abs(d) < tiny) d = tiny;
    c = 1 + aa / c; if (Math.abs(c) < tiny) c = tiny;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < 1e-15) break;
  }
  return h;
}

// Regularized incomplete beta I_x(a, b).
export function incBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  const bt = Math.exp(lbt);
  if (x < (a + 1) / (a + b + 2)) return bt * betaContinuedFraction(a, b, x) / a;
  return 1 - bt * betaContinuedFraction(b, a, 1 - x) / b;
}

export function normalCdf(z: number): number {
  // Complementary error function (Numerical Recipes erfcc, fractional error < 1.2e-7).
  const x = Math.abs(z) / Math.SQRT2, t = 1 / (1 + 0.5 * x);
  const r = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return z >= 0 ? 1 - r / 2 : r / 2;
}

// CDF of Student-t with nu degrees of freedom (nu > 200 treated as normal).
export function tCdf(x: number, nu: number): number {
  if (nu > 200) return normalCdf(x);
  const ib = incBeta(nu / (nu + x * x), nu / 2, 0.5);
  return x >= 0 ? 1 - ib / 2 : ib / 2;
}

// Probability of each of the 42 bins for a return mu + sd * (unit-variance t).
export function binProbs(mu: number, sd: number, nu: number, edges: number[]): number[] {
  const scale = nu > 200 ? sd : sd * Math.sqrt((nu - 2) / nu);
  const cdf = edges.map((e) => tCdf((e - mu) / scale, nu));
  const p: number[] = [cdf[0]];
  for (let i = 1; i < edges.length; i++) p.push(Math.max(0, cdf[i] - cdf[i - 1]));
  p.push(1 - cdf[edges.length - 1]);
  return p;
}

// ---------- linear algebra ----------
export function cholesky(m: number[][]): number[][] {
  const n = m.length, L = m.map(() => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = m[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) L[i][i] = Math.sqrt(Math.max(s, 1e-10));
      else L[i][j] = s / L[j][j];
    }
  }
  return L;
}

// ---------- history ----------
export function parseBars(text: string): Bar[] {
  const out: Bar[] = [];
  for (const line of text.split('\n')) {
    const s = line.trim();
    if (!s) continue;
    try {
      const j = JSON.parse(s);
      const t = Number(j.t), o = Number(j.o), c = Number(j.c);
      if (Number.isFinite(t) && o > 0 && c > 0) out.push({ t, o, c });
    } catch { /* a malformed line is skipped */ }
  }
  out.sort((a, b) => a.t - b.t);
  return out;
}

// Realized daily variance from hourly bars: 24 x mean squared hour-to-hour log return over the last 24 pairs.
export function hourlyVar(bars: Bar[], until: number): number | null {
  const rs: number[] = [];
  for (let i = bars.length - 1; i > 0 && rs.length < 24; i--) {
    if (bars[i].t > until) continue;
    if (bars[i].t - bars[i - 1].t === 3600000) rs.push(Math.log(bars[i].c / bars[i - 1].c));
  }
  if (rs.length < 6) return null;
  return 24 * rs.reduce((s, r) => s + r * r, 0) / rs.length;
}

// The move from the previous session's close TWAP to the history's last bar, and the hours it spans.
export function afterCloseMove(bars: Bar[], twapUntil: number, until: number): { x0: number; hours: number } {
  const usable = bars.filter((b) => b.t <= until);
  if (usable.length === 0) return { x0: 0, hours: 0 };
  const last = usable[usable.length - 1];
  const closeBar = usable.filter((b) => b.t <= twapUntil).pop();
  if (!closeBar || twapUntil - closeBar.t > 2 * 3600000 || last.t <= twapUntil) return { x0: 0, hours: 0 };
  // The TWAP runs over the last half hour of the bar that ends at twapUntil: o is the hour's first five
  // minutes and c its last five, so the TWAP sits about three quarters of the way from o to c.
  const twap = closeBar.t === twapUntil ? 0.25 * closeBar.o + 0.75 * closeBar.c : closeBar.c;
  return { x0: last.c / twap - 1, hours: (last.t - twapUntil) / 3600000 };
}

// ---------- the model ----------
export type DayInput = { date: string; previous: string };
export type NameState = { symbol: string; prior: Prior; hv: number | null; x0: number; hours: number };
export type DayForecast = {
  date: string;
  probabilities: { symbol: string; place: number; p: number }[];
  order: string[];
  sizes: { symbol: string; bin: number; p: number }[];
};

function dayGap(a: string, b: string): number {
  return Math.round((Date.parse(b + 'T00:00:00Z') - Date.parse(a + 'T00:00:00Z')) / 86400000);
}

export function forecastVariances(states: NameState[], days: DayInput[], prm: Params): { mu: number[][]; sd: number[][] } {
  // Market-wide ratio of realized to prior variance (geometric mean over names with history).
  // Each log ratio is clamped to +-ln 9 so one broken pool cannot swing a forecast.
  const clampLog = (x: number): number => Math.max(-LOG9, Math.min(LOG9, x));
  const ratios = states.filter((s) => s.hv !== null && s.hv > 0).map((s) => clampLog(Math.log((s.hv as number) / (s.prior.vol ** 2))));
  const mkt = ratios.length ? ratios.reduce((a, b) => a + b, 0) / ratios.length : 0;
  const mu: number[][] = [], sd: number[][] = [];
  days.forEach((day, d) => {
    const muD: number[] = [], sdD: number[] = [];
    for (const s of states) {
      let lv = 2 * Math.log(s.prior.vol) + prm.gMarket * mkt;
      if (s.hv !== null && s.hv > 0) lv += prm.gName * (clampLog(Math.log(s.hv / (s.prior.vol ** 2))) - mkt);
      let v = Math.exp(lv);
      const lr = s.prior.lr ** 2;
      v = lr + (v - lr) * Math.pow(1 - prm.revert, d);
      if (dayGap(day.previous, day.date) > 1) v *= prm.weekend;
      let m = 0;
      if (d === 0) {
        m = prm.day1Carry * s.x0;
        v *= Math.max(0.5, 1 - prm.hourShare * s.hours);
      }
      muD.push(m);
      sdD.push(Math.max(0.004, Math.sqrt(v)));
    }
    mu.push(muD); sd.push(sdD);
  });
  return { mu, sd };
}

function floor6(p: number): number {
  return Math.floor(p * 1e6) / 1e6;
}

export function forecast(states: NameState[], corr: number[][], days: DayInput[], edges: number[], places: number,
  prm: Params): DayForecast[] {
  const n = states.length;
  const { mu, sd } = forecastVariances(states, days, prm);
  const L = cholesky(corr);
  const u = rng(prm.seed), g = normalSource(u);
  const tScale = prm.nu > 200 ? 1 : Math.sqrt((prm.nu - 2) / prm.nu);
  const out: DayForecast[] = [];
  days.forEach((day, d) => {
    // Order: simulate the day's returns, count each name's place.
    const counts = states.map(() => new Array(n).fill(0));
    const z = new Array(n).fill(0), e = new Array(n).fill(0), r = new Array(n).fill(0), idx = states.map((_, i) => i);
    for (let k = 0; k < prm.draws; k++) {
      for (let i = 0; i < n; i++) e[i] = g();
      for (let i = 0; i < n; i++) { let s = 0; for (let j = 0; j <= i; j++) s += L[i][j] * e[j]; z[i] = s; }
      const w = prm.nu > 200 ? 1 : Math.sqrt(prm.nu / (2 * gammaDraw(prm.nu / 2, u, g)));
      for (let i = 0; i < n; i++) r[i] = mu[d][i] + sd[d][i] * tScale * w * z[i];
      idx.sort((a, b) => r[b] - r[a] || a - b);
      for (let p = 0; p < n; p++) counts[idx[p]][p]++;
    }
    const probabilities: DayForecast['probabilities'] = [];
    for (let p = 0; p < Math.min(places, n); p++) {
      for (let i = 0; i < n; i++) {
        probabilities.push({ symbol: states[i].symbol, place: p + 1, p: floor6((counts[i][p] + 0.5) / (prm.draws + 0.5 * n)) });
      }
    }
    // Best guess at the order: by expected place.
    const expPlace = states.map((_, i) => counts[i].reduce((s, c, p) => s + c * p, 0) / prm.draws);
    const order = states.map((_, i) => i).sort((a, b) => expPlace[a] - expPlace[b] || a - b)
      .slice(0, Math.min(places, n)).map((i) => states[i].symbol);
    const sizes: DayForecast['sizes'] = [];
    states.forEach((s, i) => {
      binProbs(mu[d][i], sd[d][i], prm.nu, edges).forEach((p, b) => {
        const q = floor6(p);
        if (q > 0) sizes.push({ symbol: s.symbol, bin: b, p: q });
      });
    });
    out.push({ date: day.date, probabilities, order, sizes });
  });
  return out;
}

export function corrFor(symbols: string[]): number[][] {
  return symbols.map((a) => symbols.map((b) => {
    if (a === b) return 1;
    const i = NAMES.indexOf(a), j = NAMES.indexOf(b);
    if (i >= 0 && j >= 0 && CORR.length) return CORR[i][j];
    return DEFAULT_CORR;
  }));
}

// ---------- entrypoint ----------
function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || '/input';
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || '/output';
  const task = JSON.parse(readFileSync(join(inDir, 'task.json'), 'utf8'));
  const until = Date.parse(task.history_until);
  const days: DayInput[] = task.window.days.map((d: any) => ({ date: d.date, previous: d.previous }));
  const first = task.window.days[0];
  // The previous session's close TWAP ends at the same clock time as day 1's, on the previous date.
  const prevUntil = Date.parse(first.twap_until) - dayGap(first.previous, first.date) * 86400000;
  const states: NameState[] = task.universe.map((u: any) => {
    let bars: Bar[] = [];
    try { bars = parseBars(readFileSync(join(inDir, u.file), 'utf8')); } catch { bars = []; }
    const { x0, hours } = afterCloseMove(bars, prevUntil, until);
    return { symbol: u.symbol, prior: PRIORS[u.symbol] || DEFAULT_PRIOR, hv: hourlyVar(bars, until), x0, hours };
  });
  const days5 = forecast(states, corrFor(states.map((s) => s.symbol)), days, task.bin_edges, task.places, PARAMS);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(join(outDir, task.output || 'forecast.json'), JSON.stringify({ days: days5 }));
}

// The backtest imports this file as a library (XSM_LIBRARY=1); every other run writes the forecast.
if (process.env.XSM_LIBRARY !== '1') main();
