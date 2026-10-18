import { describe, it, expect } from 'vitest';
import {
  fitAffineInvDepth, alignWindows, smoothAffineParams,
  computeClipNormalization, normalizeWithClip,
} from '../../src/depth/windowAlignment.js';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
}

const PIX = 400;
const FRAMES = 40;

/** Ground truth: scene whose inverse depth range drifts over time. */
function makeTruth(): Float32Array[] {
  const out: Float32Array[] = [];
  for (let f = 0; f < FRAMES; f++) {
    const fr = new Float32Array(PIX);
    const near = 1.0 + 1.5 * Math.sin(f / 9);
    for (let i = 0; i < PIX; i++) fr[i] = 0.5 + near * (i / PIX) + 0.2 * Math.sin(i / 17 + f / 5);
    out.push(fr);
  }
  return out;
}

function makeWindows(truth: Float32Array[], size: number, stride: number, noise: number, seed: number) {
  const r = rng(seed);
  const wins = [];
  for (let s = 0; s + size <= truth.length || s === 0; s += stride) {
    const a = 0.4 + 2.5 * r(), b = -1 + 2 * r();
    const invDepth = truth.slice(s, s + size).map((fr) => {
      const o = new Float32Array(fr.length);
      for (let i = 0; i < fr.length; i++) o[i] = (fr[i] - b) / a + noise * (r() - 0.5);
      return o;
    });
    wins.push({ startFrame: s, invDepth });
    if (s + size >= truth.length) break;
  }
  return wins;
}

function globalAffineRMSE(est: Float32Array[], truth: Float32Array[]) {
  const e = new Float32Array(est.length * PIX), t = new Float32Array(est.length * PIX);
  est.forEach((f, k) => { e.set(f, k * PIX); t.set(truth[k], k * PIX); });
  const p = fitAffineInvDepth(e, t);
  let se = 0;
  for (let i = 0; i < e.length; i++) se += (p.a * e[i] + p.b - t[i]) ** 2;
  return Math.sqrt(se / e.length);
}

function seamError(est: Float32Array[], truth: Float32Array[]) {
  // frame-to-frame change error versus truth, after one global affine
  const e = new Float32Array(est.length * PIX), t = new Float32Array(est.length * PIX);
  est.forEach((f, k) => { e.set(f, k * PIX); t.set(truth[k], k * PIX); });
  const p = fitAffineInvDepth(e, t);
  let worst = 0;
  for (let k = 1; k < est.length; k++) {
    let s = 0;
    for (let i = 0; i < PIX; i++) {
      const d = (p.a * est[k][i] + p.b - truth[k][i]) - (p.a * est[k - 1][i] + p.b - truth[k - 1][i]);
      s += d * d;
    }
    worst = Math.max(worst, Math.sqrt(s / PIX));
  }
  return worst;
}

describe('fitAffineInvDepth', () => {
  it('recovers exact affine', () => {
    const src = new Float32Array(100).map((_, i) => i / 10);
    const dst = src.map((v) => 2.5 * v - 1);
    const p = fitAffineInvDepth(src, dst);
    expect(p.a).toBeCloseTo(2.5, 4);
    expect(p.b).toBeCloseTo(-1, 4);
  });

  it('is robust to outliers and ignores non-finite values', () => {
    const r = rng(3);
    const src = new Float32Array(1000).map(() => r() * 3);
    const dst = src.map((v) => 1.7 * v + 0.4);
    for (let i = 0; i < 100; i++) dst[i] = 50 * r();
    src[500] = NaN; dst[600] = Infinity;
    const p = fitAffineInvDepth(src, dst);
    expect(p.a).toBeCloseTo(1.7, 1);
    expect(p.b).toBeCloseTo(0.4, 1);
  });

  it('honours weights', () => {
    const src = new Float32Array(200).map((_, i) => i / 20);
    const dst = src.map((v, i) => (i < 100 ? v : 3 * v + 5));
    const w = new Float32Array(200).map((_, i) => (i < 100 ? 1 : 0));
    const p = fitAffineInvDepth(src, dst, w);
    expect(p.a).toBeCloseTo(1, 3);
    expect(p.b).toBeCloseTo(0, 3);
  });

  it('handles empty/degenerate input', () => {
    expect(fitAffineInvDepth(new Float32Array(0), new Float32Array(0))).toEqual({ a: 1, b: 0 });
    const p = fitAffineInvDepth(new Float32Array(10).fill(2), new Float32Array(10).fill(5));
    expect(p.a * 2 + p.b).toBeCloseTo(5, 5);
  });
});

describe('alignWindows', () => {
  it('recovers the sequence up to one global affine with low seam error', () => {
    const truth = makeTruth();
    const wins = makeWindows(truth, 10, 6, 0.01, 7);
    const out = alignWindows(wins);
    expect(out.length).toBe(FRAMES);
    expect(globalAffineRMSE(out, truth)).toBeLessThan(0.05);
    expect(seamError(out, truth)).toBeLessThan(0.05);
  });

  it('handles a single window and empty input', () => {
    expect(alignWindows([])).toEqual([]);
    const f = [new Float32Array([1, 2, 3])];
    const out = alignWindows([{ startFrame: 4, invDepth: f }]);
    expect(Array.from(out[0])).toEqual([1, 2, 3]);
  });

  it('per-frame normalization does NOT recover consistency (documents flicker)', () => {
    const truth = makeTruth();
    const wins = makeWindows(truth, 10, 6, 0.01, 7);
    const aligned = alignWindows(wins);
    const perFrame = (frames: Float32Array[]) => frames.map((f) => {
      const { lo, hi } = computeClipNormalization([f]);
      return normalizeWithClip(f, lo, hi);
    });
    const naive = perFrame(wins.flatMap((w) => w.invDepth).slice(0, FRAMES));
    const truthNormPerFrame = perFrame(truth);
    // Per-frame norm of ground truth itself already shifts with drifting range: flicker.
    const clip = computeClipNormalization(aligned);
    const alignedNorm = aligned.map((f) => normalizeWithClip(f, clip.lo, clip.hi));
    const truthClip = computeClipNormalization(truth);
    const truthNorm = truth.map((f) => normalizeWithClip(f, truthClip.lo, truthClip.hi));
    const err = (a: Float32Array[]) => {
      let s = 0, n = 0;
      for (let k = 0; k < FRAMES; k++) for (let i = 0; i < PIX; i++) { s += (a[k][i] - truthNorm[k][i]) ** 2; n++; }
      return Math.sqrt(s / n);
    };
    expect(err(alignedNorm)).toBeLessThan(0.08);
    expect(err(perFrame(aligned))).toBeGreaterThan(err(alignedNorm) * 2);
    expect(err(naive)).toBeGreaterThan(err(alignedNorm) * 2);
    expect(err(truthNormPerFrame)).toBeGreaterThan(err(alignedNorm));
  });
});

describe('smoothAffineParams', () => {
  it('reduces jitter and preserves constants', () => {
    const r = rng(5);
    const noisy = Array.from({ length: 50 }, () => ({ a: 1 + 0.2 * (r() - 0.5), b: 0.1 * (r() - 0.5) }));
    const sm = smoothAffineParams(noisy, 0.3);
    const tv = (xs: { a: number }[]) => xs.slice(1).reduce((s, x, i) => s + Math.abs(x.a - xs[i].a), 0);
    expect(tv(sm)).toBeLessThan(tv(noisy) * 0.6);
    const c = smoothAffineParams(Array(10).fill({ a: 2, b: 1 }), 0.2);
    c.forEach((p) => { expect(p.a).toBeCloseTo(2, 6); expect(p.b).toBeCloseTo(1, 6); });
    expect(smoothAffineParams([], 0.5)).toEqual([]);
  });
});

describe('clip normalization', () => {
  it('computes robust global range and clamps', () => {
    const f = new Float32Array(1000).map((_, i) => i / 999);
    f[0] = -1000; f[1] = 1e6; f[2] = NaN;
    const { lo, hi } = computeClipNormalization([f, f]);
    expect(lo).toBeGreaterThan(0.0);
    expect(lo).toBeLessThan(0.05);
    expect(hi).toBeGreaterThan(0.95);
    expect(hi).toBeLessThan(1.01);
    const n = normalizeWithClip(f, lo, hi);
    expect(Math.min(...n)).toBe(0);
    expect(Math.max(...n)).toBe(1);
  });
});
