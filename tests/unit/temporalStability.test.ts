import { describe, it, expect } from 'vitest';
import { staticMask, flickerScore } from '../../src/depth/temporalStability.js';

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const W = 8;
const H = 8;
const N = W * H;

function baseDepth(): Float32Array {
  const d = new Float32Array(N);
  for (let i = 0; i < N; i++) d[i] = (i % W) / W;
  return d;
}

function seq(count: number, fn: (base: Float32Array, t: number) => Float32Array) {
  const base = baseDepth();
  return Array.from({ length: count }, (_, t) => fn(base, t));
}

describe('staticMask', () => {
  it('marks unchanged pixels static and changed pixels not', () => {
    const f0 = new Uint8Array(N).fill(100);
    const f1 = new Uint8Array(N).fill(102);
    f1[5] = 200;
    const f2 = new Uint8Array(N).fill(100);
    f2[5] = 200;
    const mask = staticMask([f0, f1, f2], W, H, 5);
    expect(mask[5]).toBe(0);
    expect(mask[0]).toBe(1);
  });

  it('works with Float32 frames', () => {
    const a = new Float32Array(N);
    const b = new Float32Array(N);
    b[3] = 0.5;
    const mask = staticMask([a, b], W, H, 0.1);
    expect(mask[3]).toBe(0);
    expect(mask[4]).toBe(1);
  });
});

describe('flickerScore', () => {
  const allMask = new Uint8Array(N).fill(1);

  it('is zero for constant depth', () => {
    const s = flickerScore(seq(10, (b) => b.slice()), allMask);
    expect(s.meanTemporalStd).toBeCloseTo(0, 6);
    expect(s.meanPumping).toBeCloseTo(0, 6);
    expect(s.frameToFrameMAD).toBeCloseTo(0, 6);
    expect(s.maskedPixels).toBe(N);
  });

  it('is large for per-frame random scale/shift and ranks above constant', () => {
    const r = rng(42);
    const flicker = seq(20, (b) => {
      const scale = 0.5 + r();
      const shift = r() - 0.5;
      return b.map((v) => v * scale + shift);
    });
    const stable = flickerScore(seq(20, (b) => b.slice()), allMask);
    const bad = flickerScore(flicker, allMask);
    expect(bad.meanTemporalStd).toBeGreaterThan(0.1);
    expect(bad.meanPumping).toBeGreaterThan(0.05);
    expect(bad.frameToFrameMAD).toBeGreaterThan(0.1);
    expect(bad.meanTemporalStd).toBeGreaterThan(stable.meanTemporalStd);
    expect(bad.meanPumping).toBeGreaterThan(stable.meanPumping);
    expect(bad.frameToFrameMAD).toBeGreaterThan(stable.frameToFrameMAD);
  });

  it('ranks mild jitter between constant and heavy flicker', () => {
    const mk = (amp: number, seed: number) => {
      const r = rng(seed);
      return flickerScore(seq(20, (b) => b.map((v) => v + amp * (r() - 0.5))), allMask);
    };
    const none = mk(0, 1);
    const mild = mk(0.02, 1);
    const heavy = mk(0.5, 1);
    expect(mild.frameToFrameMAD).toBeGreaterThan(none.frameToFrameMAD);
    expect(heavy.frameToFrameMAD).toBeGreaterThan(mild.frameToFrameMAD);
    expect(heavy.meanTemporalStd).toBeGreaterThan(mild.meanTemporalStd);
  });

  it('only counts masked pixels', () => {
    const mask = new Uint8Array(N);
    mask[0] = 1;
    const frames = seq(10, (b, t) => {
      const d = b.slice();
      for (let i = 1; i < N; i++) d[i] = t % 2; // flickers, but unmasked
      return d;
    });
    const s = flickerScore(frames, mask);
    expect(s.maskedPixels).toBe(1);
    expect(s.meanTemporalStd).toBeCloseTo(0, 6);
    expect(s.frameToFrameMAD).toBeCloseTo(0, 6);
  });

  it('pumping detects global shifts even when pixel pattern is preserved', () => {
    const frames = seq(10, (b, t) => b.map((v) => v + (t % 2) * 0.4));
    const s = flickerScore(frames, allMask);
    expect(s.meanPumping).toBeCloseTo(0.2, 3);
    expect(s.frameToFrameMAD).toBeCloseTo(0.4, 5);
  });

  it('handles empty mask / single frame', () => {
    expect(flickerScore(seq(5, (b) => b.slice()), new Uint8Array(N)).maskedPixels).toBe(0);
    expect(flickerScore(seq(1, (b) => b.slice()), allMask).frameToFrameMAD).toBe(0);
  });
});
