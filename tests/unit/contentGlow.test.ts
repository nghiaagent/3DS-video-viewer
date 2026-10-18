import { describe, it, expect } from 'vitest';
import { DEFAULT_GLOW, edgeGlowRgb, glowColors, quantizeGlow, smoothGlow } from '../../src/renderer/contentGlow.js';

describe('glowColors', () => {
  it('falls back to the default for missing, NaN and black input', () => {
    expect(glowColors(null)).toEqual(DEFAULT_GLOW);
    expect(glowColors([NaN, 0, 0])).toEqual(DEFAULT_GLOW);
    expect(glowColors([0, 0, 0])).toEqual(DEFAULT_GLOW);
  });

  it('fades to the default smoothly as the picture darkens (no jump)', () => {
    const at = (v: number) => glowColors([v, v * 0.2, v * 0.3]).c1[0];
    let prev = at(60);
    for (let v = 59; v >= 0; v--) {
      const cur = at(v);
      expect(Math.abs(cur - prev)).toBeLessThan(25);
      prev = cur;
    }
  });

  it('keeps a saturated pink picture pink-dominant', () => {
    const g = glowColors([200, 60, 120]);
    expect(g.c1[0]).toBeGreaterThan(g.c1[1]);
    expect(g.c1[0]).toBeGreaterThan(g.c1[2]);
    expect(Math.max(...g.c1)).toBeCloseTo(255, 3);
  });

  it('scales opacity with brightness but never above the default', () => {
    const dim = glowColors([60, 20, 30]);
    const bright = glowColors([250, 90, 120]);
    expect(dim.a1).toBeLessThanOrEqual(bright.a1);
    expect(bright.a1).toBeLessThanOrEqual(DEFAULT_GLOW.a1 + 1e-9);
    expect(bright.a2).toBeCloseTo(bright.a1 / 2, 6);
  });
});

describe('smoothGlow', () => {
  it('is within 1% of the target after five time constants', () => {
    const target = glowColors([220, 40, 60]);
    let g = DEFAULT_GLOW;
    for (let i = 0; i < 300; i++) g = smoothGlow(g, target, 1 / 60, 0.9 / 5 * 1); // 5 s of steps at tau/5
    expect(Math.abs(g.c1[0] - target.c1[0])).toBeLessThan(target.c1[0] * 0.01 + 1e-6);
  });

  it('does not move for dt = 0', () => {
    expect(smoothGlow(DEFAULT_GLOW, glowColors([220, 40, 60]), 0)).toEqual(DEFAULT_GLOW);
  });
});

describe('quantizeGlow', () => {
  it('leaves the default exactly unchanged', () => {
    expect(quantizeGlow(DEFAULT_GLOW)).toEqual(DEFAULT_GLOW);
  });
});

describe('edgeGlowRgb', () => {
  it('lightens the glow colour toward white but keeps its hue', () => {
    const e = edgeGlowRgb(glowColors([230, 40, 60]));
    expect(e[0]).toBeGreaterThan(e[1]);
    expect(Math.min(...e)).toBeGreaterThan(100); // pale, not saturated
  });
  it('matches the CSS default for the default glow', () => {
    expect(edgeGlowRgb(DEFAULT_GLOW)).toEqual([156, 216, 244]);
  });
});
