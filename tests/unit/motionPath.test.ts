import { describe, it, expect } from 'vitest';
import {
  DEFAULT_ORBIT,
  getOrbitOffset,
  getOrbitOffsetFromPhase,
  getWindowPose,
  offsetToRenderParams,
  viewShiftToX,
} from '../../src/renderer/motionPath.js';

describe('getOrbitOffset', () => {
  const s = { freqHz: 0.25, amp: 0.6, yRatio: 0.5 };

  it('starts at the top of the ellipse and quarter-turns to the side', () => {
    const o0 = getOrbitOffset(0, s);
    expect(o0.x).toBeCloseTo(0, 6);
    expect(o0.y).toBeCloseTo(0.6 * 0.5, 6);

    const quarter = getOrbitOffset(1 / s.freqHz / 4, s); // a quarter of the period
    expect(quarter.x).toBeCloseTo(0.6, 6);
    expect(quarter.y).toBeCloseTo(0, 6);
  });

  it('is periodic', () => {
    const period = 1 / s.freqHz;
    const a = getOrbitOffset(0.37, s);
    const b = getOrbitOffset(0.37 + period, s);
    expect(b.x).toBeCloseTo(a.x, 6);
    expect(b.y).toBeCloseTo(a.y, 6);
  });

  it('always lies on the ellipse x²/amp² + y²/(amp*yRatio)² = 1 (so it moves in all four directions)', () => {
    const seenSigns = new Set<string>();
    for (let t = 0; t < 4; t += 0.05) {
      const { x, y } = getOrbitOffset(t, s);
      const e = (x / s.amp) ** 2 + (y / (s.amp * s.yRatio)) ** 2;
      expect(e).toBeCloseTo(1, 6);
      if (Math.abs(x) > 0.05 && Math.abs(y) > 0.05) seenSigns.add(`${Math.sign(x)}${Math.sign(y)}`);
    }
    expect(seenSigns.size).toBe(4); // all four quadrants visited
  });

  it('phase0 shifts the starting angle', () => {
    const o = getOrbitOffset(0, s, Math.PI / 2);
    expect(o.x).toBeCloseTo(0.6, 6);
    expect(o.y).toBeCloseTo(0, 6);
  });

  it('clamps to [-1, 1]', () => {
    const o = getOrbitOffset(1 / 4, { freqHz: 1, amp: 3, yRatio: 1 });
    expect(o.x).toBe(1);
    expect(Math.abs(o.y)).toBeLessThanOrEqual(1);
  });

  it('matches getOrbitOffsetFromPhase for the same angle', () => {
    const t = 1.234;
    const phi = t * DEFAULT_ORBIT.freqHz * Math.PI * 2;
    const a = getOrbitOffset(t, DEFAULT_ORBIT);
    const b = getOrbitOffsetFromPhase(phi, DEFAULT_ORBIT.amp, DEFAULT_ORBIT.yRatio);
    expect(b.x).toBeCloseTo(a.x, 9);
    expect(b.y).toBeCloseTo(a.y, 9);
  });
});

describe('offsetToRenderParams', () => {
  it('maps the centre to viewShift 0.5 and tilt 0', () => {
    expect(offsetToRenderParams({ x: 0, y: 0 })).toEqual({ viewShift: 0.5, verticalTilt: 0 });
  });

  it('maps the extremes to the full range and clamps beyond it', () => {
    expect(offsetToRenderParams({ x: 1, y: -1 })).toEqual({ viewShift: 1, verticalTilt: -1 });
    expect(offsetToRenderParams({ x: -1, y: 1 })).toEqual({ viewShift: 0, verticalTilt: 1 });
    expect(offsetToRenderParams({ x: 5, y: -5 })).toEqual({ viewShift: 1, verticalTilt: -1 });
  });

  it('round-trips the horizontal axis with viewShiftToX', () => {
    for (const x of [-1, -0.4, 0, 0.25, 0.9]) {
      expect(viewShiftToX(offsetToRenderParams({ x, y: 0 }).viewShift)).toBeCloseTo(x, 9);
    }
  });
});

describe('getWindowPose', () => {
  it('is neutral at the centre', () => {
    const p = getWindowPose({ x: 0, y: 0 });
    expect(p.tx).toBe(0);
    expect(p.ty).toBe(0);
    expect(Math.abs(p.rotateYDeg)).toBe(0);
    expect(Math.abs(p.rotateXDeg)).toBe(0);
    expect(p.shadowY).toBe(32);
    expect(p.specAngleDeg).toBe(135);
  });

  it('is mirror-symmetric in x', () => {
    const r = getWindowPose({ x: 0.6, y: 0 });
    const l = getWindowPose({ x: -0.6, y: 0 });
    expect(l.tx).toBeCloseTo(-r.tx, 9);
    expect(l.rotateYDeg).toBeCloseTo(-r.rotateYDeg, 9);
    expect(l.shadowX).toBeCloseTo(-r.shadowX, 9);
    expect(l.glowX).toBeCloseTo(-r.glowX, 9);
    expect(l.specAngleDeg - 135).toBeCloseTo(-(r.specAngleDeg - 135), 9);
  });

  it('responds to the vertical axis too (the window tilts in both directions)', () => {
    const up = getWindowPose({ x: 0, y: -0.8 });
    const down = getWindowPose({ x: 0, y: 0.8 });
    expect(down.ty).toBeCloseTo(-up.ty, 9);
    expect(down.rotateXDeg).toBeCloseTo(-up.rotateXDeg, 9);
    expect(down.glowY).toBeCloseTo(-up.glowY, 9);
    expect(down.shadowY).toBeGreaterThan(up.shadowY);
  });

  it('clamps out-of-range offsets', () => {
    const p = getWindowPose({ x: 9, y: -9 });
    expect(p.x).toBe(1);
    expect(p.y).toBe(-1);
  });
});
