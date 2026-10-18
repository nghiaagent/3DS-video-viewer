import { describe, it, expect } from 'vitest';
import { grainSeedForTime } from '../../src/renderer/grain.js';
import { PERSPECTIVE_WINDOW_FRAG_SOURCE } from '../../src/renderer/shaders.js';

describe('grainSeedForTime', () => {
  it('changes at the source frame rate, not the render rate', () => {
    expect(grainSeedForTime(1.0)).toBe(grainSeedForTime(1.0 + 0.01));
    expect(grainSeedForTime(1.0)).not.toBe(grainSeedForTime(1.0 + 0.05));
  });
  it('is shared by live and export for the same moment', () => {
    expect(grainSeedForTime(60 / 30, 30)).toBe(60);
  });
});

describe('mode-0 window finish shader', () => {
  it('declares the rim and grain uniforms', () => {
    for (const u of ['u_canvasSize', 'u_edgeSoftness', 'u_edgeMotion', 'u_grain', 'u_grainSeed']) {
      expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain(`uniform`);
      expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain(u);
    }
  });
});
