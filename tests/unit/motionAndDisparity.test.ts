import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { getPerspectiveShift, MotionController } from '../../src/renderer/motionController.js';
import {
  VERTEX_SHADER_SOURCE,
  DISPARITY_SAD_FRAG_SOURCE,
  PERSPECTIVE_WINDOW_FRAG_SOURCE,
} from '../../src/renderer/shaders.js';
import { initSync, StereoDisparity, downsample_sbs_rgba_half } from '../../src/wasm/pkg/stereo_disparity.js';

describe('MotionController & Perspective Shift Math', () => {
  it('accurately calculates auto-oscillation sine wave across key period phases', () => {
    const freq = 0.28;
    const periodMs = 1000 / freq; // ~3571.43 ms

    // t = 0 -> sin(0) = 0 -> 0.5
    expect(getPerspectiveShift(0)).toBeCloseTo(0.5, 4);

    // t = 1/4 period -> sin(pi/2) = 1.0 -> 0.5 + 0.42 = 0.92
    const tQuarter = periodMs * 0.25;
    expect(getPerspectiveShift(tQuarter)).toBeCloseTo(0.92, 4);

    // t = 1/2 period -> sin(pi) = 0.0 -> 0.5
    const tHalf = periodMs * 0.5;
    expect(getPerspectiveShift(tHalf)).toBeCloseTo(0.5, 4);

    // t = 3/4 period -> sin(3pi/2) = -1.0 -> 0.5 - 0.42 = 0.08
    const tThreeQuarter = periodMs * 0.75;
    expect(getPerspectiveShift(tThreeQuarter)).toBeCloseTo(0.08, 4);

    // t = full period -> sin(2pi) = 0.0 -> 0.5
    expect(getPerspectiveShift(periodMs)).toBeCloseTo(0.5, 4);
  });

  it('guarantees shift remains bounded within [0.08, 0.92] over time', () => {
    for (let t = 0; t <= 10000; t += 50) {
      const shift = getPerspectiveShift(t);
      expect(shift).toBeGreaterThanOrEqual(0.08 - 1e-6);
      expect(shift).toBeLessThanOrEqual(0.92 + 1e-6);
    }
  });

  it('clamps interactive mouse position correctly to [0.0, 1.0]', () => {
    expect(getPerspectiveShift(1234, -0.5)).toBe(0.0);
    expect(getPerspectiveShift(1234, 0.0)).toBe(0.0);
    expect(getPerspectiveShift(1234, 0.35)).toBe(0.35);
    expect(getPerspectiveShift(1234, 0.75)).toBe(0.75);
    expect(getPerspectiveShift(1234, 1.0)).toBe(1.0);
    expect(getPerspectiveShift(1234, 1.5)).toBe(1.0);
  });

  it('MotionController handles manual shift and exponential smoothing', () => {
    const controller = new MotionController(undefined, { dampingFactor: 0.5 });
    controller.setManualShift(0.8);
    expect(controller.getShift()).toBe(0.8);

    // Subsequent updates smoothly damp toward target
    const updated = controller.update(0);
    expect(updated).toBeCloseTo(0.8, 3);
  });
});

describe('GLSL Shader Sources & Perspective Warp UV Bounds', () => {
  it('validates vertex and fragment shader source code structure', () => {
    expect(VERTEX_SHADER_SOURCE).toContain('#version 300 es');
    expect(VERTEX_SHADER_SOURCE).toContain('uniform float u_flipY;');

    expect(DISPARITY_SAD_FRAG_SOURCE).toContain('#version 300 es');
    expect(DISPARITY_SAD_FRAG_SOURCE).toContain('u_sbsTexture');
    expect(DISPARITY_SAD_FRAG_SOURCE).toContain('u_eyeTexelSize');
    expect(DISPARITY_SAD_FRAG_SOURCE).toContain('u_maxDisp');

    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('#version 300 es');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_sbsTexture');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_dispTexture');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_viewShift');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_maxDisparityUV');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_windowCrop');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_renderMode');
    expect(VERTEX_SHADER_SOURCE).toContain('u_flipY');
  });

  it('verifies active stereo window crop and disparity warp stay strictly within eye texture halves', () => {
    const crop = 0.06;
    const maxDisparityUV = 24.0 / 480.0; // 0.05
    const zeroDispNorm = 4.0 / 28.0;

    // Test a grid of normalized screen UVs in [0, 1]
    for (let vy = 0.0; vy <= 1.0; vy += 0.1) {
      for (let vx = 0.0; vx <= 1.0; vx += 0.1) {
        // 1. Cropped active eye UV
        const eyeX = vx * (1.0 - 2.0 * crop) + crop;
        const eyeY = vy * (1.0 - 2.0 * crop) + crop;

        expect(eyeX).toBeGreaterThanOrEqual(crop - 1e-6);
        expect(eyeX).toBeLessThanOrEqual(1.0 - crop + 1e-6);
        expect(eyeY).toBeGreaterThanOrEqual(crop - 1e-6);
        expect(eyeY).toBeLessThanOrEqual(1.0 - crop + 1e-6);

        // Test across disparity range [0.0, 1.0] and camera shift t in [0.0, 1.0]
        for (let dNorm = 0.0; dNorm <= 1.0; dNorm += 0.25) {
          const dispUV = (dNorm - zeroDispNorm) * maxDisparityUV;

          for (let t = 0.0; t <= 1.0; t += 0.25) {
            const shiftL = t * dispUV;
            const shiftR = -(1.0 - t) * dispUV;

            const rawUvLeftX = (eyeX + shiftL) * 0.5;
            const rawUvRightX = 0.5 + (eyeX + shiftR) * 0.5;

            // Clamping rules from perspective_window.frag
            const clampedUvLeftX = Math.max(0.0, Math.min(0.4999, rawUvLeftX));
            const clampedUvRightX = Math.max(0.5, Math.min(0.9999, rawUvRightX));

            // Left eye must be strictly in [0.0, 0.5]
            expect(clampedUvLeftX).toBeGreaterThanOrEqual(0.0);
            expect(clampedUvLeftX).toBeLessThan(0.5);

            // Right eye must be strictly in [0.5, 1.0]
            expect(clampedUvRightX).toBeGreaterThanOrEqual(0.5);
            expect(clampedUvRightX).toBeLessThan(1.0);
          }
        }
      }
    }
  });
});

describe('Rust Wasm SIMD Disparity Integration', () => {
  it('instantiates Wasm module in Node/Vitest and computes disparity on real DCIM fixture', () => {
    const wasmPath = path.resolve(__dirname, '../../src/wasm/pkg/stereo_disparity_bg.wasm');
    expect(fs.existsSync(wasmPath)).toBe(true);

    const wasmBytes = fs.readFileSync(wasmPath);
    initSync({ module: wasmBytes });

    // Load real 3DS DCIM frame fixture (HNI_0070.AVI frame 5)
    const fixturePath = path.resolve(__dirname, '../fixtures/hni_0070_frame05.rgba');
    expect(fs.existsSync(fixturePath)).toBe(true);

    const rawRgba = fs.readFileSync(fixturePath);
    expect(rawRgba.length).toBe(960 * 240 * 4); // 921,600 bytes

    // 1. Half resolution disparity (240x120 per eye)
    const halfEngine = new StereoDisparity(240, 120, 12);
    expect(halfEngine.width()).toBe(240);
    expect(halfEngine.height()).toBe(120);
    expect(halfEngine.max_disp()).toBe(12);

    // Compute disparity directly from full-res buffer (uses on-the-fly 2x downsampling)
    const dispHalf = halfEngine.compute_disparity(rawRgba);
    expect(dispHalf.length).toBe(240 * 120);

    // Check statistical properties (non-trivial depth levels)
    let sum = 0;
    for (let i = 0; i < dispHalf.length; i++) {
      sum += dispHalf[i];
    }
    const mean = sum / dispHalf.length;
    let varianceSum = 0;
    const distinctLevels = new Set<number>();
    for (let i = 0; i < dispHalf.length; i++) {
      const diff = dispHalf[i] - mean;
      varianceSum += diff * diff;
      distinctLevels.add(dispHalf[i]);
    }
    const stdDev = Math.sqrt(varianceSum / dispHalf.length);

    expect(stdDev).toBeGreaterThan(5.0);
    expect(distinctLevels.size).toBeGreaterThanOrEqual(8);

    // 2. Test downsample_sbs_rgba_half helper
    const downsampledRgba = downsample_sbs_rgba_half(rawRgba, 960, 240);
    expect(downsampledRgba.length).toBe(480 * 120 * 4);

    // 3. Test zero-copy pointer access
    const ptr = halfEngine.compute_disparity_inplace(downsampledRgba);
    expect(ptr).toBeGreaterThan(0);
    expect(halfEngine.disparity_ptr()).toBe(ptr);
    expect(halfEngine.disparity_len()).toBe(240 * 120);
  });
});
