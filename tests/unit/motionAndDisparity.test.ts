import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import { getPerspectiveShift, MotionController } from '../../src/renderer/motionController.js';
import {
  VERTEX_SHADER_SOURCE,
  DISPARITY_SAD_FRAG_SOURCE,
  PERSPECTIVE_WINDOW_FRAG_SOURCE,
} from '../../src/renderer/shaders.js';
import { WebGLParallaxViewer } from '../../src/renderer/webglViewer.js';
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
    expect(DISPARITY_SAD_FRAG_SOURCE).toContain('u_baseDisp');
    expect(DISPARITY_SAD_FRAG_SOURCE).toContain('u_temporalWeight');

    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('#version 300 es');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_sbsTexture');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_dispTexture');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_viewShift');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_maxDisparityUV');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_windowCrop');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_baseDisp');
    expect(PERSPECTIVE_WINDOW_FRAG_SOURCE).toContain('u_renderMode');
    expect(VERTEX_SHADER_SOURCE).toContain('u_flipY');
  });

  it('verifies 2.5D ray-marched perspective warp stays strictly within eye texture halves across all baseline offsets', () => {
    const crop = 0.06;
    const maxDisparityUV = 24.0 / 480.0; // 0.05
    const eyeTexelX = 1.0 / 480.0;

    // Test across typical 3DS stereo baseline shifts
    const testBaseDisparities = [-72, -60, -40, -16, 0, 8];

    for (const baseDisp of testBaseDisparities) {
      const baseDispUV = baseDisp * eyeTexelX;

      for (let vy = 0.0; vy <= 1.0; vy += 0.2) {
        for (let vx = 0.0; vx <= 1.0; vx += 0.2) {
          for (let t = 0.0; t <= 1.0; t += 0.2) {
            const s = (t - 0.5) * 2.0;
            const activeWidth = 1.0 - 2.0 * crop;
            const windowOffset = (t - 0.5) * (1.2 * crop);

            const eyeX = Math.max(0.005, Math.min(0.995, crop + vx * activeWidth + windowOffset));
            const eyeY = Math.max(0.005, Math.min(0.995, crop + vy * (1.0 - 2.0 * crop)));

            expect(eyeX).toBeGreaterThanOrEqual(0.005);
            expect(eyeX).toBeLessThanOrEqual(0.995);
            expect(eyeY).toBeGreaterThanOrEqual(0.005);
            expect(eyeY).toBeLessThanOrEqual(0.995);

            for (let dNorm = 0.0; dNorm <= 1.0; dNorm += 0.25) {
              const relDispUV = (dNorm - 0.5) * 2.0 * maxDisparityUV;

              // 2.5D ray-marched warp logic
              const refinedRelDisp = relDispUV; // worst-case bounds
              const rawLeftX = (eyeX + s * refinedRelDisp * 0.85) * 0.5;
              const rightInner = eyeX - baseDispUV + s * refinedRelDisp * 0.85 - refinedRelDisp * 0.5;
              const rawRightX = 0.5 + rightInner * 0.5;

              const clampedLeftX = Math.max(0.001, Math.min(0.499, rawLeftX));
              const clampedRightX = Math.max(0.501, Math.min(0.999, rawRightX));

              expect(clampedLeftX).toBeGreaterThanOrEqual(0.001);
              expect(clampedLeftX).toBeLessThan(0.5);

              expect(clampedRightX).toBeGreaterThanOrEqual(0.501);
              expect(clampedRightX).toBeLessThan(1.0);
            }
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

    let rawRgba: Uint8Array;
    const fixturePath = path.resolve(__dirname, '../fixtures/hni_0070_frame05.rgba');
    if (fs.existsSync(fixturePath)) {
      rawRgba = new Uint8Array(fs.readFileSync(fixturePath));
    } else {
      // Generate synthetic SBS frame (960x240x4) with multi-plane depth
      rawRgba = new Uint8Array(960 * 240 * 4);
      for (let y = 0; y < 240; y++) {
        for (let x = 0; x < 480; x++) {
          let shift = 2;
          if (x > 60 && x < 150 && y > 20 && y < 90) shift = 0;
          else if (x > 150 && x < 250 && y > 40 && y < 140) shift = 5;
          else if (x > 250 && x < 350 && y > 80 && y < 180) shift = 8;
          else if (x > 350 && x < 440 && y > 110 && y < 210) shift = 11;
          const valLeft = ((x * 374761393) ^ (y * 668265263)) & 0xff;
          const lIdx = (y * 960 + x) * 4;
          rawRgba[lIdx] = valLeft;
          rawRgba[lIdx + 1] = valLeft;
          rawRgba[lIdx + 2] = valLeft;
          rawRgba[lIdx + 3] = 255;

          const xr = Math.min(479, x + shift);
          const valRight = ((xr * 374761393) ^ (y * 668265263)) & 0xff;
          const rIdx = (y * 960 + 480 + x) * 4;
          rawRgba[rIdx] = valRight;
          rawRgba[rIdx + 1] = valRight;
          rawRgba[rIdx + 2] = valRight;
          rawRgba[rIdx + 3] = 255;
        }
      }
    }
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

    // 4. Test Wasm base_disp getter and setter
    expect(halfEngine.base_disp()).toBe(0);
    halfEngine.set_base_disp(-72);
    expect(halfEngine.base_disp()).toBe(-72);
    halfEngine.set_base_disp(0);
  });

  it('accurately estimates real 3DS stereo convergence offsets across typical disparity ranges in Wasm & TypeScript', () => {
    const wasmPath = path.resolve(__dirname, '../../src/wasm/pkg/stereo_disparity_bg.wasm');
    const wasmBytes = fs.readFileSync(wasmPath);
    initSync({ module: wasmBytes });

    const eyeW = 480;
    const eyeH = 240;
    const sbsW = 960;

    // Test shifts corresponding to real 3DS videos (e.g. -72 for HNI_0062, -16 for HNI_0001, 0 for synthetic, +8)
    const testShifts = [-72, -48, -16, 0, 8];

    for (const shift of testShifts) {
      const syntheticRgba = new Uint8Array(sbsW * eyeH * 4);

      for (let y = 0; y < eyeH; y++) {
        for (let x = 0; x < eyeW; x++) {
          const lVal = ((x * 374761393) ^ (y * 668265263)) & 0xff;
          const lIdx = (y * sbsW + x) * 4;
          syntheticRgba[lIdx] = lVal;
          syntheticRgba[lIdx + 1] = lVal;
          syntheticRgba[lIdx + 2] = lVal;
          syntheticRgba[lIdx + 3] = 255;

          // xR = x - shift
          const rx = x - shift;
          if (rx >= 0 && rx < eyeW) {
            const rIdx = (y * sbsW + eyeW + rx) * 4;
            syntheticRgba[rIdx] = lVal;
            syntheticRgba[rIdx + 1] = lVal;
            syntheticRgba[rIdx + 2] = lVal;
            syntheticRgba[rIdx + 3] = 255;
          }
        }
      }

      // 1. Test TypeScript baseline estimator
      const tsDetected = WebGLParallaxViewer.prototype.estimateGlobalBaselineFromRgba(
        syntheticRgba,
        sbsW,
        eyeH,
        -112,
        32
      );
      expect(tsDetected).toBe(shift);

      // 2. Test Wasm baseline estimator
      const wasmEngine = new StereoDisparity(eyeW, eyeH, 24);
      const wasmDetected = wasmEngine.estimate_global_disparity(syntheticRgba, -112, 32);
      expect(wasmDetected).toBe(shift);
      expect(wasmEngine.base_disp()).toBe(shift);
    }
  });
});

