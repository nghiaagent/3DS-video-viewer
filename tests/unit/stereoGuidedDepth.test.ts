import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import {
  calibrateDepthWithStereoSBS,
  expandDepthMap,
  resampleDepthMap,
  extractLeftEyeRgba,
  computeZeroMeanSAD,
  DepthAnythingController,
} from '../../src/depth/stereoGuidedDepth.js';
import {
  initSession,
  runDepthInference,
} from '../../src/depth/depthAnythingWorker.js';

/**
 * Helper to synthesize an SBS stereo frame with known disparity steps and rich texture.
 */
function createSyntheticSBSFrame(
  eyeW: number,
  eyeH: number,
  baseDisp: number,
  maxDisp: number,
  foregroundShift: number, // e.g. +6
  backgroundShift: number  // e.g. -6
): {
  sbsRgba: Uint8Array;
  sbsW: number;
  sbsH: number;
  trueDepthNorm: Float32Array;
} {
  const sbsW = eyeW * 2;
  const sbsH = eyeH;
  const sbsRgba = new Uint8Array(sbsW * sbsH * 4);
  const trueDepthNorm = new Float32Array(eyeW * eyeH);

  const dispMin = baseDisp - maxDisp;
  const dispSpan = 2 * maxDisp;

  const texture = (x: number, y: number) => {
    return ((x * 1234567) ^ (y * 7654321) ^ ((x >> 2) * 987654)) & 0xff;
  };

  for (let y = 0; y < eyeH; y++) {
    for (let x = 0; x < eyeW; x++) {
      const isForeground = x >= eyeW * 0.25 && x < eyeW * 0.75 && y >= eyeH * 0.25 && y < eyeH * 0.75;
      const trueDisp = baseDisp + (isForeground ? foregroundShift : backgroundShift);
      const dNorm = (trueDisp - dispMin) / dispSpan;
      trueDepthNorm[y * eyeW + x] = dNorm;

      const lVal = Math.min(240, Math.max(15, texture(x, y)));
      const lIdx = (y * sbsW + x) * 4;
      sbsRgba[lIdx] = lVal;
      sbsRgba[lIdx + 1] = lVal;
      sbsRgba[lIdx + 2] = lVal;
      sbsRgba[lIdx + 3] = 255;
    }

    for (let xr = 0; xr < eyeW; xr++) {
      const fgDisp = baseDisp + foregroundShift;
      const bgDisp = baseDisp + backgroundShift;

      const xFg = xr + fgDisp;
      const isFg = xFg >= eyeW * 0.25 && xFg < eyeW * 0.75 && y >= eyeH * 0.25 && y < eyeH * 0.75;
      const xScene = isFg ? xFg : xr + bgDisp;

      const rVal = Math.min(240, Math.max(15, texture(xScene, y)));
      const rIdx = (y * sbsW + eyeW + xr) * 4;
      sbsRgba[rIdx] = rVal;
      sbsRgba[rIdx + 1] = rVal;
      sbsRgba[rIdx + 2] = rVal;
      sbsRgba[rIdx + 3] = 255;
    }
  }

  return { sbsRgba, sbsW, sbsH, trueDepthNorm };
}

describe('stereoGuidedDepth & DepthAnything Integration', () => {
  it('extractLeftEyeRgba correctly extracts Left Eye half without Right Eye bleeding', () => {
    const sbsW = 8;
    const sbsH = 2;
    const sbsRgba = new Uint8Array(sbsW * sbsH * 4);

    // Left eye (x: 0..3) = 100, Right eye (x: 4..7) = 200
    for (let y = 0; y < sbsH; y++) {
      for (let x = 0; x < sbsW; x++) {
        const val = x < 4 ? 100 : 200;
        const idx = (y * sbsW + x) * 4;
        sbsRgba[idx] = val;
        sbsRgba[idx + 1] = val;
        sbsRgba[idx + 2] = val;
        sbsRgba[idx + 3] = 255;
      }
    }

    const leftEye = extractLeftEyeRgba(sbsRgba, sbsW, sbsH);
    expect(leftEye.length).toBe(4 * 2 * 4);
    for (let i = 0; i < leftEye.length; i += 4) {
      expect(leftEye[i]).toBe(100);
      expect(leftEye[i + 3]).toBe(255);
    }
  });

  it('resampleDepthMap smoothly interpolates Float32 depth maps', () => {
    const srcW = 2;
    const srcH = 2;
    const src = new Float32Array([0.0, 1.0, 0.0, 1.0]);

    const dst = resampleDepthMap(src, srcW, srcH, 4, 4);
    expect(dst.length).toBe(16);
    // Center pixels should be interpolated around 0.5
    expect(dst[5]).toBeGreaterThan(0.1);
    expect(dst[5]).toBeLessThan(0.9);
    // Values should stay bounded in [0, 1]
    for (let i = 0; i < dst.length; i++) {
      expect(dst[i]).toBeGreaterThanOrEqual(0.0);
      expect(dst[i]).toBeLessThanOrEqual(1.0);
    }
  });

  it('expandDepthMap spreads foreground (higher depth) values outward with radius', () => {
    const w = 5;
    const h = 5;
    const depth = new Uint8Array(w * h);
    // Center pixel is foreground (value 200)
    depth[2 * w + 2] = 200;

    const expanded = expandDepthMap(depth, w, h, 1);
    // 8-neighbors of center should now have 200
    expect(expanded[2 * w + 2]).toBe(200);
    expect(expanded[1 * w + 2]).toBe(200);
    expect(expanded[3 * w + 2]).toBe(200);
    expect(expanded[2 * w + 1]).toBe(200);
    expect(expanded[2 * w + 3]).toBe(200);
    // Corner pixels at radius 2 should still be 0
    expect(expanded[0]).toBe(0);
  });

  it('computeZeroMeanSAD achieves zero error at the true stereo disparity', () => {
    const w = 32;
    const h = 32;
    const leftLum = new Float32Array(w * h);
    const rightLum = new Float32Array(w * h);

    const shift = 4;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const val = ((x * 17) ^ (y * 31)) % 100;
        leftLum[y * w + x] = val;
        const xr = x - shift;
        if (xr >= 0 && xr < w) {
          rightLum[y * w + xr] = val;
        }
      }
    }

    // At true shift = 4, ZSAD should be 0
    const trueSad = computeZeroMeanSAD(leftLum, rightLum, w, h, 16, 16, shift);
    expect(trueSad).toBe(0);

    // At incorrect shift != 4, ZSAD should be positive
    const wrongSad = computeZeroMeanSAD(leftLum, rightLum, w, h, 16, 16, shift + 2);
    expect(wrongSad).toBeGreaterThan(0);
  });

  it('calibrateDepthWithStereoSBS accurately recovers true stereo disparity from scaled monocular depth', () => {
    const eyeW = 120;
    const eyeH = 60;
    const baseDisp = 0;
    const maxDisp = 16;
    const foregroundShift = 6;
    const backgroundShift = -6;

    const { sbsRgba, sbsW, sbsH, trueDepthNorm } = createSyntheticSBSFrame(
      eyeW,
      eyeH,
      baseDisp,
      maxDisp,
      foregroundShift,
      backgroundShift
    );

    // Monocular depth with arbitrary scale and offset: D_da = 0.5 * D_true + 0.2
    const daDepth = new Float32Array(eyeW * eyeH);
    for (let i = 0; i < daDepth.length; i++) {
      daDepth[i] = 0.5 * trueDepthNorm[i] + 0.2;
    }

    const result = calibrateDepthWithStereoSBS(
      daDepth,
      sbsRgba,
      baseDisp,
      maxDisp,
      sbsW,
      sbsH,
      { step: 2, gradThreshold: 10, expandRadius: 1 }
    );

    expect(result.width).toBe(eyeW);
    expect(result.height).toBe(eyeH);
    expect(result.anchorCount).toBeGreaterThan(10);

    // The scale factor should recover approx 2.0 (since 0.5 * 2.0 = 1.0)
    expect(result.scaleA).toBeGreaterThan(1.4);
    expect(result.scaleA).toBeLessThan(2.6);

    // Photometric MAE should be low because depth aligns to true SBS stereo
    expect(result.sbsReconstructionMae).toBeLessThan(15.0);

    // Compare uncalibrated MAE vs calibrated MAE:
    // If we warp with uncalibrated depth (scale=1, shift=0), MAE would be significantly higher
    const dispMin = baseDisp - maxDisp;
    const dispSpan = 2 * maxDisp;
    let uncalMaeSum = 0;
    let uncalCount = 0;
    for (let y = 0; y < eyeH; y++) {
      for (let x = 0; x < eyeW; x++) {
        const dUncal = daDepth[y * eyeW + x];
        const disp = dispMin + dUncal * dispSpan;
        const xr = Math.round(x - disp);
        if (xr >= 0 && xr < eyeW) {
          const lVal = sbsRgba[(y * sbsW + x) * 4];
          const rVal = sbsRgba[(y * sbsW + eyeW + xr) * 4];
          uncalMaeSum += Math.abs(lVal - rVal);
          uncalCount++;
        }
      }
    }
    const uncalMae = uncalMaeSum / uncalCount;
    expect(result.sbsReconstructionMae).toBeLessThan(uncalMae);
  });

  it('calibrateDepthWithStereoSBS detects inverted monocular depth and corrects sign', () => {
    const eyeW = 120;
    const eyeH = 60;
    const baseDisp = 0;
    const maxDisp = 16;
    const foregroundShift = 6;
    const backgroundShift = -6;

    const { sbsRgba, sbsW, sbsH, trueDepthNorm } = createSyntheticSBSFrame(
      eyeW,
      eyeH,
      baseDisp,
      maxDisp,
      foregroundShift,
      backgroundShift
    );

    // Completely inverted monocular depth: D_da = 1.0 - D_true
    const invertedDaDepth = new Float32Array(eyeW * eyeH);
    for (let i = 0; i < invertedDaDepth.length; i++) {
      invertedDaDepth[i] = 1.0 - trueDepthNorm[i];
    }

    const result = calibrateDepthWithStereoSBS(
      invertedDaDepth,
      sbsRgba,
      baseDisp,
      maxDisp,
      sbsW,
      sbsH,
      { step: 2, gradThreshold: 10, expandRadius: 1 }
    );

    // Scale A should be negative relative to raw inverted input, indicating correct inversion
    expect(result.scaleA).toBeLessThan(0);
    // Photometric MAE should be low
    expect(result.sbsReconstructionMae).toBeLessThan(15.0);
  });

  it('handles negative 3DS baseline shifts (-70) matching MakerNote calibration', () => {
    const eyeW = 120;
    const eyeH = 60;
    const baseDisp = -70; // real 3DS video shift
    const maxDisp = 16;
    const foregroundShift = 4;
    const backgroundShift = -4;

    const { sbsRgba, sbsW, sbsH, trueDepthNorm } = createSyntheticSBSFrame(
      eyeW,
      eyeH,
      baseDisp,
      maxDisp,
      foregroundShift,
      backgroundShift
    );

    const daDepth = new Float32Array(trueDepthNorm);
    const result = calibrateDepthWithStereoSBS(
      daDepth,
      sbsRgba,
      baseDisp,
      maxDisp,
      sbsW,
      sbsH,
      { step: 2, gradThreshold: 10, expandRadius: 1 }
    );

    expect(result.scaleA).toBeGreaterThan(0.7);
    expect(result.scaleA).toBeLessThan(1.3);
    expect(result.sbsReconstructionMae).toBeLessThan(15.0);
  });

  it('DepthAnythingController manages non-blocking busy state and invokes callback', async () => {
    let callbackInvoked = false;
    let receivedTelemetry: any = null;

    const eyeW = 60;
    const eyeH = 30;
    const sbsW = 120;
    const sbsH = 30;
    const { sbsRgba, trueDepthNorm } = createSyntheticSBSFrame(eyeW, eyeH, 0, 16, 4, -4);

    const controller = new DepthAnythingController({
      baseDisp: 0,
      maxDisp: 16,
      inputSize: 128,
      onDepthMapReady: (_depth, w, h, telemetry) => {
        callbackInvoked = true;
        receivedTelemetry = telemetry;
        expect(w).toBe(eyeW);
        expect(h).toBe(eyeH);
      },
      customInfer: async (_rgba, _w, _h) => {
        // Mock worker returning normalized depth
        return {
          rawDepth: new Float32Array(trueDepthNorm),
          backend: 'wasm',
          inferenceTimeMs: 12.5,
        };
      },
    });

    await controller.init();
    expect(controller.isReady()).toBe(true);
    expect(controller.isBusy()).toBe(false);

    // 1. Submit first frame -> accepted
    const accepted1 = controller.processFrame(sbsRgba, sbsW, sbsH);
    expect(accepted1).toBe(true);
    expect(controller.isBusy()).toBe(true);

    // 2. Submit second frame while busy -> rejected immediately to protect 60 FPS render loop
    const accepted2 = controller.processFrame(sbsRgba, sbsW, sbsH);
    expect(accepted2).toBe(false);

    // Allow promise tick to complete customInfer
    await new Promise((resolve) => setTimeout(resolve, 50));

    expect(callbackInvoked).toBe(true);
    expect(controller.isBusy()).toBe(false);
    expect(receivedTelemetry).toBeDefined();
    expect(receivedTelemetry.inferenceTimeMs).toBe(12.5);
    expect(receivedTelemetry.sbsReconstructionMae).toBeLessThan(15.0);

    controller.destroy();
  });

  it('runs true ONNX inference with DepthAnythingV2 model in public/models/ if present', async () => {
    const modelPath = path.resolve(__dirname, '../../public/models/depthanythingv2.onnx');
    if (!fs.existsSync(modelPath)) {
      console.log('Skipping real ONNX model test (model file not found)');
      return;
    }

    const { session, backend } = await initSession({
      modelPath,
      wasmPaths: path.resolve(__dirname, '../../node_modules/onnxruntime-web/dist') + '/',
      preferWebGPU: false,
    });

    expect(session).toBeDefined();
    expect(backend).toBe('wasm');

    // Synthetic 120x60 Left Eye image
    const w = 120;
    const h = 60;
    const testRgba = new Uint8Array(w * h * 4);
    for (let i = 0; i < testRgba.length; i += 4) {
      testRgba[i] = 128;
      testRgba[i + 1] = 128;
      testRgba[i + 2] = 128;
      testRgba[i + 3] = 255;
    }

    const { rawDepth, inferenceTimeMs } = await runDepthInference(session, testRgba, w, h, 256);
    expect(rawDepth.length).toBe(w * h);
    expect(inferenceTimeMs).toBeGreaterThan(0);

    // Check normalization bounds
    for (let i = 0; i < rawDepth.length; i++) {
      expect(rawDepth[i]).toBeGreaterThanOrEqual(0.0);
      expect(rawDepth[i]).toBeLessThanOrEqual(1.0);
    }
  });
});
