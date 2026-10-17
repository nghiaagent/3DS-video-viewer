import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import {
  calibrateDepthWithStereoSBS,
  expandDepthMap,
  resampleDepthMap,
  extractLeftEyeRgba,
  extractRightEyeRgba,
  computeZeroMeanSAD,
  DepthAnythingController,
  VideoDepthTimeline,
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

  it('DepthAnythingController supports modelVersion v3 and v2 switching and defaults', async () => {
    const controller = new DepthAnythingController({
      customInfer: async () => ({
        rawDepth: new Float32Array(100),
        backend: 'wasm',
        inferenceTimeMs: 1.0,
      }),
    });

    expect(controller.getModelVersion()).toBe('v3');
    await controller.init();
    expect(controller.isReady()).toBe(true);

    await controller.setModelVersion('v2');
    expect(controller.getModelVersion()).toBe('v2');

    await controller.setModelVersion('v3');
    expect(controller.getModelVersion()).toBe('v3');

    controller.destroy();
  });

  it('runs true ONNX inference with DepthAnythingV2 model in public/models/ at 504px if present', async () => {
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

    const { rawDepth, inferenceTimeMs } = await runDepthInference(session, testRgba, w, h, 504);
    expect(rawDepth.length).toBe(w * h);
    expect(inferenceTimeMs).toBeGreaterThan(0);

    // Check normalization bounds
    for (let i = 0; i < rawDepth.length; i++) {
      expect(rawDepth[i]).toBeGreaterThanOrEqual(0.0);
      expect(rawDepth[i]).toBeLessThanOrEqual(1.0);
    }
  });

  it('runs true ONNX inference with DepthAnythingV3 and DepthAnythingV2 at 504px and saves diagnostic heatmaps', async () => {
    const v2Path = path.resolve(__dirname, '../../public/models/depthanythingv2.onnx');
    const v3Path = path.resolve(__dirname, '../../public/models/depthanythingv3.onnx');
    const wasmPaths = path.resolve(__dirname, '../../node_modules/onnxruntime-web/dist') + '/';

    if (!fs.existsSync(v2Path) || !fs.existsSync(v3Path)) {
      console.log('Skipping dual V2/V3 test (models not found)');
      return;
    }

    // Try to load real HNI_0062.AVI frame or fallback to synthetic 480x240 frame
    let leftRgba: Uint8Array;
    let sbsRgba: Uint8Array;
    const w = 480;
    const h = 240;
    const rawLeftFile = '/tmp/hni_0062_left.raw';
    const rawSbsFile = '/tmp/hni_0062_sbs.raw';
    const realAviPath = '/Users/nghiaagent/Downloads/3DS vids/DCIM/101NIN03/HNI_0062.AVI';

    if (fs.existsSync(rawLeftFile) && fs.existsSync(rawSbsFile)) {
      leftRgba = new Uint8Array(fs.readFileSync(rawLeftFile));
      sbsRgba = new Uint8Array(fs.readFileSync(rawSbsFile));
    } else if (fs.existsSync(realAviPath)) {
      try {
        const { execSync } = await import('node:child_process');
        execSync(`ffmpeg -y -i "${realAviPath}" -vframes 1 -f rawvideo -pix_fmt rgba ${rawLeftFile} 2>/dev/null`);
        execSync(`ffmpeg -y -i "${realAviPath}" -filter_complex "[0:v:0][0:v:1]hstack" -vframes 1 -f rawvideo -pix_fmt rgba ${rawSbsFile} 2>/dev/null`);
        leftRgba = new Uint8Array(fs.readFileSync(rawLeftFile));
        sbsRgba = new Uint8Array(fs.readFileSync(rawSbsFile));
      } catch {
        const syn = createSyntheticSBSFrame(w, h, -70, 16, 4, -4);
        sbsRgba = syn.sbsRgba;
        leftRgba = extractLeftEyeRgba(sbsRgba, syn.sbsW, syn.sbsH);
      }
    } else {
      const syn = createSyntheticSBSFrame(w, h, -70, 16, 4, -4);
      sbsRgba = syn.sbsRgba;
      leftRgba = extractLeftEyeRgba(sbsRgba, syn.sbsW, syn.sbsH);
    }

    // Helper to save colormapped heatmap PNG
    const saveHeatmap = (depthUint8: Uint8Array, eyeW: number, eyeH: number, outPath: string) => {
      const ppmPath = outPath + '.ppm';
      const header = `P6\n${eyeW} ${eyeH}\n255\n`;
      const buf = Buffer.alloc(header.length + eyeW * eyeH * 3);
      buf.write(header, 0, 'ascii');
      let offset = header.length;

      for (let i = 0; i < depthUint8.length; i++) {
        const v = depthUint8[i] / 255.0;
        let r = 0, g = 0, b = 0;
        if (v < 0.25) {
          const t = v / 0.25;
          r = 0; g = Math.round(t * 255); b = 255;
        } else if (v < 0.5) {
          const t = (v - 0.25) / 0.25;
          r = 0; g = 255; b = Math.round((1 - t) * 255);
        } else if (v < 0.75) {
          const t = (v - 0.5) / 0.25;
          r = Math.round(t * 255); g = 255; b = 0;
        } else {
          const t = (v - 0.75) / 0.25;
          r = 255; g = Math.round((1 - t) * 255); b = 0;
        }
        buf[offset++] = r;
        buf[offset++] = g;
        buf[offset++] = b;
      }
      fs.writeFileSync(ppmPath, buf);
      try {
        const { execSync } = require('node:child_process');
        execSync(`ffmpeg -y -i "${ppmPath}" "${outPath}" 2>/dev/null`);
        try { fs.unlinkSync(ppmPath); } catch {}
      } catch {}
    };

    // 1. Run DepthAnythingV3 inference at 504px
    const { session: sessionV3 } = await initSession({
      modelPath: v3Path,
      wasmPaths,
      preferWebGPU: false,
    });
    const rightRgba = extractRightEyeRgba(sbsRgba, w * 2, h);

    // 1. Run DepthAnythingV3 multi-view stereo inference with both Left and Right views
    const inferV3 = await runDepthInference(sessionV3, leftRgba, w, h, 504, rightRgba);
    expect(inferV3.rawDepth.length).toBe(w * h);
    expect(inferV3.numViews).toBe(2);
    const calV3 = calibrateDepthWithStereoSBS(inferV3.rawDepth, sbsRgba, -70, 16, w * 2, h, {
      expandRadius: 1,
      rightDaDepth: inferV3.rawRightDepth,
      conf0: inferV3.conf0,
      conf1: inferV3.conf1,
    });
    expect(calV3.width).toBe(w);
    expect(calV3.height).toBe(h);
    expect(calV3.anchorCount).toBeGreaterThan(50);
    expect(calV3.sbsReconstructionMae).toBeLessThan(15.0);

    // Verify full contrast dynamic range [0, 255]
    let minV3 = 255, maxV3 = 0;
    for (let i = 0; i < calV3.calibratedDepth.length; i++) {
      const v = calV3.calibratedDepth[i];
      if (v < minV3) minV3 = v;
      if (v > maxV3) maxV3 = v;
    }
    expect(minV3).toBeLessThanOrEqual(25);
    expect(maxV3).toBeGreaterThanOrEqual(230);

    // 2. Run DepthAnythingV2 inference at 504px (monocular)
    const { session: sessionV2 } = await initSession({
      modelPath: v2Path,
      wasmPaths,
      preferWebGPU: false,
    });
    const inferV2 = await runDepthInference(sessionV2, leftRgba, w, h, 504);
    expect(inferV2.rawDepth.length).toBe(w * h);
    expect(inferV2.numViews).toBe(1);

    const calV2 = calibrateDepthWithStereoSBS(inferV2.rawDepth, sbsRgba, -70, 16, w * 2, h, { expandRadius: 1 });
    expect(calV2.width).toBe(w);
    expect(calV2.height).toBe(h);
    expect(calV2.sbsReconstructionMae).toBeLessThan(15.0);

    // 3. Save diagnostic heatmaps to scratch directory
    const scratchDir = '/Users/nghiaagent/.gemini/antigravity/brain/a683cc2e-6fb4-45b1-933b-9fcd06ec25f2/scratch';
    if (fs.existsSync(scratchDir)) {
      const v3Heatmap = path.join(scratchDir, 'depthanything_v3_heatmap.png');
      const v2Heatmap = path.join(scratchDir, 'depthanything_v2_heatmap.png');
      saveHeatmap(calV3.calibratedDepth, w, h, v3Heatmap);
      saveHeatmap(calV2.calibratedDepth, w, h, v2Heatmap);
      expect(fs.existsSync(v3Heatmap)).toBe(true);
      expect(fs.existsSync(v2Heatmap)).toBe(true);
    }
  }, 30000);

  it('runs dual-perspective (Left + Right eye) consensus depth inference on DepthAnythingV2', async () => {
    const v2Path = path.resolve(__dirname, '../../public/models/depthanythingv2.onnx');
    const wasmPaths = path.resolve(__dirname, '../../node_modules/onnxruntime-web/dist') + '/';

    if (!fs.existsSync(v2Path)) {
      console.log('Skipping dual-view V2 test (model not found)');
      return;
    }

    const { session: sessionV2 } = await initSession({
      modelPath: v2Path,
      wasmPaths,
      preferWebGPU: false,
    });

    let eyeW = 480;
    let eyeH = 240;
    let baseDisp = -70;
    let sbsRgba: Uint8Array;
    let leftRgba: Uint8Array;
    let rightRgba: Uint8Array;

    const rawSbsFile = '/tmp/hni_0062_sbs.raw';
    if (fs.existsSync(rawSbsFile)) {
      sbsRgba = new Uint8Array(fs.readFileSync(rawSbsFile));
      leftRgba = extractLeftEyeRgba(sbsRgba, eyeW * 2, eyeH);
      rightRgba = extractRightEyeRgba(sbsRgba, eyeW * 2, eyeH);
    } else {
      eyeW = 120;
      eyeH = 60;
      baseDisp = 0;
      const syn = createSyntheticSBSFrame(eyeW, eyeH, 0, 16, 6, -6);
      sbsRgba = syn.sbsRgba;
      leftRgba = extractLeftEyeRgba(sbsRgba, eyeW * 2, eyeH);
      rightRgba = extractRightEyeRgba(sbsRgba, eyeW * 2, eyeH);
    }

    // Pass both Left Eye and Right Eye to V2 at fast 336px
    const infer = await runDepthInference(sessionV2, leftRgba, eyeW, eyeH, 336, rightRgba);
    expect(infer.numViews).toBe(2);
    expect(infer.rawDepth).toBeDefined();
    expect(infer.rawDepth.length).toBe(eyeW * eyeH);
    expect(infer.rawRightDepth).toBeDefined();
    expect(infer.rawRightDepth!.length).toBe(eyeW * eyeH);

    // Calibrate with stereo consensus fusion
    const cal = calibrateDepthWithStereoSBS(infer.rawDepth, sbsRgba, baseDisp, 16, eyeW * 2, eyeH, {
      expandRadius: 1,
      rightDaDepth: infer.rawRightDepth,
    });

    expect(cal.width).toBe(eyeW);
    expect(cal.height).toBe(eyeH);
    expect(cal.calibratedDepth.length).toBe(eyeW * eyeH);
    if (fs.existsSync(rawSbsFile)) {
      expect(cal.sbsReconstructionMae).toBeLessThan(15.0);
    }
  });

  it('VideoDepthTimeline pre-computes frame-accurate dual-perspective consensus depth timelines for both V3 and V2', async () => {
    const eyeW = 60;
    const eyeH = 30;
    const sbsW = eyeW * 2;
    const sbsH = eyeH;
    const totalDuration = 2.0; // 2 seconds = 40 frames at 20 FPS
    const fps = 20;
    const totalFrames = 40;

    // Generate synthetic SBS frames for all 40 frames
    const mockFrames: Uint8Array[] = [];
    for (let f = 0; f < totalFrames; f++) {
      const shiftFg = 4 + (f % 3);
      const shiftBg = -4 - (f % 2);
      const syn = createSyntheticSBSFrame(eyeW, eyeH, 0, 16, shiftFg, shiftBg);
      mockFrames.push(syn.sbsRgba);
    }

    const progressReports: Array<{ stage: string; ratio: number }> = [];

    const timeline = new VideoDepthTimeline({
      fps,
      inputSize: 336,
      baseDisp: 0,
      maxDisp: 16,
      frameExtractor: async (_timeSec, frameIndex) => {
        return mockFrames[frameIndex] || mockFrames[0];
      },
      customInfer: async (_leftRgba, w, h, _size, rightRgba, modelVersion) => {
        // Mock dual-perspective neural pass returning depth arrays for left and right
        const leftDepth = new Float32Array(w * h);
        const rightDepth = rightRgba ? new Float32Array(w * h) : undefined;
        for (let i = 0; i < leftDepth.length; i++) {
          leftDepth[i] = modelVersion === 'v3' ? 0.6 : 0.5;
          if (rightDepth) rightDepth[i] = modelVersion === 'v3' ? 0.6 : 0.5;
        }
        return {
          rawDepth: leftDepth,
          rawRightDepth: rightDepth,
          backend: 'wasm',
          inferenceTimeMs: 2.0,
        };
      },
    });

    const mockVideo = {
      duration: totalDuration,
      videoWidth: sbsW,
      videoHeight: sbsH,
      currentTime: 0,
      readyState: 4,
    } as unknown as HTMLVideoElement;

    // 1. Build initial timeline with 5 keyframes
    await timeline.buildInitialTimeline(
      mockVideo,
      0,
      (stage, ratio) => {
        progressReports.push({ stage, ratio });
      },
      { initialKeyframes: 5, runFullStreamInBackground: true }
    );

    // Initial pre-pass progress was reported
    expect(progressReports.length).toBeGreaterThan(0);
    expect(timeline.isReady()).toBe(true);

    // Keyframes were computed for both V3 and V2
    expect(timeline.hasFrame(0, 'v3')).toBe(true);
    expect(timeline.hasFrame(0, 'v2')).toBe(true);

    // Exact keyframe query at t = 0.0s
    const depthAtZeroV3 = timeline.getDepthForTime(0.0, 'v3');
    expect(depthAtZeroV3).toBeDefined();
    expect(depthAtZeroV3!.isExact).toBe(true);
    expect(depthAtZeroV3!.frameIndex).toBe(0);
    expect(depthAtZeroV3!.depth.length).toBe(eyeW * eyeH);

    const depthAtZeroV2 = timeline.getDepthForTime(0.0, 'v2');
    expect(depthAtZeroV2).toBeDefined();
    expect(depthAtZeroV2!.isExact).toBe(true);

    // Intermediate frame query at t = 0.05s (frame 1) before background worker finishes
    // Temporal interpolation between keyframe 0 and keyframe 10 provides depth with zero desync!
    const depthInterpV3 = timeline.getDepthForTime(0.05, 'v3');
    expect(depthInterpV3).toBeDefined();
    expect(depthInterpV3!.depth.length).toBe(eyeW * eyeH);

    // 2. Wait for background worker to fill all 40 frames
    await timeline.waitForAllFrames();
    const progress = timeline.getProgress();
    expect(progress.isComplete).toBe(true);
    expect(progress.completedFrames).toBe(totalFrames);
    expect(progress.v3Frames).toBe(totalFrames);
    expect(progress.v2Frames).toBe(totalFrames);

    // Now frame 1 is exact!
    const depthExactV3 = timeline.getDepthForTime(0.05, 'v3');
    expect(depthExactV3!.isExact).toBe(true);

    // 3. Verify fast integration with DepthAnythingController
    let controllerCallbackDepth: Uint8Array | null = null;
    let controllerTelemetry: any = null;

    const controller = new DepthAnythingController({
      timeline,
      onDepthMapReady: (calDepth, w, h, telemetry) => {
        controllerCallbackDepth = calDepth;
        controllerTelemetry = telemetry;
        expect(w).toBe(eyeW);
        expect(h).toBe(eyeH);
      },
    });

    // Query controller with video element at currentTime = 0.5s (frame 10)
    (mockVideo as any).currentTime = 0.5;
    const handled = await controller.processVideoFrame(mockVideo);
    expect(handled).toBe(true);
    expect(controllerCallbackDepth).toBeDefined();
    expect(controllerTelemetry).toBeDefined();
    expect(controllerTelemetry.backend).toBe('timeline-cache');
    expect(controllerTelemetry.inferenceTimeMs).toBe(0); // 0.0ms instant timeline lookup!
    expect(controllerTelemetry.modelVersion).toBe('v3');

    // Switch model to v2 and query
    await controller.setModelVersion('v2');
    controllerCallbackDepth = null;
    const handledV2 = await controller.processVideoFrame(mockVideo);
    expect(handledV2).toBe(true);
    expect(controllerCallbackDepth).toBeDefined();
    expect(controllerTelemetry.modelVersion).toBe('v2');

    timeline.destroy();
    controller.destroy();
  });

  it('VideoDepthTimeline precomputeFullVideo synchronously computes all frames', async () => {
    const eyeW = 60;
    const eyeH = 30;
    const sbsW = eyeW * 2;
    const sbsH = eyeH;
    const totalFrames = 10;
    const fps = 20;

    const mockFrames: Uint8Array[] = [];
    for (let f = 0; f < totalFrames; f++) {
      const syn = createSyntheticSBSFrame(eyeW, eyeH, 0, 16, 4, -4);
      mockFrames.push(syn.sbsRgba);
    }

    const timeline = new VideoDepthTimeline({
      fps,
      inputSize: 336,
      frameExtractor: async (_timeSec, frameIndex) => mockFrames[frameIndex] || mockFrames[0],
      customInfer: async (_left, w, h) => ({
        rawDepth: new Float32Array(w * h).fill(0.5),
        rawRightDepth: new Float32Array(w * h).fill(0.5),
        backend: 'wasm',
        inferenceTimeMs: 1.0,
      }),
    });

    const mockVideo = {
      duration: 0.5, // 10 frames
      videoWidth: sbsW,
      videoHeight: sbsH,
      currentTime: 0,
      readyState: 4,
    } as unknown as HTMLVideoElement;

    await timeline.precomputeFullVideo(mockVideo, 0);

    const prog = timeline.getProgress();
    expect(prog.isComplete).toBe(true);
    expect(prog.completedFrames).toBe(totalFrames);

    for (let f = 0; f < totalFrames; f++) {
      expect(timeline.hasFrame(f, 'v3')).toBe(true);
      expect(timeline.hasFrame(f, 'v2')).toBe(true);
    }

    timeline.destroy();
  });
});

