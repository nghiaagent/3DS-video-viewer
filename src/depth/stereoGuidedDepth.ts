import type { WorkerDepthResultMessage, WorkerInitMessage, WorkerInferMessage, WorkerOutMessage } from './depthAnythingWorker.js';

export interface CalibrateDepthOptions {
  sbsWidth?: number;
  sbsHeight?: number;
  baseDisp?: number;
  maxDisp?: number;
  gradThreshold?: number;
  step?: number;
  expandRadius?: number;
  alphaConfidence?: number;
  contrastStretch?: boolean;
  rightDaDepth?: Float32Array;
  conf0?: Float32Array;
  conf1?: Float32Array;
}

export interface CalibrateDepthResult {
  calibratedDepth: Uint8Array;
  scaleA: number;
  shiftB: number;
  sbsReconstructionMae: number;
  width: number;
  height: number;
  anchorCount: number;
}

export interface DepthTelemetry {
  inferenceTimeMs: number;
  totalTimeMs: number;
  scaleA: number;
  shiftB: number;
  sbsReconstructionMae: number;
  backend: string;
  fps: number;
  frameId: number;
  anchorCount?: number;
  modelVersion?: 'v3' | 'v2';
  numViews?: number;
}

export type DepthMapReadyCallback = (
  calibratedDepth: Uint8Array,
  width: number,
  height: number,
  telemetry: DepthTelemetry
) => void;

export interface DepthAnythingControllerOptions {
  workerUrl?: string | URL;
  worker?: Worker;
  wasmPaths?: Record<string, string> | string;
  modelPath?: string;
  modelPathWebGPU?: string;
  modelVersion?: 'v3' | 'v2';
  inputSize?: number; // default 504 (multiple of 14: 36 * 14)
  baseDisp?: number;
  maxDisp?: number;
  expandRadius?: number;
  dilateRadius?: number;
  gradThreshold?: number;
  step?: number;
  contrastStretch?: boolean;
  onDepthMapReady?: DepthMapReadyCallback;
  customInfer?: (
    leftRgba: Uint8Array,
    width: number,
    height: number,
    size: number,
    rightRgba?: Uint8Array
  ) => Promise<{ rawDepth: Float32Array; backend: 'webgpu' | 'wasm'; inferenceTimeMs: number }>;
}

/**
 * Extracts Left Eye RGBA buffer from an SBS RGBA frame.
 * For an SBS frame with total sbsWidth, Left eye is [0, sbsWidth/2).
 */
export function extractLeftEyeRgba(
  sbsRgba: Uint8Array,
  sbsWidth: number,
  sbsHeight: number
): Uint8Array {
  const eyeWidth = Math.floor(sbsWidth / 2);
  const leftRgba = new Uint8Array(eyeWidth * sbsHeight * 4);

  for (let y = 0; y < sbsHeight; y++) {
    const sbsRowStart = y * sbsWidth * 4;
    const eyeRowStart = y * eyeWidth * 4;
    const eyeRowBytes = eyeWidth * 4;
    leftRgba.set(sbsRgba.subarray(sbsRowStart, sbsRowStart + eyeRowBytes), eyeRowStart);
  }

  return leftRgba;
}

/**
 * Extracts Right Eye RGBA buffer from an SBS RGBA frame.
 * For an SBS frame with total sbsWidth, Right eye is [sbsWidth/2, sbsWidth).
 */
export function extractRightEyeRgba(
  sbsRgba: Uint8Array,
  sbsWidth: number,
  sbsHeight: number
): Uint8Array {
  const eyeWidth = Math.floor(sbsWidth / 2);
  const rightRgba = new Uint8Array(eyeWidth * sbsHeight * 4);

  for (let y = 0; y < sbsHeight; y++) {
    const sbsRowStart = (y * sbsWidth + eyeWidth) * 4;
    const eyeRowStart = y * eyeWidth * 4;
    const eyeRowBytes = eyeWidth * 4;
    rightRgba.set(sbsRgba.subarray(sbsRowStart, sbsRowStart + eyeRowBytes), eyeRowStart);
  }

  return rightRgba;
}

/**
 * Bilinearly resamples a 1-channel Float32Array depth map.
 */
export function resampleDepthMap(
  srcDepth: Float32Array,
  srcW: number,
  srcH: number,
  dstW: number,
  dstH: number
): Float32Array {
  if (srcW === dstW && srcH === dstH) {
    return new Float32Array(srcDepth);
  }

  const dst = new Float32Array(dstW * dstH);
  const scaleX = srcW / dstW;
  const scaleY = srcH / dstH;

  for (let y = 0; y < dstH; y++) {
    const srcY = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.max(0, Math.min(srcH - 1, Math.floor(srcY)));
    const y1 = Math.min(srcH - 1, y0 + 1);
    const fy = Math.max(0, Math.min(1, srcY - y0));
    const rowOffset = y * dstW;

    for (let x = 0; x < dstW; x++) {
      const srcX = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.max(0, Math.min(srcW - 1, Math.floor(srcX)));
      const x1 = Math.min(srcW - 1, x0 + 1);
      const fx = Math.max(0, Math.min(1, srcX - x0));

      const d00 = srcDepth[y0 * srcW + x0];
      const d01 = srcDepth[y0 * srcW + x1];
      const d10 = srcDepth[y1 * srcW + x0];
      const d11 = srcDepth[y1 * srcW + x1];

      const d0 = d00 * (1 - fx) + d01 * fx;
      const d1 = d10 * (1 - fx) + d11 * fx;
      dst[rowOffset + x] = d0 * (1 - fy) + d1 * fy;
    }
  }

  return dst;
}

/**
 * Port of Tiefling's expandDepthMap foreground morphological dilation.
 * Propagates higher depth (foreground) values to neighboring pixels with lower depth values.
 */
export function expandDepthMap(
  depth: Uint8Array,
  width: number,
  height: number,
  radius: number = 2
): Uint8Array {
  if (radius <= 0) {
    return new Uint8Array(depth);
  }

  let src = new Uint8Array(depth);
  let dst = new Uint8Array(src);

  for (let r = 0; r < radius; r++) {
    for (let y = 1; y < height - 1; y++) {
      const rowOffset = y * width;
      for (let x = 1; x < width - 1; x++) {
        const currentDepth = src[rowOffset + x];
        if (currentDepth < 10) continue;

        for (let dy = -1; dy <= 1; dy++) {
          const nRowOffset = (y + dy) * width;
          for (let dx = -1; dx <= 1; dx++) {
            const nIdx = nRowOffset + (x + dx);
            if (dst[nIdx] < currentDepth) {
              dst[nIdx] = currentDepth;
            }
          }
        }
      }
    }
    src.set(dst);
  }

  return dst;
}

/**
 * Computes 5x5 Zero-Mean Sum of Absolute Differences (ZSAD) between Left eye (x, y)
 * and Right eye (x - d, y).
 */
export function computeZeroMeanSAD(
  leftLum: Float32Array,
  rightLum: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
  d: number,
  leftMeanCached?: number
): number | null {
  const xr = x - d;
  if (x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return null;
  if (xr < 2 || xr >= width - 2) return null;

  let meanL = leftMeanCached;
  if (meanL === undefined) {
    let sumL = 0;
    for (let dy = -2; dy <= 2; dy++) {
      const rowOffset = (y + dy) * width;
      for (let dx = -2; dx <= 2; dx++) {
        sumL += leftLum[rowOffset + (x + dx)];
      }
    }
    meanL = sumL / 25.0;
  }

  let sumR = 0;
  for (let dy = -2; dy <= 2; dy++) {
    const rowOffset = (y + dy) * width;
    for (let dx = -2; dx <= 2; dx++) {
      sumR += rightLum[rowOffset + (xr + dx)];
    }
  }
  const meanR = sumR / 25.0;

  let zsad = 0;
  for (let dy = -2; dy <= 2; dy++) {
    const rowOffset = (y + dy) * width;
    for (let dx = -2; dx <= 2; dx++) {
      const valL = leftLum[rowOffset + (x + dx)] - meanL;
      const valR = rightLum[rowOffset + (xr + dx)] - meanR;
      zsad += Math.abs(valL - valR);
    }
  }

  return zsad;
}

/**
 * SBS Stereo Similarity Calibrator.
 * Fuses smooth monocular DepthAnythingV2 relative depth with true 3DS SBS binocular disparity
 * via high-confidence edge anchors, weighted least-squares affine scale/shift calibration,
 * photometric polarity verification, and local residual refinement.
 */
export function calibrateDepthWithStereoSBS(
  daDepth: Float32Array,
  sbsRgba: Uint8Array,
  baseDispOrOptions: number | CalibrateDepthOptions = 0,
  maxDispArg: number = 16,
  sbsWidthArg?: number,
  sbsHeightArg?: number,
  extraOptions?: CalibrateDepthOptions
): CalibrateDepthResult {
  // Normalize parameters
  let baseDisp = 0;
  let maxDisp = maxDispArg;
  let sbsWidth = sbsWidthArg;
  let sbsHeight = sbsHeightArg;
  let gradThreshold = 12.0;
  let step = 2;
  let expandRadius = 1;
  let alphaConfidence = 0.0;
  let contrastStretch = true;
  let rightDaDepth: Float32Array | undefined;
  let conf0: Float32Array | undefined;
  let conf1: Float32Array | undefined;

  if (typeof baseDispOrOptions === 'object') {
    const opts = baseDispOrOptions;
    if (opts.baseDisp !== undefined) baseDisp = opts.baseDisp;
    if (opts.maxDisp !== undefined) maxDisp = opts.maxDisp;
    if (opts.sbsWidth !== undefined) sbsWidth = opts.sbsWidth;
    if (opts.sbsHeight !== undefined) sbsHeight = opts.sbsHeight;
    if (opts.gradThreshold !== undefined) gradThreshold = opts.gradThreshold;
    if (opts.step !== undefined) step = opts.step;
    if (opts.expandRadius !== undefined) expandRadius = opts.expandRadius;
    if (opts.alphaConfidence !== undefined) alphaConfidence = opts.alphaConfidence;
    if (opts.contrastStretch !== undefined) contrastStretch = opts.contrastStretch;
    if (opts.rightDaDepth !== undefined) rightDaDepth = opts.rightDaDepth;
    if (opts.conf0 !== undefined) conf0 = opts.conf0;
    if (opts.conf1 !== undefined) conf1 = opts.conf1;
  } else {
    baseDisp = baseDispOrOptions;
    if (extraOptions) {
      if (extraOptions.gradThreshold !== undefined) gradThreshold = extraOptions.gradThreshold;
      if (extraOptions.step !== undefined) step = extraOptions.step;
      if (extraOptions.expandRadius !== undefined) expandRadius = extraOptions.expandRadius;
      if (extraOptions.alphaConfidence !== undefined) alphaConfidence = extraOptions.alphaConfidence;
      if (extraOptions.contrastStretch !== undefined) contrastStretch = extraOptions.contrastStretch;
      if (extraOptions.rightDaDepth !== undefined) rightDaDepth = extraOptions.rightDaDepth;
      if (extraOptions.conf0 !== undefined) conf0 = extraOptions.conf0;
      if (extraOptions.conf1 !== undefined) conf1 = extraOptions.conf1;
    }
  }
  void alphaConfidence;

  // Infer dimensions if not specified
  if (!sbsWidth || !sbsHeight) {
    if (sbsRgba.length === 960 * 240 * 4) {
      sbsWidth = 960;
      sbsHeight = 240;
    } else {
      // Default to 2:1 SBS aspect ratio
      const totalPixels = Math.floor(sbsRgba.length / 4);
      sbsHeight = Math.round(Math.sqrt(totalPixels / 2));
      sbsWidth = Math.floor(totalPixels / sbsHeight);
    }
  }

  const eyeWidth = Math.floor(sbsWidth / 2);
  const eyeHeight = sbsHeight;

  // Ensure daDepth matches eyeWidth x eyeHeight
  let leftDaDepth = daDepth;
  if (daDepth.length !== eyeWidth * eyeHeight) {
    const sqrtLen = Math.round(Math.sqrt(daDepth.length));
    const srcW = sqrtLen * sqrtLen === daDepth.length ? sqrtLen : eyeWidth;
    const srcH = Math.floor(daDepth.length / srcW);
    leftDaDepth = resampleDepthMap(daDepth, srcW, srcH, eyeWidth, eyeHeight);
  }

  // Precompute grayscale luminance for Left and Right eye
  const leftLum = new Float32Array(eyeWidth * eyeHeight);
  const rightLum = new Float32Array(eyeWidth * eyeHeight);

  for (let y = 0; y < eyeHeight; y++) {
    const sbsRowStart = y * sbsWidth * 4;
    const eyeRowOffset = y * eyeWidth;

    for (let x = 0; x < eyeWidth; x++) {
      const lIdx = sbsRowStart + x * 4;
      leftLum[eyeRowOffset + x] =
        0.299 * sbsRgba[lIdx] + 0.587 * sbsRgba[lIdx + 1] + 0.114 * sbsRgba[lIdx + 2];

      const rIdx = sbsRowStart + (eyeWidth + x) * 4;
      rightLum[eyeRowOffset + x] =
        0.299 * sbsRgba[rIdx] + 0.587 * sbsRgba[rIdx + 1] + 0.114 * sbsRgba[rIdx + 2];
    }
  }

  // --------------------------------------------------------------------------
  // STEP A: High-Confidence Stereo Anchors
  // --------------------------------------------------------------------------
  const anchorNormDisp = new Float32Array(eyeWidth * eyeHeight);
  anchorNormDisp.fill(-1);
  const anchorConf = new Float32Array(eyeWidth * eyeHeight);

  const anchorX: number[] = [];
  const anchorY: number[] = [];
  const anchorDa: number[] = [];
  const anchorNorm: number[] = [];
  const anchorWeights: number[] = [];

  const dispMin = Math.round(baseDisp - maxDisp);
  const dispMax = Math.round(baseDisp + maxDisp);
  const dispSpan = dispMax - dispMin;
  const invDispSpan = dispSpan > 0 ? 1.0 / dispSpan : 1.0;

  for (let y = 2; y < eyeHeight - 2; y += step) {
    const rowOffset = y * eyeWidth;
    for (let x = 2; x < eyeWidth - 2; x += step) {
      // Horizontal and vertical gradients on Left Eye
      const gx = Math.abs(leftLum[rowOffset + (x + 1)] - leftLum[rowOffset + (x - 1)]) * 0.5;
      const gy = Math.abs(leftLum[(y + 1) * eyeWidth + x] - leftLum[(y - 1) * eyeWidth + x]) * 0.5;
      const grad = gx + gy;

      if (grad < gradThreshold) continue;

      // Calculate Left 5x5 window mean
      let sumL = 0;
      for (let dy = -2; dy <= 2; dy++) {
        const wy = (y + dy) * eyeWidth;
        for (let dx = -2; dx <= 2; dx++) {
          sumL += leftLum[wy + (x + dx)];
        }
      }
      const meanL = sumL / 25.0;

      let bestSAD = Infinity;
      let secondBestSAD = Infinity;
      let bestD = baseDisp;

      for (let d = dispMin; d <= dispMax; d++) {
        const xr = x - d;
        if (xr < 2 || xr >= eyeWidth - 2) continue;

        let sumR = 0;
        for (let dy = -2; dy <= 2; dy++) {
          const wy = (y + dy) * eyeWidth;
          for (let dx = -2; dx <= 2; dx++) {
            sumR += rightLum[wy + (xr + dx)];
          }
        }
        const meanR = sumR / 25.0;

        let zsad = 0;
        for (let dy = -2; dy <= 2; dy++) {
          const wy = (y + dy) * eyeWidth;
          for (let dx = -2; dx <= 2; dx++) {
            const vL = leftLum[wy + (x + dx)] - meanL;
            const vR = rightLum[wy + (xr + dx)] - meanR;
            zsad += Math.abs(vL - vR);
          }
        }

        if (zsad < bestSAD) {
          secondBestSAD = bestSAD;
          bestSAD = zsad;
          bestD = d;
        } else if (zsad < secondBestSAD) {
          secondBestSAD = zsad;
        }
      }

      if (bestSAD < Infinity) {
        const dNorm = Math.max(0, Math.min(1, (bestD - dispMin) * invDispSpan));
        const gradWeight = Math.min(1.0, (grad - gradThreshold) / 30.0);
        const sadWeight = 250.0 / (250.0 + bestSAD);
        const pkrn = secondBestSAD > bestSAD ? (secondBestSAD - bestSAD) / (bestSAD + 1.0) : 0;
        const conf = Math.max(0.01, Math.min(1.0, gradWeight * sadWeight * (0.5 + 0.5 * Math.min(1.0, pkrn))));

        const idx = rowOffset + x;
        anchorNormDisp[idx] = dNorm;
        anchorConf[idx] = conf;

        anchorX.push(x);
        anchorY.push(y);
        anchorDa.push(leftDaDepth[idx]);
        anchorNorm.push(dNorm);
        anchorWeights.push(conf);
      }
    }
  }

  // --------------------------------------------------------------------------
  // STEP B: Weighted Least-Squares Affine Fit + Photometric Polarity Verification
  // --------------------------------------------------------------------------
  const numAnchors = anchorDa.length;
  let scaleA = 1.0;
  let shiftB = 0.0;

  if (numAnchors >= 3) {
    // Helper to fit (a, b) for given monocular u values
    const fitAffine = (uArr: number[], vArr: number[], wArr: number[]) => {
      let sw = 0;
      let su = 0;
      let sv = 0;
      for (let i = 0; i < uArr.length; i++) {
        const w = wArr[i];
        sw += w;
        su += w * uArr[i];
        sv += w * vArr[i];
      }
      if (sw <= 1e-6) return { a: 1.0, b: 0.0 };

      const uMean = su / sw;
      const vMean = sv / sw;

      let cov = 0;
      let varU = 0;
      for (let i = 0; i < uArr.length; i++) {
        const w = wArr[i];
        const du = uArr[i] - uMean;
        const dv = vArr[i] - vMean;
        cov += w * du * dv;
        varU += w * du * du;
      }

      if (varU <= 1e-6) return { a: 1.0, b: vMean - uMean };
      let a = cov / varU;
      let b = vMean - a * uMean;

      // Robust iterative reweighted least squares (IRLS) to reject boundary/occlusion outliers
      for (let iter = 0; iter < 2; iter++) {
        let rSw = 0;
        let rSu = 0;
        let rSv = 0;
        for (let i = 0; i < uArr.length; i++) {
          const res = Math.abs(vArr[i] - (a * uArr[i] + b));
          const huberW = wArr[i] / (1.0 + 100.0 * res * res);
          rSw += huberW;
          rSu += huberW * uArr[i];
          rSv += huberW * vArr[i];
        }
        if (rSw > 1e-6) {
          const rUMean = rSu / rSw;
          const rVMean = rSv / rSw;
          let rCov = 0;
          let rVarU = 0;
          for (let i = 0; i < uArr.length; i++) {
            const res = Math.abs(vArr[i] - (a * uArr[i] + b));
            const huberW = wArr[i] / (1.0 + 100.0 * res * res);
            rCov += huberW * (uArr[i] - rUMean) * (vArr[i] - rVMean);
            rVarU += huberW * (uArr[i] - rUMean) ** 2;
          }
          if (rVarU > 1e-6) {
            a = rCov / rVarU;
            b = rVMean - a * rUMean;
          }
        }
      }

      return { a, b };
    };

    // Candidate 1: Direct (+D_DA)
    const fit1 = fitAffine(anchorDa, anchorNorm, anchorWeights);
    // Candidate 2: Inverted (1 - D_DA)
    const invDa = anchorDa.map((u) => 1.0 - u);
    const fit2 = fitAffine(invDa, anchorNorm, anchorWeights);

    // Evaluate photometric error on anchors for both fits to determine physical polarity
    let err1 = 0;
    let err2 = 0;
    let validCount1 = 0;
    let validCount2 = 0;

    for (let i = 0; i < numAnchors; i++) {
      const ax = anchorX[i];
      const ay = anchorY[i];
      const lVal = leftLum[ay * eyeWidth + ax];

      // Predict candidate 1 disparity
      const dPred1Norm = Math.max(0, Math.min(1, fit1.a * anchorDa[i] + fit1.b));
      const disp1 = dispMin + dPred1Norm * dispSpan;
      const xr1 = Math.round(ax - disp1);
      if (xr1 >= 0 && xr1 < eyeWidth) {
        err1 += anchorWeights[i] * Math.abs(lVal - rightLum[ay * eyeWidth + xr1]);
        validCount1 += anchorWeights[i];
      }

      // Predict candidate 2 disparity
      const dPred2Norm = Math.max(0, Math.min(1, fit2.a * (1.0 - anchorDa[i]) + fit2.b));
      const disp2 = dispMin + dPred2Norm * dispSpan;
      const xr2 = Math.round(ax - disp2);
      if (xr2 >= 0 && xr2 < eyeWidth) {
        err2 += anchorWeights[i] * Math.abs(lVal - rightLum[ay * eyeWidth + xr2]);
        validCount2 += anchorWeights[i];
      }
    }

    const normErr1 = validCount1 > 0 ? err1 / validCount1 : Infinity;
    const normErr2 = validCount2 > 0 ? err2 / validCount2 : Infinity;

    if (normErr2 < normErr1) {
      // Inverted depth yielded lower photometric reconstruction error!
      // a2 * (1 - D) + b2 = -a2 * D + (a2 + b2)
      scaleA = -fit2.a;
      shiftB = fit2.a + fit2.b;
    } else {
      scaleA = fit1.a;
      shiftB = fit1.b;
    }
  }

  // --------------------------------------------------------------------------
  // STEP C: Calibrated Depth Mapping & Percentile Contrast Stretching
  // Multi-view stereo consensus fusion if right eye depth is present
  // --------------------------------------------------------------------------
  const calibratedFloat = new Float32Array(eyeWidth * eyeHeight);
  for (let y = 0; y < eyeHeight; y++) {
    const rowOffset = y * eyeWidth;
    for (let x = 0; x < eyeWidth; x++) {
      const idx = rowOffset + x;
      const d0 = leftDaDepth[idx];

      if (rightDaDepth) {
        // Physical stereo disparity from calibrated left depth
        const d0CalNorm = Math.max(0, Math.min(1, scaleA * d0 + shiftB));
        const disp = dispMin + d0CalNorm * dispSpan;
        const xr = Math.max(0, Math.min(eyeWidth - 1, Math.round(x - disp)));
        const d1 = rightDaDepth[rowOffset + xr];

        const c0 = conf0 ? Math.max(1e-4, conf0[idx]) : 1.0;
        const c1 = conf1 ? Math.max(1e-4, conf1[rowOffset + xr]) : 1.0;

        // Where both views agree in 3D space, weight by confidence.
        // In half-occluded areas, exp(-diff * 6.0) falls to 0, smoothly preserving View 0.
        const diff = Math.abs(d0 - d1);
        const w0 = c0;
        const w1 = c1 * Math.exp(-diff * 6.0);
        const dConsensus = (w0 * d0 + w1 * d1) / (w0 + w1);

        calibratedFloat[idx] = scaleA * dConsensus + shiftB;
      } else {
        calibratedFloat[idx] = scaleA * d0 + shiftB;
      }
    }
  }

  // Compute robust percentiles (p2 and p98) to stretch depth map to full [0, 255] dynamic range
  let p2 = 0.0;
  let p98 = 1.0;
  if (contrastStretch && calibratedFloat.length > 0) {
    const sorted = Float32Array.from(calibratedFloat).sort();
    p2 = sorted[Math.floor(sorted.length * 0.02)];
    p98 = sorted[Math.floor(sorted.length * 0.98)];
  }
  const range = p98 - p2;
  const invRange = range > 1e-5 ? 1.0 / range : 1.0;

  const refinedDepthUint8 = new Uint8Array(eyeWidth * eyeHeight);
  for (let i = 0; i < calibratedFloat.length; i++) {
    const norm = contrastStretch
      ? Math.max(0, Math.min(1, (calibratedFloat[i] - p2) * invRange))
      : Math.max(0, Math.min(1, calibratedFloat[i]));
    refinedDepthUint8[i] = Math.round(norm * 255);
  }

  // Foreground dilation (Tiefling expandDepthMap) with clean default radius = 1
  const finalCalibratedDepth =
    expandRadius > 0
      ? expandDepthMap(refinedDepthUint8, eyeWidth, eyeHeight, expandRadius)
      : refinedDepthUint8;

  // Compute SBS Photometric Reconstruction MAE from physical calibrated disparity
  let sumMae = 0;
  let countMae = 0;

  for (let y = 0; y < eyeHeight; y++) {
    const rowOffset = y * eyeWidth;
    for (let x = 0; x < eyeWidth; x++) {
      const idx = rowOffset + x;
      const dNorm = Math.max(0, Math.min(1, calibratedFloat[idx]));
      const disp = dispMin + dNorm * dispSpan;
      const xr = Math.round(x - disp);

      if (xr >= 0 && xr < eyeWidth) {
        sumMae += Math.abs(leftLum[idx] - rightLum[rowOffset + xr]);
        countMae++;
      }
    }
  }

  const sbsReconstructionMae = countMae > 0 ? sumMae / countMae : 0.0;

  return {
    calibratedDepth: finalCalibratedDepth,
    scaleA,
    shiftB,
    sbsReconstructionMae,
    width: eyeWidth,
    height: eyeHeight,
    anchorCount: numAnchors,
  };
}

/**
 * Controller orchestrating DepthAnythingV2 Web Worker inference and SBS photogrammetric stereo calibration.
 * Non-blocking isBusy flag ensures the 60 FPS WebGL render loop never stalls.
 */
export class DepthAnythingController {
  public lastTelemetry: DepthTelemetry | null = null;
  private worker: Worker | null = null;
  private busy: boolean = false;
  private baseDisp: number = 0;
  private maxDisp: number = 16;
  private modelVersion: 'v3' | 'v2' = 'v3';
  private options: DepthAnythingControllerOptions;
  private frameCounter: number = 0;
  private lastFpsTime: number = 0;
  private framesSinceFps: number = 0;
  private currentFps: number = 0;
  private initialized: boolean = false;
  private initPromise: Promise<void> | null = null;
  private currentBackend: 'webgpu' | 'wasm' | 'custom' = 'wasm';
  private pendingSbsRgba: Uint8Array | null = null;
  private pendingSbsWidth: number = 0;
  private pendingSbsHeight: number = 0;
  private pendingBaseDisp: number = 0;
  private pendingMaxDisp: number = 16;
  private frameStartTime: number = 0;
  private videoCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private videoCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

  constructor(
    options?: DepthAnythingControllerOptions,
    onDepthMapReady?: DepthMapReadyCallback
  ) {
    this.options = { ...options };
    this.modelVersion = options?.modelVersion || 'v3';
    if (this.options.inputSize === undefined) this.options.inputSize = 504;
    if (this.options.expandRadius === undefined && this.options.dilateRadius === undefined) {
      this.options.expandRadius = 1;
    }
    if (onDepthMapReady) {
      this.options.onDepthMapReady = onDepthMapReady;
    }
    if (options?.baseDisp !== undefined) this.baseDisp = options.baseDisp;
    if (options?.maxDisp !== undefined) this.maxDisp = options.maxDisp;
  }

  public isBusy(): boolean {
    return this.busy;
  }

  public isReady(): boolean {
    return this.initialized;
  }

  public getModelVersion(): 'v3' | 'v2' {
    return this.modelVersion;
  }

  public async setModelVersion(version: 'v3' | 'v2'): Promise<void> {
    if (this.modelVersion === version && this.initialized) {
      return;
    }
    this.modelVersion = version;
    this.options.modelVersion = version;
    this.options.inputSize = version === 'v2' ? 518 : 504;

    const defaultModel = version === 'v3' ? '/models/depthanythingv3.onnx' : '/models/depthanythingv2.onnx';
    const modelPath = typeof window !== 'undefined'
      ? new URL('.' + defaultModel, window.location.href).href
      : defaultModel;
    this.options.modelPath = modelPath;

    if (this.worker) {
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => resolve(), 15000);
        const handler = (e: MessageEvent<WorkerOutMessage>) => {
          if (e.data.type === 'initDone') {
            clearTimeout(timeout);
            this.currentBackend = e.data.backend;
            this.initialized = true;
            this.worker?.removeEventListener('message', handler);
            resolve();
          } else if (e.data.type === 'error') {
            clearTimeout(timeout);
            this.worker?.removeEventListener('message', handler);
            reject(new Error(e.data.error));
          }
        };
        this.worker?.addEventListener('message', handler);

        const defaultWasmPaths = typeof window !== 'undefined'
          ? new URL('./onnx-wasm/', window.location.href).href
          : '/onnx-wasm/';

        const initMsg: WorkerInitMessage = {
          type: 'init',
          wasmPaths: this.options.wasmPaths || defaultWasmPaths,
          modelPath: this.options.modelPath,
          modelPathWebGPU: this.options.modelPathWebGPU,
          preferWebGPU: true,
        };
        this.worker?.postMessage(initMsg);
      });
    } else {
      this.initialized = false;
      this.initPromise = null;
      await this.init();
    }
  }

  public setBaseDisparity(baseDisp: number): void {
    this.baseDisp = baseDisp;
  }

  public getBaseDisparity(): number {
    return this.baseDisp;
  }

  public setMaxDisparity(maxDisp: number): void {
    this.maxDisp = maxDisp;
  }

  public getMaxDisparity(): number {
    return this.maxDisp;
  }

  public getBackend(): string {
    return this.currentBackend;
  }

  public get sbsReconstructionMae(): number {
    return this.lastTelemetry?.sbsReconstructionMae ?? 0;
  }

  public get scaleA(): number {
    return this.lastTelemetry?.scaleA ?? 1;
  }

  public get shiftB(): number {
    return this.lastTelemetry?.shiftB ?? 0;
  }

  public get anchorCount(): number {
    return this.lastTelemetry?.anchorCount ?? 0;
  }

  public get inferenceTimeMs(): number {
    return this.lastTelemetry?.inferenceTimeMs ?? 0;
  }

  public hasExternalDepthMap(): boolean {
    return this.lastTelemetry !== null;
  }

  public getLastTelemetry(): DepthTelemetry | null {
    return this.lastTelemetry;
  }

  public async init(): Promise<void> {
    if (this.initialized) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      if (this.options.customInfer) {
        this.initialized = true;
        this.currentBackend = 'custom';
        return;
      }

      if (typeof Worker === 'undefined') {
        console.warn('DepthAnythingController: Web Worker is unavailable in this environment');
        this.initialized = true;
        return;
      }

      const tryInitModel = async (version: 'v3' | 'v2'): Promise<void> => {
        const defaultModel = version === 'v3' ? '/models/depthanythingv3.onnx' : '/models/depthanythingv2.onnx';
        const defaultModelPath = typeof window !== 'undefined'
          ? new URL('.' + defaultModel, window.location.href).href
          : defaultModel;

        const defaultWasmPaths = typeof window !== 'undefined'
          ? new URL('./onnx-wasm/', window.location.href).href
          : '/onnx-wasm/';

        const initMsg: WorkerInitMessage = {
          type: 'init',
          wasmPaths: this.options.wasmPaths || defaultWasmPaths,
          modelPath: this.options.modelPath || defaultModelPath,
          modelPathWebGPU: this.options.modelPathWebGPU,
          preferWebGPU: true,
        };

        if (!this.worker) {
          if (this.options.worker) {
            this.worker = this.options.worker;
          } else if (this.options.workerUrl) {
            this.worker = new Worker(this.options.workerUrl, { type: 'module' });
          } else {
            this.worker = new Worker(new URL('./depthAnythingWorker.ts', import.meta.url), { type: 'module' });
          }
          this.setupWorkerListeners();
        }

        await new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => resolve(), 15000);
          const initHandler = (e: MessageEvent<WorkerOutMessage>) => {
            if (e.data.type === 'initDone') {
              clearTimeout(timeout);
              this.currentBackend = e.data.backend;
              this.initialized = true;
              this.worker?.removeEventListener('message', initHandler);
              resolve();
            } else if (e.data.type === 'error') {
              clearTimeout(timeout);
              this.worker?.removeEventListener('message', initHandler);
              reject(new Error(e.data.error));
            }
          };
          this.worker?.addEventListener('message', initHandler);
          this.worker?.postMessage(initMsg);
        });
      };

      try {
        await tryInitModel(this.modelVersion);
      } catch (err) {
        if (this.modelVersion === 'v3') {
          console.warn('DepthAnythingController: v3 init failed, falling back to v2:', err);
          this.modelVersion = 'v2';
          this.options.modelVersion = 'v2';
          this.options.modelPath = undefined;
          await tryInitModel('v2');
        } else {
          console.warn('DepthAnythingController init error:', err);
          this.initialized = true;
        }
      }
    })();

    return this.initPromise;
  }

  private setupWorkerListeners(): void {
    if (!this.worker) return;

    this.worker.onmessage = (e: MessageEvent<WorkerOutMessage>) => {
      const data = e.data;
      if (data.type === 'depthReady') {
        this.handleDepthResult(data);
      } else if (data.type === 'error') {
        console.warn('DepthAnythingWorker error:', data.error);
        this.busy = false;
        this.pendingSbsRgba = null;
      }
    };
  }

  private handleDepthResult(result: WorkerDepthResultMessage): void {
    try {
      const totalTimeMs = performance.now() - this.frameStartTime;
      const sbsRgba = this.pendingSbsRgba;
      const sbsWidth = this.pendingSbsWidth;
      const sbsHeight = this.pendingSbsHeight;
      const baseDisp = this.pendingBaseDisp;
      const maxDisp = this.pendingMaxDisp;

      if (!sbsRgba) {
        this.busy = false;
        return;
      }

      // Step: Lock raw DepthAnything depth to real SBS stereo pair
      const expandRadius = this.options.dilateRadius ?? this.options.expandRadius ?? 1;
      const calResult = calibrateDepthWithStereoSBS(
        result.rawDepth,
        sbsRgba,
        baseDisp,
        maxDisp,
        sbsWidth,
        sbsHeight,
        {
          expandRadius,
          gradThreshold: this.options.gradThreshold ?? 12,
          step: this.options.step ?? 2,
          contrastStretch: this.options.contrastStretch ?? true,
          rightDaDepth: result.rawRightDepth,
          conf0: result.conf0 ?? result.confidence,
          conf1: result.conf1,
        }
      );

      // Track FPS
      this.framesSinceFps++;
      const now = performance.now();
      if (now - this.lastFpsTime >= 1000) {
        this.currentFps = (this.framesSinceFps * 1000) / (now - this.lastFpsTime);
        this.lastFpsTime = now;
        this.framesSinceFps = 0;
      }

      const telemetry: DepthTelemetry = {
        inferenceTimeMs: result.inferenceTimeMs,
        totalTimeMs,
        scaleA: calResult.scaleA,
        shiftB: calResult.shiftB,
        sbsReconstructionMae: calResult.sbsReconstructionMae,
        backend: result.backend,
        fps: Math.round(this.currentFps * 10) / 10,
        frameId: result.frameId || this.frameCounter,
        anchorCount: calResult.anchorCount,
        modelVersion: this.modelVersion,
        numViews: result.numViews ?? (this.modelVersion === 'v3' ? 2 : 1),
      };

      this.lastTelemetry = telemetry;

      this.options.onDepthMapReady?.(
        calResult.calibratedDepth,
        calResult.width,
        calResult.height,
        telemetry
      );
    } finally {
      this.busy = false;
      this.pendingSbsRgba = null;
    }
  }

  /**
   * Submits an SBS video element frame for async depth estimation and calibration.
   * If already busy inferring a previous frame, returns false immediately (non-blocking).
   */
  public async processVideoFrame(
    video: HTMLVideoElement,
    baseDisp?: number,
    maxDisp?: number
  ): Promise<boolean> {
    if (this.busy) {
      return false;
    }
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return false;
    }
    const sbsW = video.videoWidth;
    const sbsH = video.videoHeight;
    if (!sbsW || !sbsH) {
      return false;
    }

    if (!this.initialized && !this.initPromise) {
      this.init().catch((e) => console.warn('DepthAnythingController auto-init error:', e));
    }

    if (!this.videoCanvas) {
      if (typeof document !== 'undefined') {
        this.videoCanvas = document.createElement('canvas');
      } else if (typeof OffscreenCanvas !== 'undefined') {
        this.videoCanvas = new OffscreenCanvas(sbsW, sbsH);
      }
    }
    if (this.videoCanvas) {
      if (this.videoCanvas.width !== sbsW || this.videoCanvas.height !== sbsH) {
        this.videoCanvas.width = sbsW;
        this.videoCanvas.height = sbsH;
      }
      if (!this.videoCtx) {
        this.videoCtx = this.videoCanvas.getContext('2d', { willReadFrequently: true }) as any;
      }
    }
    if (!this.videoCtx) {
      return false;
    }

    this.videoCtx.drawImage(video, 0, 0, sbsW, sbsH);
    const imgData = this.videoCtx.getImageData(0, 0, sbsW, sbsH);
    const rawBytes = new Uint8Array(imgData.data.buffer, imgData.data.byteOffset, imgData.data.byteLength);
    return this.processFrame(rawBytes, sbsW, sbsH, baseDisp, maxDisp);
  }

  /**
   * Submits an SBS frame for async depth estimation and calibration.
   * If already busy inferring a previous frame, returns false immediately (non-blocking).
   */
  public processFrame(
    sbsRgba: Uint8Array,
    sbsWidth: number = 960,
    sbsHeight: number = 240,
    baseDisp?: number,
    maxDisp?: number
  ): boolean {
    if (this.busy) {
      return false; // Skip frame to keep 60 FPS render loop fluid
    }

    this.busy = true;
    this.frameStartTime = performance.now();
    this.frameCounter++;

    const eyeWidth = Math.floor(sbsWidth / 2);
    const eyeHeight = sbsHeight;
    const effectiveBaseDisp = baseDisp !== undefined ? baseDisp : this.baseDisp;
    const effectiveMaxDisp = maxDisp !== undefined ? maxDisp : this.maxDisp;

    // Cache SBS frame for post-worker stereo calibration
    this.pendingSbsRgba = new Uint8Array(sbsRgba);
    this.pendingSbsWidth = sbsWidth;
    this.pendingSbsHeight = sbsHeight;
    this.pendingBaseDisp = effectiveBaseDisp;
    this.pendingMaxDisp = effectiveMaxDisp;

    const leftEyeRgba = extractLeftEyeRgba(sbsRgba, sbsWidth, sbsHeight);
    const rightEyeRgba = this.modelVersion === 'v3' ? extractRightEyeRgba(sbsRgba, sbsWidth, sbsHeight) : undefined;
    const inputSize = this.options.inputSize || 504;

    if (this.options.customInfer) {
      this.options
        .customInfer(leftEyeRgba, eyeWidth, eyeHeight, inputSize, rightEyeRgba)
        .then((res) => {
          this.handleDepthResult({
            type: 'depthReady',
            frameId: this.frameCounter,
            rawDepth: res.rawDepth,
            numViews: rightEyeRgba ? 2 : 1,
            width: eyeWidth,
            height: eyeHeight,
            backend: res.backend,
            inferenceTimeMs: res.inferenceTimeMs,
          });
        })
        .catch((err) => {
          console.warn('Custom infer error:', err);
          this.busy = false;
          this.pendingSbsRgba = null;
        });
      return true;
    }

    if (this.worker) {
      const msg: WorkerInferMessage = {
        type: 'infer',
        frameId: this.frameCounter,
        rgbaData: leftEyeRgba,
        rightRgbaData: rightEyeRgba,
        width: eyeWidth,
        height: eyeHeight,
        size: inputSize,
      };
      // Transfer buffers to avoid memory copies
      const transferables: Transferable[] = [leftEyeRgba.buffer];
      if (rightEyeRgba) {
        transferables.push(rightEyeRgba.buffer);
      }
      this.worker.postMessage(msg, transferables);
      return true;
    }

    // Fallback if no worker available
    this.busy = false;
    this.pendingSbsRgba = null;
    return false;
  }

  public destroy(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.busy = false;
    this.initialized = false;
    this.initPromise = null;
    this.pendingSbsRgba = null;
    this.videoCanvas = null;
    this.videoCtx = null;
  }
}
