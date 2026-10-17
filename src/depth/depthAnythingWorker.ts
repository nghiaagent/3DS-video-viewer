import * as ort from 'onnxruntime-web';

export interface WorkerInitMessage {
  type: 'init';
  wasmPaths?: Record<string, string> | string;
  modelPath?: string;
  modelPathWebGPU?: string;
  preferWebGPU?: boolean;
  modelVersion?: 'v3';
}

export interface WorkerInferMessage {
  type: 'infer';
  frameId?: number;
  rgbaData: Uint8Array;
  rightRgbaData?: Uint8Array;
  width: number;
  height: number;
  size?: number; // model input resolution, default 392 (392x196: 28*14 x 14*14 exact 2:1)
}

export interface WorkerDepthResultMessage {
  type: 'depthReady';
  frameId?: number;
  rawDepth: Float32Array; // View 0 (left) joint inverse depth in [0, 1], resampled to original width x height
  rawRightDepth?: Float32Array; // View 1 (right) joint inverse depth in [0, 1]
  conf0?: Float32Array;
  conf1?: Float32Array;
  confidence?: Float32Array;
  v3BaselinePx?: number;
  numViews?: number;
  width: number;
  height: number;
  backend: 'webgpu' | 'wasm';
  inferenceTimeMs: number;
}

export interface WorkerErrorMessage {
  type: 'error';
  frameId?: number;
  error: string;
}

export type WorkerInMessage = WorkerInitMessage | WorkerInferMessage;
export type WorkerOutMessage =
  | { type: 'initDone'; backend: 'webgpu' | 'wasm' }
  | WorkerDepthResultMessage
  | WorkerErrorMessage;

let cachedSession: ort.InferenceSession | null = null;
let cachedBackend: 'webgpu' | 'wasm' = 'wasm';
let isInitialized = false;

/**
 * Bilinearly scales RGBA image to native 2:1 rectangular ViT-14 resolution (modelW x modelH),
 * ensuring modelW and modelH are multiples of 14 with ZERO letterbox waste when aspect ratios match.
 * Applies ImageNet normalization ((rgb - mean) / std).
 */
export function preprocessRgbaToPlanarTensor(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
  targetSize: number | { targetW?: number; targetH?: number } = 392,
  normalizeImageNet: boolean = true
): {
  planarData: Float32Array;
  modelW: number;
  modelH: number;
  letterbox: { offsetX: number; offsetY: number; scaledW: number; scaledH: number };
} {
  let modelW: number;
  let modelH: number;

  if (typeof targetSize === 'object' && targetSize !== null) {
    const tw = targetSize.targetW || 392;
    modelW = Math.max(14, Math.round(tw / 14) * 14);
    const th = targetSize.targetH || (modelW * (srcH / srcW));
    modelH = Math.max(14, Math.round(th / 14) * 14);
  } else {
    const size = typeof targetSize === 'number' ? targetSize : 392;
    modelW = Math.max(14, Math.round(size / 14) * 14);
    modelH = Math.max(14, Math.round((modelW * (srcH / srcW)) / 14) * 14);
  }

  const planarData = new Float32Array(3 * modelH * modelW);
  const rOffset = 0;
  const gOffset = modelH * modelW;
  const bOffset = 2 * modelH * modelW;

  const meanR = 0.485;
  const meanG = 0.456;
  const meanB = 0.406;
  const stdR = 0.229;
  const stdG = 0.224;
  const stdB = 0.225;

  const scaleY = srcH / modelH;
  const scaleX = srcW / modelW;

  for (let ty = 0; ty < modelH; ty++) {
    const syClamped = Math.max(0, Math.min(srcH - 1, ty * scaleY));
    const sy0 = Math.floor(syClamped);
    const sy1 = Math.min(srcH - 1, sy0 + 1);
    const fy = syClamped - sy0;

    const rowOffset = ty * modelW;

    for (let tx = 0; tx < modelW; tx++) {
      const sxClamped = Math.max(0, Math.min(srcW - 1, tx * scaleX));
      const sx0 = Math.floor(sxClamped);
      const sx1 = Math.min(srcW - 1, sx0 + 1);
      const fx = sxClamped - sx0;

      const idx00 = (sy0 * srcW + sx0) * 4;
      const idx01 = (sy0 * srcW + sx1) * 4;
      const idx10 = (sy1 * srcW + sx0) * 4;
      const idx11 = (sy1 * srcW + sx1) * 4;

      const r = ((rgba[idx00] * (1 - fx) + rgba[idx01] * fx) * (1 - fy) +
                (rgba[idx10] * (1 - fx) + rgba[idx11] * fx) * fy) / 255.0;
      const g = ((rgba[idx00 + 1] * (1 - fx) + rgba[idx01 + 1] * fx) * (1 - fy) +
                (rgba[idx10 + 1] * (1 - fx) + rgba[idx11 + 1] * fx) * fy) / 255.0;
      const b = ((rgba[idx00 + 2] * (1 - fx) + rgba[idx01 + 2] * fx) * (1 - fy) +
                (rgba[idx10 + 2] * (1 - fx) + rgba[idx11 + 2] * fx) * fy) / 255.0;

      const destIdx = rowOffset + tx;
      if (normalizeImageNet) {
        planarData[rOffset + destIdx] = (r - meanR) / stdR;
        planarData[gOffset + destIdx] = (g - meanG) / stdG;
        planarData[bOffset + destIdx] = (b - meanB) / stdB;
      } else {
        planarData[rOffset + destIdx] = r;
        planarData[gOffset + destIdx] = g;
        planarData[bOffset + destIdx] = b;
      }
    }
  }

  return {
    planarData,
    modelW,
    modelH,
    letterbox: { offsetX: 0, offsetY: 0, scaledW: modelW, scaledH: modelH },
  };
}

/**
 * Resamples depth tensor back to target width x height, optionally un-letterboxing or normalizing.
 */
export function postprocessDepthTensor(
  tensorData: Float32Array,
  modelW: number,
  modelH: number,
  targetW: number,
  targetH: number,
  letterbox?: { offsetX: number; offsetY: number; scaledW: number; scaledH: number; inputSize?: number },
  normalizeMinMax: boolean = true
): Float32Array {
  const total = modelW * modelH;
  let minDepth = 0;
  let invRange = 1.0;

  if (normalizeMinMax) {
    minDepth = Infinity;
    let maxDepth = -Infinity;
    for (let i = 0; i < total; i++) {
      const val = tensorData[i];
      if (val < minDepth) minDepth = val;
      if (val > maxDepth) maxDepth = val;
    }
    const range = maxDepth - minDepth;
    invRange = range > 1e-6 ? 1.0 / range : 0;
  }

  // If dimensions already match target and no letterbox offset
  const hasLetterbox = letterbox && (letterbox.offsetX !== 0 || letterbox.offsetY !== 0 ||
    letterbox.scaledW !== modelW || letterbox.scaledH !== modelH);

  if (!hasLetterbox && modelW === targetW && modelH === targetH) {
    if (!normalizeMinMax) {
      return new Float32Array(tensorData);
    }
    const direct = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      direct[i] = invRange > 0 ? (tensorData[i] - minDepth) * invRange : 0.5;
    }
    return direct;
  }

  const out = new Float32Array(targetW * targetH);
  const inputSize = letterbox ? (letterbox.inputSize ?? modelW) : modelW;
  const scaleXToModel = modelW / inputSize;
  const scaleYToModel = modelH / inputSize;

  const validOffsetX = (letterbox ? letterbox.offsetX : 0) * scaleXToModel;
  const validOffsetY = (letterbox ? letterbox.offsetY : 0) * scaleYToModel;
  const validScaledW = (letterbox ? letterbox.scaledW : modelW) * scaleXToModel;
  const validScaledH = (letterbox ? letterbox.scaledH : modelH) * scaleYToModel;

  for (let y = 0; y < targetH; y++) {
    const my = validOffsetY + (y / targetH) * validScaledH;
    const myClamped = Math.max(0, Math.min(modelH - 1, my));
    const y0 = Math.floor(myClamped);
    const y1 = Math.min(modelH - 1, y0 + 1);
    const fy = myClamped - y0;

    const rowOffset = y * targetW;

    for (let x = 0; x < targetW; x++) {
      const mx = validOffsetX + (x / targetW) * validScaledW;
      const mxClamped = Math.max(0, Math.min(modelW - 1, mx));
      const x0 = Math.floor(mxClamped);
      const x1 = Math.min(modelW - 1, x0 + 1);
      const fx = mxClamped - x0;

      const d00 = tensorData[y0 * modelW + x0];
      const d01 = tensorData[y0 * modelW + x1];
      const d10 = tensorData[y1 * modelW + x0];
      const d11 = tensorData[y1 * modelW + x1];

      const val = (d00 * (1 - fx) + d01 * fx) * (1 - fy) +
                  (d10 * (1 - fx) + d11 * fx) * fy;

      if (normalizeMinMax) {
        out[rowOffset + x] = invRange > 0 ? (val - minDepth) * invRange : 0.5;
      } else {
        out[rowOffset + x] = val;
      }
    }
  }

  return out;
}

/**
 * Initializes the ONNX inference session with WebGPU or WASM fallback.
 * Uses DepthAnythingV3 exclusively.
 */
export async function initSession(options: {
  wasmPaths?: Record<string, string> | string;
  modelPath?: string;
  modelPathWebGPU?: string;
  preferWebGPU?: boolean;
  modelVersion?: 'v3';
}): Promise<{ session: ort.InferenceSession; backend: 'webgpu' | 'wasm' }> {
  let wasmPaths = options.wasmPaths || '/onnx-wasm/';
  if (typeof wasmPaths === 'string') {
    if (typeof self !== 'undefined' && self.location && !wasmPaths.startsWith('http://') && !wasmPaths.startsWith('https://')) {
      wasmPaths = new URL(wasmPaths, self.location.origin + '/').href;
    }
    if (!wasmPaths.endsWith('/')) {
      wasmPaths += '/';
    }
    ort.env.wasm.wasmPaths = wasmPaths;
  } else {
    ort.env.wasm.wasmPaths = wasmPaths;
  }

  if (typeof navigator !== 'undefined') {
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
  }

  const defaultModel = '/models/depthanythingv3.onnx';
  let modelPath = options.modelPath || defaultModel;
  if (typeof self !== 'undefined' && self.location && !modelPath.startsWith('http://') && !modelPath.startsWith('https://') && !modelPath.startsWith('blob:')) {
    modelPath = new URL(modelPath, self.location.origin + '/').href;
  }

  let modelPathWebGPU = options.modelPathWebGPU || modelPath;
  if (typeof self !== 'undefined' && self.location && modelPathWebGPU && !modelPathWebGPU.startsWith('http://') && !modelPathWebGPU.startsWith('https://') && !modelPathWebGPU.startsWith('blob:')) {
    modelPathWebGPU = new URL(modelPathWebGPU, self.location.origin + '/').href;
  }

  const hasWebGPU =
    options.preferWebGPU !== false &&
    typeof navigator !== 'undefined' &&
    'gpu' in navigator;

  let session: ort.InferenceSession | null = null;
  let backend: 'webgpu' | 'wasm' = 'wasm';

  if (hasWebGPU && modelPathWebGPU) {
    try {
      session = await ort.InferenceSession.create(modelPathWebGPU, {
        executionProviders: ['webgpu', 'wasm'],
      });
      backend = 'webgpu';
    } catch (e) {
      console.warn('DepthAnythingWorker: WebGPU init failed, falling back to WASM:', e);
    }
  }

  if (!session) {
    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm'],
    });
    backend = 'wasm';
  }

  cachedSession = session;
  cachedBackend = backend;
  isInitialized = true;
  return { session, backend };
}

/**
 * Runs inference on raw RGBA Left eye (and optional Right eye) pixels using DepthAnythingV3.
 * Supports rank 5 [1, 2, 3, H, W] for 2-view multi-view stereo or [1, 1, 3, H, W] for mono.
 * Preserves shared multi-view inverse-depth space and extracts camera baseline shift.
 */
export async function runDepthInference(
  session: ort.InferenceSession,
  rgbaData: Uint8Array,
  width: number,
  height: number,
  size: number = 392,
  rightRgbaData?: Uint8Array
): Promise<{
  rawDepth: Float32Array;
  rawRightDepth?: Float32Array;
  conf0?: Float32Array;
  conf1?: Float32Array;
  confidence?: Float32Array;
  v3BaselinePx?: number;
  numViews: number;
  inferenceTimeMs: number;
}> {
  const { planarData: leftPlanar, modelW, modelH, letterbox } = preprocessRgbaToPlanarTensor(
    rgbaData,
    width,
    height,
    size,
    true
  );

  const inputName = session.inputNames[0] || 'pixel_values';
  let inputTensor: ort.Tensor;
  let numViews = 1;

  if (rightRgbaData) {
    // Multi-view stereo forward pass for DepthAnything V3: [1, 2, 3, modelH, modelW]
    const { planarData: rightPlanar } = preprocessRgbaToPlanarTensor(
      rightRgbaData,
      width,
      height,
      size,
      true
    );
    const stereoPlanar = new Float32Array(2 * 3 * modelH * modelW);
    stereoPlanar.set(leftPlanar, 0);
    stereoPlanar.set(rightPlanar, 3 * modelH * modelW);
    inputTensor = new ort.Tensor('float32', stereoPlanar, [1, 2, 3, modelH, modelW]);
    numViews = 2;
  } else {
    // Single view: [1, 1, 3, modelH, modelW]
    inputTensor = new ort.Tensor('float32', leftPlanar, [1, 1, 3, modelH, modelW]);
    numViews = 1;
  }

  const t0 = performance.now();
  let results: Record<string, ort.Tensor>;
  try {
    results = await session.run({ [inputName]: inputTensor });
  } catch (err: any) {
    if (numViews === 2) {
      console.warn('DepthAnythingWorker: 2-view stereo run failed, falling back to 1-view:', err?.message || err);
      const singleViewTensor = new ort.Tensor('float32', leftPlanar, [1, 1, 3, modelH, modelW]);
      results = await session.run({ [inputName]: singleViewTensor });
      numViews = 1;
    } else {
      throw err;
    }
  }
  const inferenceTimeMs = performance.now() - t0;

  const outputName = session.outputNames.includes('predicted_depth')
    ? 'predicted_depth'
    : (session.outputNames[0] || 'predicted_depth');

  const outputTensor = results[outputName];
  if (!outputTensor) {
    throw new Error(`Output tensor '${outputName}' not found in results: ${Object.keys(results)}`);
  }

  const dims = outputTensor.dims;
  const outH = dims.length >= 2 ? dims[dims.length - 2] : modelH;
  const outW = dims.length >= 2 ? dims[dims.length - 1] : modelW;
  const viewElements = outH * outW;
  const totalElements = numViews * viewElements;
  const rawZ = outputTensor.data as Float32Array;

  // Convert positive camera depth Z to inverse depth (disparity space)
  const jointInvZ = new Float32Array(totalElements);
  let sumZ = 0;
  for (let i = 0; i < totalElements; i++) {
    const z = rawZ[i];
    sumZ += z;
    jointInvZ[i] = 1.0 / Math.max(1e-3, z);
  }
  const meanZ = sumZ / Math.max(1, totalElements);

  // Joint percentile normalization across both views to preserve shared metric disparity space
  const sorted = Float32Array.from(jointInvZ).sort();
  const p2 = sorted[Math.floor(sorted.length * 0.02)];
  const p98 = sorted[Math.floor(sorted.length * 0.98)];
  const invRange = (p98 - p2) > 1e-6 ? 1.0 / (p98 - p2) : 1.0;

  const normJointInvZ = new Float32Array(totalElements);
  for (let i = 0; i < totalElements; i++) {
    normJointInvZ[i] = Math.max(0, Math.min(1, (jointInvZ[i] - p2) * invRange));
  }

  const rawDepth = postprocessDepthTensor(
    normJointInvZ.subarray(0, viewElements),
    outW,
    outH,
    width,
    height,
    letterbox,
    false
  );

  let rawRightDepth: Float32Array | undefined;
  if (numViews > 1) {
    rawRightDepth = postprocessDepthTensor(
      normJointInvZ.subarray(viewElements, 2 * viewElements),
      outW,
      outH,
      width,
      height,
      letterbox,
      false
    );
  }

  // Parse excess confidence weight: c = max(0.01, confVal - 0.99)
  let conf0: Float32Array | undefined;
  let conf1: Float32Array | undefined;
  if (results['confidence']) {
    const confData = results['confidence'].data as Float32Array;
    const c0 = new Float32Array(viewElements);
    for (let i = 0; i < viewElements; i++) {
      c0[i] = Math.max(0.01, confData[i] - 0.99);
    }
    conf0 = postprocessDepthTensor(c0, outW, outH, width, height, letterbox, false);

    if (numViews > 1) {
      const c1 = new Float32Array(viewElements);
      for (let i = 0; i < viewElements; i++) {
        c1[i] = Math.max(0.01, confData[viewElements + i] - 0.99);
      }
      conf1 = postprocessDepthTensor(c1, outW, outH, width, height, letterbox, false);
    }
  }

  // Parse extrinsics and intrinsics to compute predicted baseline shift in pixels:
  // v3PredictedBaselinePx = (fx * tx) / meanZ * (width / modelW)
  let v3BaselinePx: number | undefined;
  if (results['extrinsics'] && results['intrinsics'] && numViews > 1) {
    const ext = results['extrinsics'].data as Float32Array;
    const intr = results['intrinsics'].data as Float32Array;
    if (ext.length >= 16 && intr.length >= 10) {
      const tx = Math.abs(ext[15] - ext[3]);
      const fx = (intr[0] + intr[9]) * 0.5;
      v3BaselinePx = (fx * tx) / Math.max(1e-3, meanZ) * (width / outW);
    }
  }

  return {
    rawDepth,
    rawRightDepth,
    conf0,
    conf1,
    confidence: conf0,
    v3BaselinePx,
    numViews,
    inferenceTimeMs,
  };
}

// Web Worker message listener (only active in worker execution context)
if (typeof self !== 'undefined' && typeof (self as any).postMessage === 'function' && typeof window === 'undefined') {
  self.onmessage = async (e: MessageEvent<WorkerInMessage>) => {
    const data = e.data;

    if (data.type === 'init') {
      try {
        const { backend } = await initSession(data);
        (self as any).postMessage({ type: 'initDone', backend });
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        (self as any).postMessage({ type: 'error', error: msg });
      }
      return;
    }

    if (data.type === 'infer') {
      try {
        if (!cachedSession || !isInitialized) {
          throw new Error('DepthAnythingWorker not initialized');
        }
        const inputSize = data.size || 392;
        const { rawDepth, rawRightDepth, conf0, conf1, confidence, v3BaselinePx, numViews, inferenceTimeMs } =
          await runDepthInference(
            cachedSession,
            data.rgbaData,
            data.width,
            data.height,
            inputSize,
            data.rightRgbaData
          );

        const response: WorkerDepthResultMessage = {
          type: 'depthReady',
          frameId: data.frameId,
          rawDepth,
          rawRightDepth,
          conf0,
          conf1,
          confidence,
          v3BaselinePx,
          numViews,
          width: data.width,
          height: data.height,
          backend: cachedBackend,
          inferenceTimeMs,
        };

        const transferables: Transferable[] = [rawDepth.buffer];
        if (rawRightDepth) {
          transferables.push(rawRightDepth.buffer);
        }
        if (conf0) {
          transferables.push(conf0.buffer);
        }
        if (conf1) {
          transferables.push(conf1.buffer);
        }
        (self as any).postMessage(response, transferables);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        (self as any).postMessage({ type: 'error', frameId: data.frameId, error: msg });
      }
    }
  };
}



