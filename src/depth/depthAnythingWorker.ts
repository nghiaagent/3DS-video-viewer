import * as ort from 'onnxruntime-web';

export interface WorkerInitMessage {
  type: 'init';
  wasmPaths?: Record<string, string> | string;
  modelPath?: string;
  modelPathWebGPU?: string;
  preferWebGPU?: boolean;
  modelVersion?: 'v3' | 'v2';
}

export interface WorkerInferMessage {
  type: 'infer';
  frameId?: number;
  rgbaData: Uint8Array;
  rightRgbaData?: Uint8Array;
  width: number;
  height: number;
  size?: number; // model input resolution, default 504 (multiple of 14: 36 * 14)
}

export interface WorkerDepthResultMessage {
  type: 'depthReady';
  frameId?: number;
  rawDepth: Float32Array; // View 0 (left) normalized [0, 1], resampled to original width x height
  rawRightDepth?: Float32Array; // View 1 (right) normalized [0, 1]
  conf0?: Float32Array;
  conf1?: Float32Array;
  modelDepth?: Float32Array;
  confidence?: Float32Array;
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
 * Bilinearly scales RGBA image to targetSize x targetSize planar RGB Float32Array (RRR...GGG...BBB).
 * Applies ImageNet normalization ((rgb - mean) / std) by default for sharpest ViT features.
 * Preserves aspect ratio using center letterboxing and edge clamping to eliminate border artifacts.
 */
export function preprocessRgbaToPlanarTensor(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
  targetSize: number,
  normalizeImageNet: boolean = true
): {
  planarData: Float32Array;
  letterbox: { offsetX: number; offsetY: number; scaledW: number; scaledH: number };
} {
  const planarData = new Float32Array(3 * targetSize * targetSize);
  const scale = Math.min(targetSize / srcW, targetSize / srcH);
  const scaledW = Math.max(1, Math.min(targetSize, Math.round(srcW * scale)));
  const scaledH = Math.max(1, Math.min(targetSize, Math.round(srcH * scale)));
  const offsetX = Math.floor((targetSize - scaledW) / 2);
  const offsetY = Math.floor((targetSize - scaledH) / 2);

  const rOffset = 0;
  const gOffset = targetSize * targetSize;
  const bOffset = 2 * targetSize * targetSize;

  const meanR = 0.485;
  const meanG = 0.456;
  const meanB = 0.406;
  const stdR = 0.229;
  const stdG = 0.224;
  const stdB = 0.225;

  for (let ty = 0; ty < targetSize; ty++) {
    // Map ty back to src image coordinate with border clamping
    const syClamped = Math.max(0, Math.min(srcH - 1, (ty - offsetY) * (srcH / scaledH)));
    const sy0 = Math.floor(syClamped);
    const sy1 = Math.min(srcH - 1, sy0 + 1);
    const fy = syClamped - sy0;

    const rowOffset = ty * targetSize;

    for (let tx = 0; tx < targetSize; tx++) {
      const sxClamped = Math.max(0, Math.min(srcW - 1, (tx - offsetX) * (srcW / scaledW)));
      const sx0 = Math.floor(sxClamped);
      const sx1 = Math.min(srcW - 1, sx0 + 1);
      const fx = sxClamped - sx0;

      const idx00 = (sy0 * srcW + sx0) * 4;
      const idx01 = (sy0 * srcW + sx1) * 4;
      const idx10 = (sy1 * srcW + sx0) * 4;
      const idx11 = (sy1 * srcW + sx1) * 4;

      // Bilinear interpolation for R, G, B
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

  return { planarData, letterbox: { offsetX, offsetY, scaledW, scaledH } };
}

/**
 * Normalizes raw model output tensor to [0, 1] and resamples back to target width x height,
 * properly removing letterbox padding.
 */
export function postprocessDepthTensor(
  tensorData: Float32Array,
  modelW: number,
  modelH: number,
  targetW: number,
  targetH: number,
  letterbox?: { offsetX: number; offsetY: number; scaledW: number; scaledH: number; inputSize: number }
): Float32Array {
  // 1. Find min and max
  let minDepth = Infinity;
  let maxDepth = -Infinity;
  const total = modelW * modelH;
  for (let i = 0; i < total; i++) {
    const val = tensorData[i];
    if (val < minDepth) minDepth = val;
    if (val > maxDepth) maxDepth = val;
  }

  const range = maxDepth - minDepth;
  const invRange = range > 1e-6 ? 1.0 / range : 0;

  // 2. If dimensions already match target and no letterbox, normalize directly
  if (!letterbox && modelW === targetW && modelH === targetH) {
    const direct = new Float32Array(total);
    for (let i = 0; i < total; i++) {
      direct[i] = invRange > 0 ? (tensorData[i] - minDepth) * invRange : 0.5;
    }
    return direct;
  }

  // 3. Bilinearly resample to targetW x targetH, un-letterboxing
  const out = new Float32Array(targetW * targetH);
  const inputSize = letterbox ? letterbox.inputSize : modelW;
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

      out[rowOffset + x] = invRange > 0 ? (val - minDepth) * invRange : 0.5;
    }
  }

  return out;
}

/**
 * Initializes the ONNX inference session with WebGPU or WASM fallback.
 */
export async function initSession(options: {
  wasmPaths?: Record<string, string> | string;
  modelPath?: string;
  modelPathWebGPU?: string;
  preferWebGPU?: boolean;
  modelVersion?: 'v3' | 'v2';
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

  const defaultModel = options.modelVersion === 'v2'
    ? '/models/depthanythingv2.onnx'
    : '/models/depthanythingv3.onnx';
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
    try {
      session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
      });
      backend = 'wasm';
    } catch (wasmErr) {
      if (modelPath.includes('depthanythingv3.onnx')) {
        const fallbackPath = modelPath.replace('depthanythingv3.onnx', 'depthanythingv2.onnx');
        console.warn(`DepthAnythingWorker: Failed to load ${modelPath}, falling back to ${fallbackPath}:`, wasmErr);
        session = await ort.InferenceSession.create(fallbackPath, {
          executionProviders: ['wasm'],
        });
        backend = 'wasm';
      } else {
        throw wasmErr;
      }
    }
  }

  cachedSession = session;
  cachedBackend = backend;
  isInitialized = true;
  return { session, backend };
}

/**
 * Runs inference on raw RGBA Left eye (and optional Right eye) pixels using an active session.
 * Supports both DepthAnythingV3 (rank 5 [1, 2, 3, H, W] for 2-view multi-view stereo or [1, 1, 3, H, W] for mono)
 * and DepthAnythingV2 (rank 4 [1, 3, H, W], 'image', 'depth').
 */
export async function runDepthInference(
  session: ort.InferenceSession,
  rgbaData: Uint8Array,
  width: number,
  height: number,
  size: number = 504,
  rightRgbaData?: Uint8Array
): Promise<{
  rawDepth: Float32Array;
  rawRightDepth?: Float32Array;
  conf0?: Float32Array;
  conf1?: Float32Array;
  confidence?: Float32Array;
  numViews: number;
  inferenceTimeMs: number;
}> {
  const { planarData: leftPlanar, letterbox } = preprocessRgbaToPlanarTensor(rgbaData, width, height, size, true);

  const inputName = session.inputNames[0] || 'pixel_values';
  const isV3 = inputName === 'pixel_values';

  let inputTensor: ort.Tensor;
  let numViews = 1;

  if (isV3 && rightRgbaData) {
    // Multi-view stereo forward pass for DepthAnything V3: [1, 2, 3, size, size]
    const { planarData: rightPlanar } = preprocessRgbaToPlanarTensor(rightRgbaData, width, height, size, true);
    const stereoPlanar = new Float32Array(2 * 3 * size * size);
    stereoPlanar.set(leftPlanar, 0);
    stereoPlanar.set(rightPlanar, 3 * size * size);
    inputTensor = new ort.Tensor('float32', stereoPlanar, [1, 2, 3, size, size]);
    numViews = 2;
  } else if (isV3) {
    inputTensor = new ort.Tensor('float32', leftPlanar, [1, 1, 3, size, size]);
    numViews = 1;
  } else if (rightRgbaData) {
    // Both perspectives for DepthAnything V2: [2, 3, size, size]
    const { planarData: rightPlanar } = preprocessRgbaToPlanarTensor(rightRgbaData, width, height, size, true);
    const stereoPlanar = new Float32Array(2 * 3 * size * size);
    stereoPlanar.set(leftPlanar, 0);
    stereoPlanar.set(rightPlanar, 3 * size * size);
    inputTensor = new ort.Tensor('float32', stereoPlanar, [2, 3, size, size]);
    numViews = 2;
  } else {
    // Monocular rank 4 [1, 3, H, W] for DepthAnything V2
    inputTensor = new ort.Tensor('float32', leftPlanar, [1, 3, size, size]);
    numViews = 1;
  }

  const t0 = performance.now();
  let results: Record<string, ort.Tensor>;
  try {
    results = await session.run({ [inputName]: inputTensor });
  } catch (err: any) {
    if (numViews === 2) {
      if (isV3) {
        console.warn('DepthAnythingWorker: 2-view stereo run failed for V3, falling back to 1-view:', err?.message || err);
        const singleViewTensor = new ort.Tensor('float32', leftPlanar, [1, 1, 3, size, size]);
        results = await session.run({ [inputName]: singleViewTensor });
        numViews = 1;
      } else {
        try {
          console.warn('DepthAnythingWorker: batch 2 failed for V2, running 2 sequential single-view passes:', err?.message || err);
          const tLeft = new ort.Tensor('float32', leftPlanar, [1, 3, size, size]);
          const resLeft = await session.run({ [inputName]: tLeft });
          const outKey = session.outputNames[0] || 'depth';
          const leftOutTensor = resLeft[outKey];
          const leftData = leftOutTensor.data as Float32Array;

          const { planarData: rightPlanar } = preprocessRgbaToPlanarTensor(rightRgbaData!, width, height, size, true);
          const tRight = new ort.Tensor('float32', rightPlanar, [1, 3, size, size]);
          const resRight = await session.run({ [inputName]: tRight });
          const rightOutTensor = resRight[outKey];
          const rightData = rightOutTensor.data as Float32Array;

          const combinedData = new Float32Array(leftData.length + rightData.length);
          combinedData.set(leftData, 0);
          combinedData.set(rightData, leftData.length);

          const outDims = [2, ...leftOutTensor.dims.slice(1)];
          results = {
            [outKey]: new ort.Tensor('float32', combinedData, outDims),
          };
        } catch (seqErr) {
          console.warn('DepthAnythingWorker: sequential 2-view V2 run failed, falling back to 1-view:', seqErr);
          const singleViewTensor = new ort.Tensor('float32', leftPlanar, [1, 3, size, size]);
          results = await session.run({ [inputName]: singleViewTensor });
          numViews = 1;
        }
      }
    } else if (err && String(err.message).includes('Expected: 5')) {
      const tensor5D = new ort.Tensor('float32', leftPlanar, [1, 1, 3, size, size]);
      results = await session.run({ [inputName]: tensor5D });
    } else if (err && String(err.message).includes('Expected: 4')) {
      const tensor4D = new ort.Tensor('float32', leftPlanar, [1, 3, size, size]);
      results = await session.run({ [inputName]: tensor4D });
    } else {
      throw err;
    }
  }
  const inferenceTimeMs = performance.now() - t0;

  let outputName = session.outputNames[0] || 'predicted_depth';
  if (session.outputNames.includes('predicted_depth')) {
    outputName = 'predicted_depth';
  } else if (session.outputNames.includes('depth')) {
    outputName = 'depth';
  }

  const outputTensor = results[outputName];
  if (!outputTensor) {
    throw new Error(`Output tensor '${outputName}' not found in results: ${Object.keys(results)}`);
  }

  const dims = outputTensor.dims;
  const modelH = dims.length >= 2 ? dims[dims.length - 2] : size;
  const modelW = dims.length >= 2 ? dims[dims.length - 1] : size;
  const viewElements = modelH * modelW;

  const tensorData = outputTensor.data as Float32Array;
  // View 0 is Left Eye [0, viewElements)
  const leftDepthData = numViews > 1 ? tensorData.subarray(0, viewElements) : tensorData;

  const rawDepth = postprocessDepthTensor(leftDepthData, modelW, modelH, width, height, {
    ...letterbox,
    inputSize: size,
  });

  let rawRightDepth: Float32Array | undefined;
  if (numViews > 1) {
    const rightDepthData = tensorData.subarray(viewElements, 2 * viewElements);
    rawRightDepth = postprocessDepthTensor(rightDepthData, modelW, modelH, width, height, {
      ...letterbox,
      inputSize: size,
    });
  }

  let conf0: Float32Array | undefined;
  let conf1: Float32Array | undefined;
  if (results['confidence']) {
    const confTensor = results['confidence'];
    const confData = confTensor.data as Float32Array;
    const leftConf = numViews > 1 ? confData.subarray(0, viewElements) : confData;
    conf0 = postprocessDepthTensor(leftConf, modelW, modelH, width, height, {
      ...letterbox,
      inputSize: size,
    });
    if (numViews > 1) {
      const rightConf = confData.subarray(viewElements, 2 * viewElements);
      conf1 = postprocessDepthTensor(rightConf, modelW, modelH, width, height, {
        ...letterbox,
        inputSize: size,
      });
    }
  }

  return { rawDepth, rawRightDepth, conf0, conf1, confidence: conf0, numViews, inferenceTimeMs };
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
        const inputSize = data.size || 504;
        const { rawDepth, rawRightDepth, conf0, conf1, confidence, numViews, inferenceTimeMs } = await runDepthInference(
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
          numViews,
          width: data.width,
          height: data.height,
          backend: cachedBackend,
          inferenceTimeMs,
        };

        // Transfer rawDepth (and optional buffers) for zero-copy IPC
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


