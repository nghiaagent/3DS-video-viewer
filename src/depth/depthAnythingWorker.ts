import * as ort from 'onnxruntime-web';

export interface WorkerInitMessage {
  type: 'init';
  wasmPaths?: Record<string, string> | string;
  modelPath?: string;
  modelPathWebGPU?: string;
  preferWebGPU?: boolean;
}

export interface WorkerInferMessage {
  type: 'infer';
  frameId?: number;
  rgbaData: Uint8Array;
  width: number;
  height: number;
  size?: number; // model input resolution, default 256 or 364
}

export interface WorkerDepthResultMessage {
  type: 'depthReady';
  frameId?: number;
  rawDepth: Float32Array; // normalized [0, 1], resampled to original width x height
  modelDepth?: Float32Array;
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
 * Bilinearly scales RGBA image to targetSize x targetSize planar RGB Float32Array (RRR...GGG...BBB) in [0, 1].
 * Preserves aspect ratio using center letterboxing and edge clamping to eliminate border artifacts.
 */
export function preprocessRgbaToPlanarTensor(
  rgba: Uint8Array,
  srcW: number,
  srcH: number,
  targetSize: number
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
      const r = (rgba[idx00] * (1 - fx) + rgba[idx01] * fx) * (1 - fy) +
                (rgba[idx10] * (1 - fx) + rgba[idx11] * fx) * fy;
      const g = (rgba[idx00 + 1] * (1 - fx) + rgba[idx01 + 1] * fx) * (1 - fy) +
                (rgba[idx10 + 1] * (1 - fx) + rgba[idx11 + 1] * fx) * fy;
      const b = (rgba[idx00 + 2] * (1 - fx) + rgba[idx01 + 2] * fx) * (1 - fy) +
                (rgba[idx10 + 2] * (1 - fx) + rgba[idx11 + 2] * fx) * fy;

      const destIdx = rowOffset + tx;
      planarData[rOffset + destIdx] = r / 255.0;
      planarData[gOffset + destIdx] = g / 255.0;
      planarData[bOffset + destIdx] = b / 255.0;
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

  let modelPath = options.modelPath || '/models/depthanythingv2.onnx';
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
 * Runs inference on raw RGBA Left eye pixels using an active session.
 */
export async function runDepthInference(
  session: ort.InferenceSession,
  rgbaData: Uint8Array,
  width: number,
  height: number,
  size: number = 256
): Promise<{ rawDepth: Float32Array; inferenceTimeMs: number }> {
  const { planarData, letterbox } = preprocessRgbaToPlanarTensor(rgbaData, width, height, size);
  const inputTensor = new ort.Tensor('float32', planarData, [1, 3, size, size]);

  const inputName = session.inputNames[0] || 'image';
  const outputName = session.outputNames[0] || 'depth';

  const t0 = performance.now();
  const results = await session.run({ [inputName]: inputTensor });
  const inferenceTimeMs = performance.now() - t0;

  const outputTensor = results[outputName];
  const dims = outputTensor.dims;
  const modelH = dims.length >= 3 ? dims[dims.length - 2] : size;
  const modelW = dims.length >= 3 ? dims[dims.length - 1] : size;

  const tensorData = outputTensor.data as Float32Array;
  const rawDepth = postprocessDepthTensor(tensorData, modelW, modelH, width, height, {
    ...letterbox,
    inputSize: size,
  });

  return { rawDepth, inferenceTimeMs };
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
        const inputSize = data.size || 256;
        const { rawDepth, inferenceTimeMs } = await runDepthInference(
          cachedSession,
          data.rgbaData,
          data.width,
          data.height,
          inputSize
        );

        const response: WorkerDepthResultMessage = {
          type: 'depthReady',
          frameId: data.frameId,
          rawDepth,
          width: data.width,
          height: data.height,
          backend: cachedBackend,
          inferenceTimeMs,
        };

        // Transfer rawDepth buffer for zero-copy IPC
        (self as any).postMessage(response, [rawDepth.buffer]);
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : String(err);
        (self as any).postMessage({ type: 'error', frameId: data.frameId, error: msg });
      }
    }
  };
}
