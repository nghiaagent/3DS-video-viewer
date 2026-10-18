/**
 * Objective temporal-flicker metrics for per-frame depth maps.
 * Pure functions, no DOM.
 */

export interface FlickerScore {
  /** Mean over masked pixels of the temporal std of depth (raw, no normalization). */
  meanTemporalStd: number;
  /** Std over time of the per-frame masked-mean depth (global "pumping"). */
  meanPumping: number;
  /** Mean |d_t - d_{t-1}| over masked pixels and consecutive frame pairs. */
  frameToFrameMAD: number;
  /** Number of masked pixels used. */
  maskedPixels: number;
}

/**
 * Mask (1 = static) of pixels whose luma never changes by more than `thresh`
 * between consecutive frames. Frames are w*h luma (Uint8 0..255 or Float32).
 */
export function staticMask(
  frames: Array<Uint8Array | Float32Array>,
  w: number,
  h: number,
  thresh: number,
): Uint8Array {
  const n = w * h;
  const mask = new Uint8Array(n).fill(1);
  for (let t = 1; t < frames.length; t++) {
    const a = frames[t - 1];
    const b = frames[t];
    for (let i = 0; i < n; i++) {
      if (mask[i] && Math.abs(b[i] - a[i]) > thresh) mask[i] = 0;
    }
  }
  return mask;
}

export function flickerScore(depthFrames: Float32Array[], mask: Uint8Array): FlickerScore {
  const T = depthFrames.length;
  const n = mask.length;
  const idx: number[] = [];
  for (let i = 0; i < n; i++) if (mask[i]) idx.push(i);
  const m = idx.length;
  if (T < 2 || m === 0) {
    return { meanTemporalStd: 0, meanPumping: 0, frameToFrameMAD: 0, maskedPixels: m };
  }

  // Per-frame masked mean.
  const frameMeans = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const f = depthFrames[t];
    let s = 0;
    for (let k = 0; k < m; k++) s += f[idx[k]];
    frameMeans[t] = s / m;
  }
  const meanOfMeans = frameMeans.reduce((a, b) => a + b, 0) / T;
  let v = 0;
  for (let t = 0; t < T; t++) v += (frameMeans[t] - meanOfMeans) ** 2;
  const meanPumping = Math.sqrt(v / T);

  // Per-pixel temporal std and frame-to-frame MAD.
  let stdSum = 0;
  let madSum = 0;
  for (let k = 0; k < m; k++) {
    const i = idx[k];
    let s = 0;
    for (let t = 0; t < T; t++) s += depthFrames[t][i];
    const mu = s / T;
    let ss = 0;
    for (let t = 0; t < T; t++) ss += (depthFrames[t][i] - mu) ** 2;
    stdSum += Math.sqrt(ss / T);
    for (let t = 1; t < T; t++) madSum += Math.abs(depthFrames[t][i] - depthFrames[t - 1][i]);
  }

  return {
    meanTemporalStd: stdSum / m,
    meanPumping,
    frameToFrameMAD: madSum / (m * (T - 1)),
    maskedPixels: m,
  };
}
