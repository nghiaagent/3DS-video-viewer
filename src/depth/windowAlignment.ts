/**
 * Window alignment for multi-frame DepthAnything3 inference.
 *
 * DA3 inverse depth (1/Z) is only scale/shift-consistent within ONE forward pass.
 * Overlapping windows are chained into a single clip-level space by fitting an
 * affine map on the shared frames, then cross-fading the overlaps.
 * Pure functions, no ONNX/DOM.
 */

export interface AffineParams {
  a: number;
  b: number;
}

export interface DepthWindow {
  startFrame: number;
  invDepth: Float32Array[];
  conf?: Float32Array[];
}

export interface ClipRange {
  lo: number;
  hi: number;
}

const HUBER_K = 1.345;
const MAX_FIT_SAMPLES = 200000;

function weightedLS(
  src: Float32Array, dst: Float32Array, w: Float32Array | undefined,
  rw: Float32Array | null, idx: Int32Array
): AffineParams | null {
  let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k];
    const wi = (w ? w[i] : 1) * (rw ? rw[k] : 1);
    if (!(wi > 0)) continue;
    const x = src[i], y = dst[i];
    sw += wi; sx += wi * x; sy += wi * y; sxx += wi * x * x; sxy += wi * x * y;
  }
  if (sw <= 0) return null;
  const mx = sx / sw, my = sy / sw;
  const vx = sxx / sw - mx * mx;
  if (vx < 1e-12 * Math.max(1, mx * mx)) return { a: 1, b: my - mx };
  const a = (sxy / sw - mx * my) / vx;
  return { a, b: my - a * mx };
}

/**
 * Robust affine fit dst ~ a*src + b (inverse-depth space) using Huber IRLS.
 * Non-finite samples and non-positive weights are ignored.
 */
export function fitAffineInvDepth(
  src: Float32Array, dst: Float32Array, weights?: Float32Array, iterations = 8
): AffineParams {
  const n = Math.min(src.length, dst.length);
  const valid: number[] = [];
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(src[i]) || !Number.isFinite(dst[i])) continue;
    if (weights && !(Number.isFinite(weights[i]) && weights[i] > 0)) continue;
    valid.push(i);
  }
  if (valid.length === 0) return { a: 1, b: 0 };
  const stride = Math.max(1, Math.ceil(valid.length / MAX_FIT_SAMPLES));
  const m = Math.ceil(valid.length / stride);
  const idx = new Int32Array(m);
  for (let k = 0; k < m; k++) idx[k] = valid[k * stride];

  let p = weightedLS(src, dst, weights, null, idx) ?? { a: 1, b: 0 };
  const rw = new Float32Array(m);
  const res = new Float32Array(m);
  for (let it = 0; it < iterations; it++) {
    for (let k = 0; k < m; k++) res[k] = Math.abs(dst[idx[k]] - (p.a * src[idx[k]] + p.b));
    const sorted = Float32Array.from(res).sort();
    const sigma = Math.max(1e-6, (sorted[Math.floor(m / 2)] / 0.6745));
    const c = HUBER_K * sigma;
    for (let k = 0; k < m; k++) rw[k] = res[k] <= c ? 1 : c / res[k];
    const next = weightedLS(src, dst, weights, rw, idx);
    if (!next) break;
    const d = Math.abs(next.a - p.a) + Math.abs(next.b - p.b);
    p = next;
    if (d < 1e-7) break;
  }
  return p;
}

function applyAffine(src: Float32Array, p: AffineParams): Float32Array {
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = p.a * src[i] + p.b;
  return out;
}

function combineConf(c0?: Float32Array, c1?: Float32Array): Float32Array | undefined {
  if (!c0 && !c1) return undefined;
  if (!c0) return c1;
  if (!c1) return c0;
  const out = new Float32Array(Math.min(c0.length, c1.length));
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(Math.max(0, c0[i]) * Math.max(0, c1[i]));
  return out;
}

function concat(arrs: Float32Array[]): Float32Array {
  let n = 0;
  for (const a of arrs) n += a.length;
  const out = new Float32Array(n);
  let o = 0;
  for (const a of arrs) { out.set(a, o); o += a.length; }
  return out;
}

/**
 * Chain overlapping windows into one global inverse-depth space (the first
 * window defines the space) and cross-fade shared frames linearly.
 * Windows must be sorted by startFrame and overlap/abut their predecessor.
 * Returns one aligned inverse-depth array per frame, from windows[0].startFrame.
 */
export function alignWindows(windows: DepthWindow[]): Float32Array[] {
  if (windows.length === 0) return [];
  const base = windows[0].startFrame;
  const aligned: Float32Array[][] = [];
  for (let wi = 0; wi < windows.length; wi++) {
    const w = windows[wi];
    if (wi === 0) { aligned.push(w.invDepth.slice()); continue; }
    const prev = windows[wi - 1];
    const prevAligned = aligned[wi - 1];
    const srcs: Float32Array[] = [], dsts: Float32Array[] = [], confs: (Float32Array | undefined)[] = [];
    for (let k = 0; k < w.invDepth.length; k++) {
      const f = w.startFrame + k;
      const pk = f - prev.startFrame;
      if (pk < 0 || pk >= prev.invDepth.length) continue;
      srcs.push(w.invDepth[k]);
      dsts.push(prevAligned[pk]);
      confs.push(combineConf(w.conf?.[k], prev.conf?.[pk]));
    }
    let p: AffineParams = { a: 1, b: 0 };
    if (srcs.length > 0) {
      const allConf = confs.every((c) => c);
      p = fitAffineInvDepth(concat(srcs), concat(dsts), allConf ? concat(confs as Float32Array[]) : undefined);
    }
    aligned.push(w.invDepth.map((fr) => applyAffine(fr, p)));
  }

  let total = 0;
  for (const w of windows) total = Math.max(total, w.startFrame - base + w.invDepth.length);
  const out: Float32Array[] = new Array(total);
  const sums: (Float32Array | null)[] = new Array(total).fill(null);
  const wsum = new Float64Array(total);

  for (let wi = 0; wi < windows.length; wi++) {
    const w = windows[wi];
    const len = w.invDepth.length;
    const prev = wi > 0 ? windows[wi - 1] : null;
    const next = wi + 1 < windows.length ? windows[wi + 1] : null;
    const ovIn = prev ? Math.max(0, prev.startFrame + prev.invDepth.length - w.startFrame) : 0;
    const ovOut = next ? Math.max(0, w.startFrame + len - next.startFrame) : 0;
    for (let k = 0; k < len; k++) {
      const g = w.startFrame - base + k;
      let wt = 1;
      if (ovIn > 0 && k < ovIn) wt *= (k + 1) / (ovIn + 1);
      if (ovOut > 0 && k >= len - ovOut) wt *= (len - k) / (ovOut + 1);
      const fr = aligned[wi][k];
      let acc = sums[g];
      if (!acc) { acc = new Float32Array(fr.length); sums[g] = acc; }
      for (let i = 0; i < fr.length; i++) acc[i] += wt * fr[i];
      wsum[g] += wt;
    }
  }
  for (let g = 0; g < total; g++) {
    const acc = sums[g];
    if (!acc) { out[g] = new Float32Array(0); continue; }
    const inv = wsum[g] > 0 ? 1 / wsum[g] : 0;
    for (let i = 0; i < acc.length; i++) acc[i] *= inv;
    out[g] = acc;
  }
  return out;
}

/**
 * Temporal low-pass of per-frame affine params (zero-phase: forward and backward
 * EMA averaged). alpha in (0,1]; 1 = no smoothing.
 */
export function smoothAffineParams(params: AffineParams[], alpha: number): AffineParams[] {
  const n = params.length;
  if (n === 0) return [];
  const al = Math.min(1, Math.max(1e-3, alpha));
  const fwd: AffineParams[] = new Array(n);
  const bwd: AffineParams[] = new Array(n);
  fwd[0] = { ...params[0] };
  for (let i = 1; i < n; i++) {
    fwd[i] = { a: al * params[i].a + (1 - al) * fwd[i - 1].a, b: al * params[i].b + (1 - al) * fwd[i - 1].b };
  }
  bwd[n - 1] = { ...params[n - 1] };
  for (let i = n - 2; i >= 0; i--) {
    bwd[i] = { a: al * params[i].a + (1 - al) * bwd[i + 1].a, b: al * params[i].b + (1 - al) * bwd[i + 1].b };
  }
  return fwd.map((f, i) => ({ a: (f.a + bwd[i].a) / 2, b: (f.b + bwd[i].b) / 2 }));
}

/** Single global robust [p2, p98] over all frames (strided subsample). */
export function computeClipNormalization(frames: Float32Array[], maxSamples = 262144): ClipRange {
  let total = 0;
  for (const f of frames) total += f.length;
  if (total === 0) return { lo: 0, hi: 1 };
  const stride = Math.max(1, Math.floor(total / maxSamples));
  const vals: number[] = [];
  let counter = 0;
  for (const f of frames) {
    for (let i = 0; i < f.length; i++, counter++) {
      if (counter % stride !== 0) continue;
      const v = f[i];
      if (Number.isFinite(v)) vals.push(v);
    }
  }
  if (vals.length === 0) return { lo: 0, hi: 1 };
  const s = Float32Array.from(vals).sort();
  const at = (q: number) => s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))];
  const lo = at(0.02);
  let hi = at(0.98);
  if (!(hi > lo)) hi = lo + 1e-6;
  return { lo, hi };
}

/** Map to [0,1] with clamping using a clip-level range. */
export function normalizeWithClip(frame: Float32Array, lo: number, hi: number): Float32Array {
  const out = new Float32Array(frame.length);
  const inv = 1 / Math.max(1e-12, hi - lo);
  for (let i = 0; i < frame.length; i++) {
    const v = (frame[i] - lo) * inv;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : Number.isFinite(v) ? v : 0;
  }
  return out;
}
