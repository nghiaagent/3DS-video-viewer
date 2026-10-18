#!/usr/bin/env node
/**
 * Measure temporal depth flicker on static-background pixels.
 *
 * Usage:
 *   node scripts/measure-depth-flicker.mjs <depth.f32> <video> <w> <h> [--thresh=6] [--left-eye-width=480]
 *
 * Input format of <depth.f32>: raw little-endian float32, frames concatenated,
 * each frame w*h values (row-major). Frame count = filesize / (4*w*h).
 * Depth frame i must correspond to video frame i (same fps, from t=0).
 *
 * Video luma is decoded with ffmpeg: for side-by-side video, the left eye is
 * cropped (x 0..left-eye-width-1, default = full height crop of width 480 for
 * 960x240 3DS clips), then scaled to w x h.
 * Pass --left-eye-width=0 to disable cropping.
 *
 * Example:
 *   node scripts/measure-depth-flicker.mjs depth.f32 \
 *     "$HOME/Downloads/3DS vids/DCIM/101NIN03/HNI_0062.AVI" 480 240
 *
 * Prints JSON with meanTemporalStd, meanPumping, frameToFrameMAD, maskedPixels.
 */
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';

const FFMPEG = fs.existsSync('/opt/homebrew/bin/ffmpeg') ? '/opt/homebrew/bin/ffmpeg' : 'ffmpeg';

const args = process.argv.slice(2);
const flags = Object.fromEntries(
  args.filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('=')),
);
const pos = args.filter((a) => !a.startsWith('--'));
if (pos.length < 4) {
  console.error('usage: measure-depth-flicker.mjs <depth.f32> <video> <w> <h> [--thresh=6] [--left-eye-width=480]');
  process.exit(1);
}
const [depthPath, videoPath] = pos;
const w = parseInt(pos[2], 10);
const h = parseInt(pos[3], 10);
const thresh = parseFloat(flags.thresh ?? '6');
const cropW = parseInt(flags['left-eye-width'] ?? '480', 10);

const raw = fs.readFileSync(depthPath);
const all = new Float32Array(raw.buffer, raw.byteOffset, Math.floor(raw.byteLength / 4));
const T = Math.floor(all.length / (w * h));
if (T < 2) throw new Error(`need >=2 frames, got ${T}`);
const depthFrames = Array.from({ length: T }, (_, t) => all.subarray(t * w * h, (t + 1) * w * h));

const vf = [cropW > 0 ? `crop=${cropW}:ih:0:0` : null, `scale=${w}:${h}`, 'format=gray'].filter(Boolean).join(',');
const luma = execFileSync(
  FFMPEG,
  ['-v', 'error', '-i', videoPath, '-vf', vf, '-frames:v', String(T), '-f', 'rawvideo', '-pix_fmt', 'gray', '-'],
  { maxBuffer: 1 << 30 },
);
const lumaT = Math.floor(luma.length / (w * h));
if (lumaT < T) console.error(`warning: video gave ${lumaT} frames, depth has ${T}; truncating`);
const F = Math.min(T, lumaT);
const lumaFrames = Array.from({ length: F }, (_, t) => new Uint8Array(luma.buffer, luma.byteOffset + t * w * h, w * h));
const dFrames = depthFrames.slice(0, F);

// --- metrics (mirrors src/depth/temporalStability.ts) ---
const n = w * h;
const mask = new Uint8Array(n).fill(1);
for (let t = 1; t < F; t++) {
  for (let i = 0; i < n; i++) {
    if (mask[i] && Math.abs(lumaFrames[t][i] - lumaFrames[t - 1][i]) > thresh) mask[i] = 0;
  }
}
const idx = [];
for (let i = 0; i < n; i++) if (mask[i]) idx.push(i);
const m = idx.length;
if (m === 0) throw new Error('static mask is empty; raise --thresh');

const means = new Float64Array(F);
for (let t = 0; t < F; t++) {
  let s = 0;
  for (let k = 0; k < m; k++) s += dFrames[t][idx[k]];
  means[t] = s / m;
}
const mm = means.reduce((a, b) => a + b, 0) / F;
const meanPumping = Math.sqrt(means.reduce((a, b) => a + (b - mm) ** 2, 0) / F);
let stdSum = 0;
let madSum = 0;
for (const i of idx) {
  let s = 0;
  for (let t = 0; t < F; t++) s += dFrames[t][i];
  const mu = s / F;
  let ss = 0;
  for (let t = 0; t < F; t++) ss += (dFrames[t][i] - mu) ** 2;
  stdSum += Math.sqrt(ss / F);
  for (let t = 1; t < F; t++) madSum += Math.abs(dFrames[t][i] - dFrames[t - 1][i]);
}
console.log(JSON.stringify({
  frames: F,
  maskedPixels: m,
  maskedFraction: +(m / n).toFixed(4),
  meanTemporalStd: stdSum / m,
  meanPumping,
  frameToFrameMAD: madSum / (m * (F - 1)),
}, null, 2));
