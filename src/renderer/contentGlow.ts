/**
 * The ambient glow behind the window takes its colour from the picture, so the frame feels lit by the scene.
 * Pure functions, shared by the live page and the exporter.
 */
export type Rgb = [number, number, number];

export interface GlowState {
  /** Inner and outer gradient colours, 0..255. */
  c1: Rgb;
  c2: Rgb;
  /** Their opacities. */
  a1: number;
  a2: number;
}

/** The cyan / indigo glow; used for black or unreadable frames and before the first measurement. */
export const DEFAULT_GLOW: GlowState = {
  c1: [56, 189, 248],
  c2: [129, 140, 248],
  a1: 0.32,
  a2: 0.16,
};

const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const smoothstep = (e0: number, e1: number, x: number) => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};
const mixRgb = (a: Rgb, b: Rgb, t: number): Rgb => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];

/** Glow for the picture's average colour (0..255 per channel). Dark or invalid input fades smoothly to the default. */
export function glowColors(avg: Rgb | null | undefined): GlowState {
  if (!avg || avg.some((v) => !Number.isFinite(v))) return DEFAULT_GLOW;
  const brightness = Math.max(avg[0], avg[1], avg[2]) / 255;
  const luma = (0.299 * avg[0] + 0.587 * avg[1] + 0.114 * avg[2]);
  // a little more saturated than the picture, scaled so the strongest channel is full
  let sat = avg.map((v) => Math.max(0, luma + (v - luma) * 1.3)) as Rgb;
  const peak = Math.max(sat[0], sat[1], sat[2]);
  if (peak < 1e-6) return DEFAULT_GLOW;
  sat = sat.map((v) => (v / peak) * 255) as Rgb;

  const a1 = DEFAULT_GLOW.a1 * clamp(brightness, 0.5, 1);
  const own: GlowState = { c1: sat, c2: mixRgb(sat, DEFAULT_GLOW.c2, 0.3), a1, a2: a1 / 2 };
  const w = smoothstep(0.02, 0.12, brightness); // 0 near black: no threshold jump
  return {
    c1: mixRgb(DEFAULT_GLOW.c1, own.c1, w),
    c2: mixRgb(DEFAULT_GLOW.c2, own.c2, w),
    a1: lerp(DEFAULT_GLOW.a1, own.a1, w),
    a2: lerp(DEFAULT_GLOW.a2, own.a2, w),
  };
}

/** One step of exponential smoothing toward `target`, time constant `tau` seconds. */
export function smoothGlow(prev: GlowState, target: GlowState, dtSec: number, tau = 0.9): GlowState {
  const k = 1 - Math.exp(-Math.max(0, dtSec) / tau);
  return {
    c1: mixRgb(prev.c1, target.c1, k),
    c2: mixRgb(prev.c2, target.c2, k),
    a1: lerp(prev.a1, target.a1, k),
    a2: lerp(prev.a2, target.a2, k),
  };
}

/** The window's edge glow: the glow's colour lightened toward soft white, so a pink picture gets a pale pink edge. */
export function edgeGlowRgb(g: GlowState): Rgb {
  const white: Rgb = [238, 238, 241];
  const c = mixRgb(g.c1, white, 0.55);
  return [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])];
}

/** Rounds to what is worth writing to CSS (whole colour steps, alpha in 0.005 steps) so the blurred layer repaints rarely. */
export function quantizeGlow(g: GlowState): GlowState {
  const q = (c: Rgb): Rgb => [Math.round(c[0]), Math.round(c[1]), Math.round(c[2])];
  const qa = (a: number) => Math.round(a / 0.005) * 0.005;
  return { c1: q(g.c1), c2: q(g.c2), a1: qa(g.a1), a2: qa(g.a2) };
}

export function glowEquals(a: GlowState, b: GlowState): boolean {
  return a.c1.every((v, i) => v === b.c1[i]) && a.c2.every((v, i) => v === b.c2[i]) && a.a1 === b.a1 && a.a2 === b.a2;
}
