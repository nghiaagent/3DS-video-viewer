/**
 * Geometry of the viewer window ("the frame"), shared by the page and by the export stage renderer so the exported video
 * can reproduce what is on screen.
 *
 * All lengths are CSS pixels at the reference size, a 960 x 480 window; the export scales them uniformly.
 */
import type { WindowPose } from './motionPath.js';

export const FRAME = {
  windowWidth: 960,
  windowHeight: 480,
  cornerRadius: 28,
  /** CSS `perspective` of the stage; the eye sits this far in front of the stage plane, looking at its centre. */
  perspective: 1200,
  /** Drop shadow under the window: blur radius (Gaussian sigma = blur / 2) and opacity. Its offset comes from the pose. */
  shadow: { blur: 80, alpha: 0.85 },
  /** Faint cyan halo around the window. */
  halo: { blur: 90, rgb: [56, 189, 248] as const, alpha: 0.08 },
  /** 1px ring just outside the window. */
  rimAlpha: 0.14,
  /** 1px ring just inside the window (the glass edge). */
  borderAlpha: 0.2,
  /** Soft elliptical ambient glow behind the window. */
  glow: { width: 900, height: 480, blur: 80 },
  background: [6, 7, 10] as const,
} as const;

export type Vec3 = [number, number, number];

const DEG = Math.PI / 180;

/**
 * Directions of the window's own x axis (right) and y axis (down) in stage space, for the CSS transform
 * `rotateY(ry) rotateX(rx)` (the rotateX is applied to the window first). Stage space: x right, y down, z toward the viewer.
 */
export function windowBasis(rotateYDeg: number, rotateXDeg: number): { ex: Vec3; ey: Vec3 } {
  const ry = rotateYDeg * DEG;
  const rx = rotateXDeg * DEG;
  const [cy, sy] = [Math.cos(ry), Math.sin(ry)];
  const [cx, sx] = [Math.cos(rx), Math.sin(rx)];
  return {
    ex: [cy, 0, -sy],
    ey: [sx * sy, cx, sx * cy],
  };
}

/**
 * Where a point of the window, (u, v) px from the window centre, lands on screen (px from the stage centre) after the
 * pose's translation, rotation and the stage perspective. `scale` converts reference px to output px.
 */
export function projectWindowPoint(
  u: number,
  v: number,
  pose: Pick<WindowPose, 'tx' | 'ty' | 'rotateYDeg' | 'rotateXDeg'>,
  scale = 1
): { x: number; y: number; z: number } {
  const { ex, ey } = windowBasis(pose.rotateYDeg, pose.rotateXDeg);
  const P = FRAME.perspective * scale;
  const X = pose.tx * scale + u * ex[0] + v * ey[0];
  const Y = pose.ty * scale + u * ex[1] + v * ey[1];
  const Z = u * ex[2] + v * ey[2];
  const k = P / (P - Z);
  return { x: X * k, y: Y * k, z: Z };
}

/**
 * Inverse of projectWindowPoint: the point (u, v) of the window plane that is seen at screen position (qx, qy), px from
 * the stage centre. This is the ray-plane intersection the export shader performs for every output pixel.
 */
export function unprojectToWindow(
  qx: number,
  qy: number,
  pose: Pick<WindowPose, 'tx' | 'ty' | 'rotateYDeg' | 'rotateXDeg'>,
  scale = 1
): { u: number; v: number; z: number } {
  const { ex, ey } = windowBasis(pose.rotateYDeg, pose.rotateXDeg);
  const P = FRAME.perspective * scale;
  const cx = pose.tx * scale;
  const cy = pose.ty * scale;
  const a11 = ex[0] + (qx / P) * ex[2];
  const a12 = ey[0] + (qx / P) * ey[2];
  const a21 = ex[1] + (qy / P) * ex[2];
  const a22 = ey[1] + (qy / P) * ey[2];
  const det = a11 * a22 - a12 * a21;
  const rx = qx - cx;
  const ry = qy - cy;
  const u = (rx * a22 - ry * a12) / det;
  const v = (a11 * ry - a21 * rx) / det;
  return { u, v, z: u * ex[2] + v * ey[2] };
}
