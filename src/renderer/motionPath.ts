/**
 * Pure motion math shared by the live render loop and (later) the exporter, so an exported clip
 * moves exactly like the viewer does.
 *
 * A motion "offset" is the normalised view direction: x and y each in [-1, 1]
 * (x: -1 = view from the left, +1 = from the right; y: -1/+1 = tilt up/down).
 */

export type MotionMode = 'interactive' | 'orbit' | 'manual';

export interface MotionOffset {
  x: number;
  y: number;
}

export interface OrbitSettings {
  /** Orbits per second. */
  freqHz: number;
  /** Horizontal radius in [0, 1]. */
  amp: number;
  /**
   * Vertical radius as a fraction of the horizontal one. 3DS stereo only carries horizontal information, so vertical
   * motion exposes more filled-in background; keep it smaller than the horizontal radius.
   */
  yRatio: number;
}

export const DEFAULT_ORBIT: OrbitSettings = { freqHz: 0.18, amp: 0.7, yRatio: 0.55 };

const TWO_PI = Math.PI * 2;

function clampUnit(v: number): number {
  return Math.max(-1, Math.min(1, v));
}

/** Point on the orbit (an ellipse) at time tSec. phase0 shifts the starting angle. */
export function getOrbitOffset(tSec: number, settings: OrbitSettings = DEFAULT_ORBIT, phase0 = 0): MotionOffset {
  const phi = tSec * settings.freqHz * TWO_PI + phase0;
  return {
    x: clampUnit(settings.amp * Math.sin(phi)),
    y: clampUnit(settings.amp * settings.yRatio * Math.cos(phi)),
  };
}

/** Same, from an already accumulated phase (the live loop integrates phase so speed changes never jump). */
export function getOrbitOffsetFromPhase(phase: number, amp: number, yRatio: number): MotionOffset {
  return { x: clampUnit(amp * Math.sin(phase)), y: clampUnit(amp * yRatio * Math.cos(phase)) };
}

/** Parameters for WebGLParallaxViewer.renderVideoFrame: viewShift in [0,1] (0.5 = centred) and verticalTilt in [-1,1]. */
export function offsetToRenderParams(o: MotionOffset): { viewShift: number; verticalTilt: number } {
  return {
    viewShift: Math.max(0, Math.min(1, 0.5 + clampUnit(o.x) / 2)),
    verticalTilt: clampUnit(o.y),
  };
}

/** Inverse of offsetToRenderParams for the horizontal axis. */
export function viewShiftToX(viewShift: number): number {
  return clampUnit((viewShift - 0.5) * 2);
}

/** How the surrounding "window" (CSS now, an export shader later) moves for a given view direction. */
export interface WindowPose {
  /** Translation in px at reference scale. */
  tx: number;
  ty: number;
  /** Rotations in degrees. */
  rotateYDeg: number;
  rotateXDeg: number;
  /** Drop-shadow offset in px. */
  shadowX: number;
  shadowY: number;
  /** Ambient glow offset in px. */
  glowX: number;
  glowY: number;
  /** Direction of the glass highlight in degrees. */
  specAngleDeg: number;
  /** The raw offset, for CSS custom properties. */
  x: number;
  y: number;
}

export function getWindowPose(o: MotionOffset): WindowPose {
  const x = clampUnit(o.x);
  const y = clampUnit(o.y);
  return {
    tx: x * 26,
    ty: y * 10,
    rotateYDeg: -x * 8,
    rotateXDeg: y * 5,
    shadowX: -x * 32,
    shadowY: 32 + y * 12,
    glowX: -x * 50,
    glowY: -y * 24,
    specAngleDeg: 135 + x * 35 + y * 15,
    x,
    y,
  };
}
