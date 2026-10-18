import { describe, it, expect } from 'vitest';
import { FRAME, projectWindowPoint, unprojectToWindow, windowBasis } from '../../src/renderer/frameGeometry.js';
import { getWindowPose } from '../../src/renderer/motionPath.js';

describe('window plane geometry', () => {
  it('an untilted window is just translated (perspective does nothing at z = 0)', () => {
    const pose = { tx: 12, ty: -7, rotateYDeg: 0, rotateXDeg: 0 };
    const p = projectWindowPoint(100, -40, pose, 1.4);
    expect(p.x).toBeCloseTo(12 * 1.4 + 100, 9);
    expect(p.y).toBeCloseTo(-7 * 1.4 - 40, 9);
    const q = unprojectToWindow(p.x, p.y, pose, 1.4);
    expect(q.u).toBeCloseTo(100, 9);
    expect(q.v).toBeCloseTo(-40, 9);
  });

  it('unprojecting a projected point returns the original window coordinates, for any pose', () => {
    const poses = [
      getWindowPose({ x: 0.7, y: 0.385 }),
      getWindowPose({ x: -0.7, y: -0.385 }),
      getWindowPose({ x: 1, y: -1 }),
      getWindowPose({ x: 0.2, y: 0.9 }),
    ];
    const corners: Array<[number, number]> = [
      [-FRAME.windowWidth / 2, -FRAME.windowHeight / 2],
      [FRAME.windowWidth / 2, -FRAME.windowHeight / 2],
      [FRAME.windowWidth / 2, FRAME.windowHeight / 2],
      [-FRAME.windowWidth / 2, FRAME.windowHeight / 2],
      [0, 0],
      [333, -111],
    ];
    for (const pose of poses) {
      for (const scale of [1, 1.4]) {
        for (const [u, v] of corners) {
          const s = projectWindowPoint(u * scale, v * scale, pose, scale);
          const back = unprojectToWindow(s.x, s.y, pose, scale);
          expect(back.u).toBeCloseTo(u * scale, 6);
          expect(back.v).toBeCloseTo(v * scale, 6);
        }
      }
    }
  });

  it('follows the CSS conventions: positive rotateY sends the right edge away, positive rotateX sends the top edge away', () => {
    const right = projectWindowPoint(480, 0, { tx: 0, ty: 0, rotateYDeg: 20, rotateXDeg: 0 });
    expect(right.z).toBeLessThan(0); // away from the viewer
    const left = projectWindowPoint(-480, 0, { tx: 0, ty: 0, rotateYDeg: 20, rotateXDeg: 0 });
    expect(left.z).toBeGreaterThan(0); // toward the viewer
    // farther away means smaller on screen: the right edge ends up closer to the centre than the left edge is far from it
    expect(Math.abs(right.x)).toBeLessThan(Math.abs(left.x));

    const top = projectWindowPoint(0, -240, { tx: 0, ty: 0, rotateYDeg: 0, rotateXDeg: 20 });
    expect(top.z).toBeLessThan(0);
  });

  it('basis vectors are unit length and perpendicular', () => {
    const { ex, ey } = windowBasis(8, -5);
    const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    expect(dot(ex, ex)).toBeCloseTo(1, 9);
    expect(dot(ey, ey)).toBeCloseTo(1, 9);
    expect(dot(ex, ey)).toBeCloseTo(0, 9);
  });
});
