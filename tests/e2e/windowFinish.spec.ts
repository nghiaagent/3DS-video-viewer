import { test, expect } from '@playwright/test';

// Renders a flat-colour SBS frame in the 3D mesh mode and reads the canvas back.
async function renderFlat(page: any, value: number, opts: { soft?: number; grain?: number; seed?: number; shift?: number }) {
  return page.evaluate(
    ({ value, opts }: any) => {
      const v = (window as any).__viewerApp.viewer;
      const w = 128, h = 64;
      const buf = new Uint8Array(w * 2 * h * 4).fill(value);
      for (let i = 3; i < buf.length; i += 4) buf[i] = 255;
      v.setRenderMode(0);
      v.setEdgeSoftness(opts.soft ?? 0.8);
      v.setGrain(opts.grain ?? 0);
      v.setGrainSeed(opts.seed ?? 0);
      v.renderRawRgbaFrame(buf, w * 2, h, opts.shift ?? 0.5, false, 0);
      const gl = v.getGLContext();
      const c = v.getCanvas();
      const px = new Uint8Array(c.width * c.height * 4);
      gl.readPixels(0, 0, c.width, c.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
      return { px: Array.from(px), width: c.width, height: c.height };
    },
    { value, opts }
  );
}

const at = (r: any, x: number, y: number) => r.px[(y * r.width + x) * 4];

test.describe('window finish', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await page.waitForFunction(() => !!(window as any).__viewerApp?.viewer);
  });

  test('the rim dims the picture near the edge and leaves the interior alone', async ({ page }) => {
    const r = await renderFlat(page, 200, { soft: 0.8, shift: 1 });
    const mid = at(r, r.width >> 1, r.height >> 1);
    const nearEdge = at(r, r.width >> 1, 3);
    expect(mid).toBeGreaterThan(190);
    expect(nearEdge).toBeLessThan(mid * 0.95);
    const off = await renderFlat(page, 200, { soft: 0, shift: 1 });
    expect(at(off, off.width >> 1, 3)).toBeGreaterThan(190);
  });

  test('grain is mean-neutral, seeded, and absent on black', async ({ page }) => {
    const a = await renderFlat(page, 128, { grain: 0.03, seed: 5 });
    const b = await renderFlat(page, 128, { grain: 0.03, seed: 5 });
    const c = await renderFlat(page, 128, { grain: 0.03, seed: 6 });
    expect(a.px).toEqual(b.px);
    expect(a.px).not.toEqual(c.px);
    const vals: number[] = [];
    for (let y = a.height / 4; y < (a.height * 3) / 4; y += 2) for (let x = a.width / 4; x < (a.width * 3) / 4; x += 2) vals.push(at(a, x, y));
    const mean = vals.reduce((s, x) => s + x, 0) / vals.length;
    const std = Math.sqrt(vals.reduce((s, x) => s + (x - mean) ** 2, 0) / vals.length);
    expect(Math.abs(mean - 128)).toBeLessThan(1.5);
    expect(std).toBeGreaterThan(1);
    const black = await renderFlat(page, 0, { grain: 0.03, seed: 5 });
    expect(at(black, black.width >> 1, black.height >> 1)).toBe(0);
  });
});
