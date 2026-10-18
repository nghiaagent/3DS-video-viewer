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

  test('the rim fades the picture toward soft white near the edge and leaves the interior alone', async ({ page }) => {
    const r = await renderFlat(page, 200, { soft: 0.8, shift: 1 });
    const mid = at(r, r.width >> 1, r.height >> 1);
    const nearEdge = at(r, r.width >> 1, 3);
    expect(mid).toBeGreaterThan(190);
    expect(nearEdge).toBeGreaterThan(mid + 3); // lifts toward the soft white (238, 238, 241)
    const off = await renderFlat(page, 200, { soft: 0, shift: 1 });
    expect(Math.abs(at(off, off.width >> 1, 3) - 200)).toBeLessThan(3);
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

test('glow takes the picture colour: CSS variables and the exported stage follow it', async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => !!(window as any).__viewerApp?.viewer);

  // red-minus-blue just above the window, inside the glow
  const redness = () =>
    page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.pauseRenderLoop();
      const url = app.renderStageSnapshot(1372, 772, 1);
      const img = new Image();
      await new Promise((res) => { img.onload = res; img.src = url; });
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      const g = c.getContext('2d')!;
      g.drawImage(img, 0, 0);
      const d = g.getImageData(686, 130, 1, 1).data;
      return d[0] - d[2];
    });

  const css0 = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--glow-c1').trim());
  expect(css0).toBe('56, 189, 248'); // the original cyan until a picture or override says otherwise
  const cyan = await redness();

  await page.evaluate(() => (window as any).__viewerApp.setGlowOverride([230, 40, 50]));
  const css = await page.evaluate(() => getComputedStyle(document.documentElement).getPropertyValue('--glow-c1').trim());
  const [r, , b] = css.split(',').map((x) => parseInt(x, 10));
  expect(r).toBeGreaterThan(b);
  expect(await redness()).toBeGreaterThan(cyan + 3);
});
