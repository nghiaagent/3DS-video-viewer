import { test, expect } from '@playwright/test';

/**
 * The export stage renderer must reproduce the page: the same stage, glow, shadow, rim and perspective-tilted window.
 * The page is screenshotted at a pinned view direction and compared with the stage renderer's snapshot of the same
 * picture at the same size. (Measured mean error is about 0.5 / 255; the thresholds leave room for driver differences.)
 */
const STAGE = { width: 1372, height: 772 }; // 1372 - 64 > 960, so the page window is its full 960 x 480

for (const pose of [
  { name: 'tilted with a red glow', x: 0.5, y: 0.3, glow: [220, 40, 60] as [number, number, number] },
  { name: 'tilted right and up', x: 0.6, y: -0.4 },
  { name: 'tilted left and down', x: -0.7, y: 0.385 },
  { name: 'centred', x: 0, y: 0 },
]) {
  test(`stage renderer matches the page: ${pose.name}`, async ({ page }) => {
    await page.setViewportSize(STAGE);
    await page.goto('/');
    await page.waitForFunction(() => !!(window as any).__viewerApp?.viewer);
    // hide everything that is not the stage and the window
    await page.addStyleTag({
      content: 'header, footer, #empty-state-overlay, #drop-overlay, #progress-banner, .glass-pill { display: none !important; }',
    });
    await page.evaluate((p) => {
      const app = (window as any).__viewerApp;
      app.setMotionOverride({ x: p.x, y: p.y });
      if (p.glow) app.setGlowOverride(p.glow);
    }, pose);
    await page.waitForTimeout(400); // let the live loop apply the pose to the page

    const stageUrl = await page.evaluate(
      ({ x, y, w, h }) => {
        const app = (window as any).__viewerApp;
        app.pauseRenderLoop();
        const viewer = app.viewer;
        const EW = 480;
        const EH = 240;
        const sbs = new Uint8Array(EW * 2 * EH * 4);
        for (let yy = 0; yy < EH; yy++) {
          for (let xx = 0; xx < EW; xx++) {
            const inBox = xx >= 160 && xx < 320 && yy >= 80 && yy < 160;
            const check = (Math.floor(xx / 16) + Math.floor(yy / 16)) % 2 === 0;
            const rgb = inBox ? [230, 30, 30] : check ? [30, 120, 230] : [30, 200, 90];
            for (const off of [0, EW]) {
              const i = (yy * EW * 2 + off + xx) * 4;
              sbs[i] = rgb[0];
              sbs[i + 1] = rgb[1];
              sbs[i + 2] = rgb[2];
              sbs[i + 3] = 255;
            }
          }
        }
        viewer.uploadExternalDepthMap(new Uint8Array(EW * EH).fill(128), EW, EH);
        viewer.renderRawRgbaFrame(sbs, EW * 2, EH, 0.5 + x / 2, undefined, y);
        return app.renderStageSnapshot(w, h, 1);
      },
      { x: pose.x, y: pose.y, w: STAGE.width, h: STAGE.height }
    );
    await page.waitForTimeout(200);
    const domPng = (await page.screenshot()).toString('base64');

    const diff = await page.evaluate(
      async ({ domPng, stageUrl }) => {
        const load = (src: string) =>
          new Promise<HTMLImageElement>((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
          });
        const [a, b] = await Promise.all([load(`data:image/png;base64,${domPng}`), load(stageUrl)]);
        const read = (img: HTMLImageElement) => {
          const c = document.createElement('canvas');
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
          const g = c.getContext('2d', { willReadFrequently: true })!;
          g.drawImage(img, 0, 0);
          return g.getImageData(0, 0, c.width, c.height).data;
        };
        const da = read(a);
        const db = read(b);
        let sum = 0;
        let big = 0;
        let n = 0;
        for (let i = 0; i < da.length; i += 4) {
          const d = (Math.abs(da[i] - db[i]) + Math.abs(da[i + 1] - db[i + 1]) + Math.abs(da[i + 2] - db[i + 2])) / 3;
          sum += d;
          if (d > 40) big++;
          n++;
        }
        return { sameSize: a.naturalWidth === b.naturalWidth && a.naturalHeight === b.naturalHeight, mae: sum / n, bigFraction: big / n };
      },
      { domPng, stageUrl }
    );

    expect(diff.sameSize).toBe(true);
    expect(diff.mae).toBeLessThan(3); // out of 255, averaged over the whole stage
    expect(diff.bigFraction).toBeLessThan(0.004); // pixels that differ a lot: only thin edge lines may
  });
}
