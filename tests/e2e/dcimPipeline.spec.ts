import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const FFPROBE_BIN = fs.existsSync('/opt/homebrew/bin/ffprobe')
  ? '/opt/homebrew/bin/ffprobe'
  : 'ffprobe';

test.describe('3DS Stereoscopic Parallax Viewer E2E Suite', () => {
  test.beforeEach(async ({ page }) => {
    page.on('console', (msg) => console.log(`[BROWSER ${msg.type()}]`, msg.text()));
    page.on('pageerror', (err) => console.error('[BROWSER ERROR]', err.message));
  });

  test('1. DCIM Library Discovery & Initial SBS Video Playback', async ({ page }) => {
    await page.goto('/');

    // 1. Verify Cross-Origin Isolation for SharedArrayBuffer
    const isIsolated = await page.evaluate(() => window.crossOriginIsolated);
    expect(isIsolated).toBe(true);

    // 2. Verify DCIM library populated all 36 recordings from local DCIM
    await page.waitForSelector('#dcim-recordings-list .dcim-item', { timeout: 15_000 });
    const dcimCount = await page.locator('#dcim-recordings-list .dcim-item').count();
    console.log(`Discovered ${dcimCount} local DCIM recordings`);
    expect(dcimCount).toBeGreaterThanOrEqual(36);

    // 3. Verify bundled samples manifest loaded
    const sampleCount = await page.locator('#bundled-samples-list .dcim-item').count();
    expect(sampleCount).toBeGreaterThanOrEqual(3);

    // 4. Verify WebGL2 Canvas resolution and active playback
    const canvas = page.locator('#parallax-canvas');
    await expect(canvas).toBeVisible();

    const canvasDims = await canvas.evaluate((el: HTMLCanvasElement) => ({
      width: el.width,
      height: el.height,
    }));
    expect(canvasDims.width).toBe(960);
    expect(canvasDims.height).toBe(480);

    // Wait for video playback and verify non-black canvas pixels
    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return app && app.videoElement && app.videoElement.readyState >= 2 && app.videoElement.currentTime > 0;
    }, { timeout: 15_000 });

    const pixelCheck = await page.evaluate(() => {
      const app = (window as any).__viewerApp;
      const gl = app.viewer.getGLContext() as WebGL2RenderingContext;
      const p = new Uint8Array(4);
      gl.readPixels(480, 240, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, p);
      return { r: p[0], g: p[1], b: p[2], a: p[3] };
    });

    console.log('Center canvas pixel sample:', pixelCheck);
    expect(pixelCheck.r + pixelCheck.g + pixelCheck.b).toBeGreaterThan(0);
  });

  test('2. GPU & Wasm SIMD Disparity Backends with Perspective Shifting', async ({ page }) => {
    await page.goto('/');

    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return app && app.videoElement && app.videoElement.readyState >= 2 && app.videoElement.currentTime > 0;
    }, { timeout: 15_000 });

    // 1. Test Perspective Shift on GPU backend (Left vs Right shift difference)
    const shiftResult = await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.pauseRenderLoop();
      const gl = app.viewer.getGLContext() as WebGL2RenderingContext;

      // Render Left perspective (0.0)
      await app.viewer.renderVideoFrame(app.videoElement, 0.0);
      const pLeft = new Uint8Array(960 * 480 * 4);
      gl.readPixels(0, 0, 960, 480, gl.RGBA, gl.UNSIGNED_BYTE, pLeft);

      // Render Right perspective (1.0)
      await app.viewer.renderVideoFrame(app.videoElement, 1.0);
      const pRight = new Uint8Array(960 * 480 * 4);
      gl.readPixels(0, 0, 960, 480, gl.RGBA, gl.UNSIGNED_BYTE, pRight);

      // Calculate total pixel difference across frames
      let diff = 0;
      for (let i = 0; i < pLeft.length; i += 4) {
        diff += Math.abs(pLeft[i] - pRight[i]) +
                Math.abs(pLeft[i + 1] - pRight[i + 1]) +
                Math.abs(pLeft[i + 2] - pRight[i + 2]);
      }

      app.resumeRenderLoop();
      return { diff };
    });

    console.log('Perspective shift total pixel delta:', shiftResult.diff);
    expect(shiftResult.diff).toBeGreaterThan(1000);

    // 2. Switch to Wasm SIMD backend and verify telemetry
    const wasmTelemetry = await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.viewer.setBackend('wasm');
      const tel = await app.viewer.renderVideoFrame(app.videoElement, 0.5);
      return { backend: tel.backend, dispMs: tel.disparityMs };
    });

    expect(wasmTelemetry.backend).toBe('wasm');
    expect(wasmTelemetry.dispMs).toBeGreaterThanOrEqual(0);

    // 3. Test Disparity Depth Heatmap mode
    const heatmapResult = await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.viewer.setRenderMode(1); // Mode 1: Heatmap
      await app.viewer.renderVideoFrame(app.videoElement, 0.5);
      const gl = app.viewer.getGLContext() as WebGL2RenderingContext;
      const p = new Uint8Array(960 * 480 * 4);
      gl.readPixels(0, 0, 960, 480, gl.RGBA, gl.UNSIGNED_BYTE, p);
      app.viewer.setRenderMode(0); // restore

      let rSum = 0, gSum = 0, bSum = 0;
      for (let i = 0; i < p.length; i += 16) {
        rSum += p[i];
        gSum += p[i + 1];
        bSum += p[i + 2];
      }
      return { rSum, gSum, bSum };
    });

    expect(heatmapResult.rSum + heatmapResult.gSum + heatmapResult.bSum).toBeGreaterThan(0);
  });

  test('3. Raw Dual-Stream 3DS .AVI Import and Transcoding', async ({ page }) => {
    await page.goto('/');

    // Wait for DCIM list to load
    await page.waitForSelector('#dcim-recordings-list .dcim-item', { timeout: 15_000 });

    // Click to load HNI_0065.AVI (compact real 3DS recording ~68KB)
    const aviLoadBtn = page.locator('button[data-dcim-path="101NIN03/HNI_0065.AVI"]');
    await expect(aviLoadBtn).toBeVisible();
    await aviLoadBtn.click();

    // Verify FFmpeg progress banner appears or finishes
    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return (
        app &&
        app.videoElement &&
        app.videoElement.videoWidth === 960 &&
        app.videoElement.videoHeight === 240
      );
    }, { timeout: 45_000 });

    const videoDimensions = await page.evaluate(() => {
      const app = (window as any).__viewerApp;
      return {
        width: app.videoElement.videoWidth,
        height: app.videoElement.videoHeight,
      };
    });

    expect(videoDimensions.width).toBe(960);
    expect(videoDimensions.height).toBe(240);
  });

  test('4. Hardcoded GPU Video Exports (.mp4 H.264 & .webm VP9) + ffprobe Verification', async ({ page }) => {
    await page.goto('/');

    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return app && app.videoElement && app.videoElement.readyState >= 2 && app.videoElement.currentTime > 0;
    }, { timeout: 15_000 });

    const testResultsDir = path.resolve(process.cwd(), 'test-results');
    fs.mkdirSync(testResultsDir, { recursive: true });

    // Helper to export and retrieve binary buffer from page context
    const exportAndSave = async (format: 'mp4' | 'webm'): Promise<string> => {
      const exportData = await page.evaluate(async (fmt) => {
        const app = (window as any).__viewerApp;
        const res = await app.exportOffline(fmt, {
          maxDurationSec: 0.5,
          fps: 30,
          triggerDownload: false,
        });

        const arrayBuf = await res.blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuf);
        // Convert to base64 in chunks
        let binary = '';
        const chunkSize = 8192;
        for (let i = 0; i < bytes.length; i += chunkSize) {
          const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
          binary += String.fromCharCode.apply(null, Array.from(chunk));
        }

        return {
          base64: btoa(binary),
          codec: res.codec,
          filename: res.filename,
          width: res.width,
          height: res.height,
        };
      }, format);

      const filePath = path.join(testResultsDir, `e2e_exported.${format}`);
      fs.writeFileSync(filePath, Buffer.from(exportData.base64, 'base64'));
      return filePath;
    };

    // 1. Test .mp4 (H.264)
    console.log('\nExporting H.264 .mp4...');
    const mp4Path = await exportAndSave('mp4');
    expect(fs.existsSync(mp4Path)).toBe(true);
    const mp4Stat = fs.statSync(mp4Path);
    console.log(`Exported .mp4 size: ${mp4Stat.size} bytes`);
    expect(mp4Stat.size).toBeGreaterThan(1000);

    // Verify .mp4 with ffprobe
    const mp4ProbeOutput = execFileSync(
      FFPROBE_BIN,
      ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', mp4Path],
      { encoding: 'utf-8' }
    );
    const mp4Metadata = JSON.parse(mp4ProbeOutput);
    const mp4VideoStream = mp4Metadata.streams.find((s: any) => s.codec_type === 'video');
    expect(mp4VideoStream).toBeDefined();
    expect(mp4VideoStream.codec_name).toBe('h264');
    expect(mp4VideoStream.width).toBe(960);
    expect(mp4VideoStream.height).toBe(480);
    console.log(`Verified MP4 with ffprobe: codec=${mp4VideoStream.codec_name}, ${mp4VideoStream.width}x${mp4VideoStream.height}`);

    // 2. Test .webm (VP9)
    console.log('\nExporting VP9 .webm...');
    const webmPath = await exportAndSave('webm');
    expect(fs.existsSync(webmPath)).toBe(true);
    const webmStat = fs.statSync(webmPath);
    console.log(`Exported .webm size: ${webmStat.size} bytes`);
    expect(webmStat.size).toBeGreaterThan(1000);

    // Verify .webm with ffprobe
    const webmProbeOutput = execFileSync(
      FFPROBE_BIN,
      ['-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', webmPath],
      { encoding: 'utf-8' }
    );
    const webmMetadata = JSON.parse(webmProbeOutput);
    const webmVideoStream = webmMetadata.streams.find((s: any) => s.codec_type === 'video');
    expect(webmVideoStream).toBeDefined();
    expect(webmVideoStream.codec_name).toBe('vp9');
    expect(webmVideoStream.width).toBe(960);
    expect(webmVideoStream.height).toBe(480);
    console.log(`Verified WebM with ffprobe: codec=${webmVideoStream.codec_name}, ${webmVideoStream.width}x${webmVideoStream.height}`);
  });
});
