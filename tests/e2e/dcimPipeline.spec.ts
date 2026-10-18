import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const FFPROBE_BIN = fs.existsSync('/opt/homebrew/bin/ffprobe')
  ? '/opt/homebrew/bin/ffprobe'
  : 'ffprobe';

const SCRATCH_DIR = '/Users/nghiaagent/.gemini/antigravity/brain/a683cc2e-6fb4-45b1-933b-9fcd06ec25f2/scratch';

test.describe('Spatial 3DS Video Viewer E2E Suite (HNI_0062.AVI)', () => {
  test.beforeEach(async ({ page }) => {
    fs.mkdirSync(SCRATCH_DIR, { recursive: true });
    page.on('console', (msg) => console.log(`[BROWSER ${msg.type()}]`, msg.text()));
    page.on('pageerror', (err) => console.error('[BROWSER ERROR]', err.message));
  });

  test('1. DCIM API Discovery, Speed Slider, Circular Export Rings & Auto-Convergence', async ({ page }) => {
    await page.goto('/');

    // 1. Verify Cross-Origin Isolation for SharedArrayBuffer & WebAssembly SIMD
    const isIsolated = await page.evaluate(() => window.crossOriginIsolated);
    expect(isIsolated).toBe(true);

    // 2. Verify DCIM catalog endpoint discovers local recordings without exposing them in public UI
    const dcimResponse = await page.request.get('/api/dcim/list');
    expect(dcimResponse.ok()).toBe(true);
    const dcimList = await dcimResponse.json();
    console.log(`Discovered ${dcimList.length} local DCIM recordings via private local dev server`);
    expect(dcimList.length).toBeGreaterThanOrEqual(10);

    // 3. Verify Apple Vision-style Spatial Window & WebGL2 Canvas are rendered
    const spatialWindow = page.locator('#spatial-window');
    await expect(spatialWindow).toBeVisible();

    const canvas = page.locator('#parallax-canvas');
    await expect(canvas).toBeVisible();

    const canvasDims = await canvas.evaluate((el: HTMLCanvasElement) => ({
      width: el.width,
      height: el.height,
    }));
    expect(canvasDims.width).toBe(960);
    expect(canvasDims.height).toBe(480);

    // 4. Default motion is Pointer & Tilt: nothing sways on its own, and the Orbit controls are hidden
    const speedControl = page.locator('#speed-dock-control');
    const sizeControl = page.locator('#orbit-size-dock-control');
    const freqSlider = page.locator('#freq-slider');
    const motionModeSelect = page.locator('#motion-mode-select');
    await expect(motionModeSelect).toHaveValue('interactive');
    await expect(speedControl).toBeHidden();
    await expect(sizeControl).toBeHidden();

    const shiftReadout = page.locator('#telemetry-shift');
    const shiftAtLoad = await shiftReadout.textContent();
    await page.waitForTimeout(1500);
    expect(await shiftReadout.textContent()).toBe(shiftAtLoad);
    expect(shiftAtLoad).toBe('0.50');

    // Orbit shows Speed and Size and genuinely moves; Manual hides them again
    await motionModeSelect.selectOption('orbit');
    await expect(speedControl).toBeVisible();
    await expect(sizeControl).toBeVisible();
    await expect(freqSlider).toBeVisible();
    await page.waitForFunction(() => {
      const el = document.getElementById('telemetry-shift');
      return !!el && el.textContent !== '0.50';
    }, null, { timeout: 5000 });

    await motionModeSelect.selectOption('manual');
    await expect(speedControl).toBeHidden();
    await expect(sizeControl).toBeHidden();

    await motionModeSelect.selectOption('interactive');
    await expect(speedControl).toBeHidden();

    // 5. Verify Circular Progress Rings are attached to export buttons with zero text status clutter
    const mp4Ring = page.locator('#export-mp4-btn .progress-ring');
    await expect(mp4Ring).toBeAttached();
    const webmRing = page.locator('#export-webm-btn .progress-ring');
    await expect(webmRing).toBeAttached();

    // 6. Import primary real 3DS video: HNI_0062.AVI
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      await app.importVideo({
        url: '/api/dcim/file?path=101NIN03/HNI_0062.AVI',
        name: 'HNI_0062.AVI',
      });
    });

    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return app && app.videoElement && app.videoElement.readyState >= 2 && app.videoElement.currentTime > 0;
    }, { timeout: 45_000 });

    // 7. Verify Auto-Calibrated Global Baseline Convergence locks around -70 to -72px via 3DS1 metadata
    await page.waitForTimeout(600); // Allow 4-frame auto-convergence loop
    const telemetry = await page.evaluate(() => {
      const app = (window as any).__viewerApp;
      return app.viewer.getTelemetry();
    });
    console.log('HNI_0062.AVI Auto-Convergence Telemetry:', telemetry);
    expect(telemetry.baseDisp).toBeLessThanOrEqual(-60);
    expect(telemetry.baseDisp).toBeGreaterThanOrEqual(-80);

    // 8. Verify upright render (flipY: true) produces active non-black center pixels
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

  test('2. 2.5D Parallax View Synthesis & Diagnostic Visual Inspection Screenshots', async ({ page }) => {
    await page.goto('/');

    // Import primary test video HNI_0062.AVI
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      await app.importVideo({
        url: '/api/dcim/file?path=101NIN03/HNI_0062.AVI',
        name: 'HNI_0062.AVI',
      });
    });

    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return app && app.videoElement && app.videoElement.readyState >= 2 && app.videoElement.currentTime > 0;
    }, { timeout: 45_000 });

    // Seek and pause at 0.8s where DJ, stage lighting tubes, and foreground subject have rich depth
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.videoElement.pause();
      app.videoElement.currentTime = 0.8;
      await new Promise((r) => setTimeout(r, 400));
    });

    const canvasLocator = page.locator('#parallax-canvas');

    // 1. Capture Diagnostic Screenshot: Left Eye Perspective (t = 0.08)
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.setManualShift(0.08);
      await app.viewer.renderVideoFrame(app.videoElement, 0.08);
    });
    await page.waitForTimeout(150);
    const leftScreenshotPath = path.join(SCRATCH_DIR, 'hni_0062_left_t08.png');
    await canvasLocator.screenshot({ path: leftScreenshotPath });
    console.log(`Saved Left perspective screenshot: ${leftScreenshotPath}`);
    expect(fs.existsSync(leftScreenshotPath)).toBe(true);

    // 2. Capture Diagnostic Screenshot: Center View (t = 0.50)
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.setManualShift(0.50);
      await app.viewer.renderVideoFrame(app.videoElement, 0.50);
    });
    await page.waitForTimeout(150);
    const centerScreenshotPath = path.join(SCRATCH_DIR, 'hni_0062_center_t50.png');
    await canvasLocator.screenshot({ path: centerScreenshotPath });
    console.log(`Saved Center perspective screenshot: ${centerScreenshotPath}`);
    expect(fs.existsSync(centerScreenshotPath)).toBe(true);

    // 3. Capture Diagnostic Screenshot: Right Eye Perspective (t = 0.92)
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.setManualShift(0.92);
      await app.viewer.renderVideoFrame(app.videoElement, 0.92);
    });
    await page.waitForTimeout(150);
    const rightScreenshotPath = path.join(SCRATCH_DIR, 'hni_0062_right_t92.png');
    await canvasLocator.screenshot({ path: rightScreenshotPath });
    console.log(`Saved Right perspective screenshot: ${rightScreenshotPath}`);
    expect(fs.existsSync(rightScreenshotPath)).toBe(true);

    // 4. Capture Diagnostic Screenshot: Disparity Heatmap Mode (renderMode = 1)
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.viewer.setRenderMode(1);
      await app.viewer.renderVideoFrame(app.videoElement, 0.50);
    });
    await page.waitForTimeout(150);
    const heatmapScreenshotPath = path.join(SCRATCH_DIR, 'hni_0062_heatmap_mode1.png');
    await canvasLocator.screenshot({ path: heatmapScreenshotPath });
    console.log(`Saved Disparity Heatmap screenshot: ${heatmapScreenshotPath}`);
    expect(fs.existsSync(heatmapScreenshotPath)).toBe(true);

    // Restore render mode to 0
    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.viewer.setRenderMode(0);
      await app.viewer.renderVideoFrame(app.videoElement, 0.50);
    });

    // 5. Measure genuine parallax pixel difference between Left and Right viewpoints
    const shiftDiff = await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      app.pauseRenderLoop();
      const gl = app.viewer.getGLContext() as WebGL2RenderingContext;

      await app.viewer.renderVideoFrame(app.videoElement, 0.08);
      const pLeft = new Uint8Array(960 * 480 * 4);
      gl.readPixels(0, 0, 960, 480, gl.RGBA, gl.UNSIGNED_BYTE, pLeft);

      await app.viewer.renderVideoFrame(app.videoElement, 0.92);
      const pRight = new Uint8Array(960 * 480 * 4);
      gl.readPixels(0, 0, 960, 480, gl.RGBA, gl.UNSIGNED_BYTE, pRight);

      let delta = 0;
      for (let i = 0; i < pLeft.length; i += 4) {
        delta += Math.abs(pLeft[i] - pRight[i]) +
                 Math.abs(pLeft[i + 1] - pRight[i + 1]) +
                 Math.abs(pLeft[i + 2] - pRight[i + 2]);
      }
      app.resumeRenderLoop();
      return delta;
    });

    console.log(`Left vs Right viewpoint total pixel delta: ${shiftDiff}`);
    expect(shiftDiff).toBeGreaterThan(50_000);
  });

  test('3. Raw Dual-Stream 3DS .AVI Import and Transcoding (HNI_0062.AVI)', async ({ page }) => {
    await page.goto('/');

    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      await app.importVideo({
        url: '/api/dcim/file?path=101NIN03/HNI_0062.AVI',
        name: 'HNI_0062.AVI',
      });
    });

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

    await page.evaluate(async () => {
      const app = (window as any).__viewerApp;
      await app.importVideo({
        url: '/api/dcim/file?path=101NIN03/HNI_0062.AVI',
        name: 'HNI_0062.AVI',
      });
    });

    await page.waitForFunction(() => {
      const app = (window as any).__viewerApp;
      return app && app.videoElement && app.videoElement.readyState >= 2 && app.videoElement.currentTime > 0;
    }, { timeout: 45_000 });

    const testResultsDir = path.resolve(process.cwd(), 'test-results');
    fs.mkdirSync(testResultsDir, { recursive: true });

    const exportAndSave = async (format: 'mp4' | 'webm', frame: 'stage' | 'content' = 'content', fps = 30): Promise<string> => {
      const exportData = await page.evaluate(async ({ fmt, frame, fps }) => {
        const app = (window as any).__viewerApp;
        const res = await app.exportOffline(fmt, {
          maxDurationSec: 0.5,
          fps,
          frame,
          triggerDownload: false,
        });

        const arrayBuf = await res.blob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuf);
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
      }, { fmt: format, frame, fps });

      const filePath = path.join(testResultsDir, `e2e_exported_${frame}.${format}`);
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

    // 3. Full-stage exports (the default): the whole framed window at 1920x1080, 60 fps
    const probe = (file: string) =>
      JSON.parse(
        execFileSync(FFPROBE_BIN, ['-v', 'quiet', '-print_format', 'json', '-show_streams', file], { encoding: 'utf-8' })
      ).streams.find((st: any) => st.codec_type === 'video');

    const stageMp4 = probe(await exportAndSave('mp4', 'stage', 60));
    expect(stageMp4.codec_name).toBe('h264');
    expect(stageMp4.width).toBe(1920);
    expect(stageMp4.height).toBe(1080);
    expect(stageMp4.r_frame_rate).toBe('60/1');
    expect(Number(stageMp4.level)).toBeGreaterThanOrEqual(42); // 1080p60 needs level 4.2
    expect(Number(stageMp4.nb_frames)).toBe(30); // 0.5 s at 60 fps

    const stageWebm = probe(await exportAndSave('webm', 'stage', 60));
    expect(stageWebm.codec_name).toBe('vp9');
    expect(stageWebm.width).toBe(1920);
    expect(stageWebm.height).toBe(1080);
  });

  test('5. Subpath URL Resolution & In-Browser FFmpeg Subpath Loading', async ({ page }) => {
    await page.goto('/');

    const resolutionResults = await page.evaluate(async () => {
      const ghPagesBase = 'https://nghiaagent.github.io/3DS-video-viewer/';
      const ghPagesIndex = 'https://nghiaagent.github.io/3DS-video-viewer/index.html';

      const coreFromBase = new URL('./ffmpeg/ffmpeg-core.js', ghPagesBase).href;
      const wasmFromBase = new URL('./ffmpeg/ffmpeg-core.wasm', ghPagesBase).href;
      const coreFromIndex = new URL('./ffmpeg/ffmpeg-core.js', ghPagesIndex).href;

      const currentCore = new URL('./ffmpeg/ffmpeg-core.js', window.location.href).href;
      const currentWasm = new URL('./ffmpeg/ffmpeg-core.wasm', window.location.href).href;

      const resCore = await fetch(currentCore, { method: 'HEAD' });
      const resWasm = await fetch(currentWasm, { method: 'HEAD' });

      return {
        coreFromBase,
        wasmFromBase,
        coreFromIndex,
        coreStatus: resCore.status,
        wasmStatus: resWasm.status,
      };
    });

    expect(resolutionResults.coreFromBase).toBe('https://nghiaagent.github.io/3DS-video-viewer/ffmpeg/ffmpeg-core.js');
    expect(resolutionResults.wasmFromBase).toBe('https://nghiaagent.github.io/3DS-video-viewer/ffmpeg/ffmpeg-core.wasm');
    expect(resolutionResults.coreFromIndex).toBe('https://nghiaagent.github.io/3DS-video-viewer/ffmpeg/ffmpeg-core.js');
    expect(resolutionResults.coreStatus).toBe(200);
    expect(resolutionResults.wasmStatus).toBe(200);
  });
});
