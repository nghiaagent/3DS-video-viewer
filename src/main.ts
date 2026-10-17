import './style.css';
import { WebGLParallaxViewer, type RenderMode, type DisparityBackend, type UpscaleMultiplier } from './renderer/webglViewer.js';
import { MotionController } from './renderer/motionController.js';
import { import3DSVideo, type VideoSource } from './importer/import3DSVideo.js';
import { fetchDcimList, fetchBundledManifest, formatBytes } from './importer/dcimCatalog.js';
import { exportParallaxVideo, exportOfflineGPUVideo, type ExportFormat, type ExportMode } from './exporter/index.js';

// DOM Elements
const canvas = document.getElementById('parallax-canvas') as HTMLCanvasElement;
const sourceVideo = document.getElementById('source-video') as HTMLVideoElement;
const viewportContainer = document.getElementById('viewport-container') as HTMLElement;
const shiftPin = document.getElementById('shift-pin') as HTMLElement;

// Transport Elements
const playPauseBtn = document.getElementById('play-pause-btn') as HTMLButtonElement;
const playIcon = document.getElementById('play-icon') as HTMLElement;
const seekBar = document.getElementById('seek-bar') as HTMLInputElement;
const timeDisplay = document.getElementById('time-display') as HTMLElement;
const muteBtn = document.getElementById('mute-btn') as HTMLButtonElement;
const muteIcon = document.getElementById('mute-icon') as HTMLElement;
const browseFileBtn = document.getElementById('browse-file-btn') as HTMLButtonElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const dropOverlay = document.getElementById('drop-overlay') as HTMLElement;
const progressBanner = document.getElementById('progress-banner') as HTMLElement;
const progressStatusText = document.getElementById('progress-status-text') as HTMLElement;
const progressPercent = document.getElementById('progress-percent') as HTMLElement;

// Telemetry Elements
const telemetryFps = document.getElementById('telemetry-fps') as HTMLElement;
const telemetryDisp = document.getElementById('telemetry-disp') as HTMLElement;
const telemetryRender = document.getElementById('telemetry-render') as HTMLElement;
const telemetryRes = document.getElementById('telemetry-res') as HTMLElement;
const telemetryShift = document.getElementById('telemetry-shift') as HTMLElement;

// Export Controls
const exportMp4Btn = document.getElementById('export-mp4-btn') as HTMLButtonElement;
const exportWebmBtn = document.getElementById('export-webm-btn') as HTMLButtonElement;
const exportModeSelect = document.getElementById('export-mode-select') as HTMLSelectElement;
const exportStatus = document.getElementById('export-status') as HTMLElement;

// Shader & Disparity Controls
const backendSelect = document.getElementById('backend-select') as HTMLSelectElement;
const renderModeSelect = document.getElementById('render-mode-select') as HTMLSelectElement;
const upscaleSelect = document.getElementById('upscale-select') as HTMLSelectElement;

// Motion Controls
const motionModeSelect = document.getElementById('motion-mode-select') as HTMLSelectElement;
const manualShiftGroup = document.getElementById('manual-shift-group') as HTMLElement;
const manualShiftSlider = document.getElementById('manual-shift-slider') as HTMLInputElement;
const manualShiftVal = document.getElementById('manual-shift-val') as HTMLElement;

const cropSlider = document.getElementById('crop-slider') as HTMLInputElement;
const cropVal = document.getElementById('crop-val') as HTMLElement;
const maxdispSlider = document.getElementById('maxdisp-slider') as HTMLInputElement;
const maxdispVal = document.getElementById('maxdisp-val') as HTMLElement;
const freqSlider = document.getElementById('freq-slider') as HTMLInputElement;
const freqVal = document.getElementById('freq-val') as HTMLElement;
const ampSlider = document.getElementById('amp-slider') as HTMLInputElement;
const ampVal = document.getElementById('amp-val') as HTMLElement;

// Catalog Elements
const bundledSamplesList = document.getElementById('bundled-samples-list') as HTMLElement;
const dcimRecordingsList = document.getElementById('dcim-recordings-list') as HTMLElement;
const refreshDcimBtn = document.getElementById('refresh-dcim-btn') as HTMLButtonElement;

// Instantiate WebGL Viewer & Motion Controller
const viewer = new WebGLParallaxViewer(canvas, {
  upscale: 2,
  backend: 'gpu',
  renderMode: 0,
  windowCrop: 0.06,
  maxDisp: 24,
});

const motionController = new MotionController(viewportContainer, {
  frequencyHz: 0.28,
  amplitude: 0.42,
});

let motionMode: 'auto' | 'interactive' | 'manual' = 'auto';
let isSeeking = false;
let isExporting = false;

// Format seconds into MM:SS
function formatTime(seconds: number): string {
  if (!seconds || !isFinite(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s < 10 ? '0' : ''}${s}`;
}

// Progress Banner Helper
function setImportProgress(stage: string, ratio: number) {
  if (ratio >= 1.0) {
    progressBanner.style.display = 'none';
  } else {
    progressBanner.style.display = 'flex';
    progressStatusText.textContent = stage;
    progressPercent.textContent = `${Math.round(ratio * 100)}%`;
  }
}

// Load Video Helper
async function loadVideo(source: VideoSource) {
  try {
    setImportProgress('Starting video import...', 0.05);
    const result = await import3DSVideo(source, sourceVideo, (stage, ratio) => {
      setImportProgress(stage, ratio);
    });
    console.log('[ViewerApp] Loaded video:', result.filename, result.format);
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[ViewerApp] Failed to load video:', msg);
    progressBanner.style.display = 'flex';
    progressStatusText.textContent = `Error: ${msg}`;
    progressPercent.textContent = '!';
    setTimeout(() => {
      progressBanner.style.display = 'none';
    }, 5000);
  }
}

// Wire Transport Controls
playPauseBtn.addEventListener('click', () => {
  if (sourceVideo.paused) {
    sourceVideo.play().catch(() => {
      sourceVideo.muted = true;
      sourceVideo.play();
    });
  } else {
    sourceVideo.pause();
  }
});

sourceVideo.addEventListener('play', () => {
  playIcon.textContent = '⏸ Pause';
});

sourceVideo.addEventListener('pause', () => {
  playIcon.textContent = '▶ Play';
});

sourceVideo.addEventListener('timeupdate', () => {
  if (!isSeeking && sourceVideo.duration) {
    const pct = (sourceVideo.currentTime / sourceVideo.duration) * 100;
    seekBar.value = pct.toFixed(1);
    timeDisplay.textContent = `${formatTime(sourceVideo.currentTime)} / ${formatTime(sourceVideo.duration)}`;
  }
});

seekBar.addEventListener('input', () => {
  isSeeking = true;
  if (sourceVideo.duration) {
    const targetTime = (parseFloat(seekBar.value) / 100) * sourceVideo.duration;
    timeDisplay.textContent = `${formatTime(targetTime)} / ${formatTime(sourceVideo.duration)}`;
  }
});

seekBar.addEventListener('change', () => {
  if (sourceVideo.duration) {
    sourceVideo.currentTime = (parseFloat(seekBar.value) / 100) * sourceVideo.duration;
  }
  isSeeking = false;
});

muteBtn.addEventListener('click', () => {
  sourceVideo.muted = !sourceVideo.muted;
  muteIcon.textContent = sourceVideo.muted ? '🔇 Muted' : '🔊 Sound';
});

// File Browsing & Drag/Drop
browseFileBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (file) {
    loadVideo(file);
  }
});

window.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropOverlay.classList.add('active');
});

dropOverlay.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropOverlay.classList.remove('active');
});

dropOverlay.addEventListener('drop', (e) => {
  e.preventDefault();
  dropOverlay.classList.remove('active');
  const file = e.dataTransfer?.files[0];
  if (file) {
    loadVideo(file);
  }
});

// Sidebar Controls Wiring
backendSelect.addEventListener('change', () => {
  viewer.setBackend(backendSelect.value as DisparityBackend);
});

renderModeSelect.addEventListener('change', () => {
  viewer.setRenderMode(parseInt(renderModeSelect.value, 10) as RenderMode);
});

upscaleSelect.addEventListener('change', () => {
  const upscale = parseInt(upscaleSelect.value, 10) as UpscaleMultiplier;
  viewer.setUpscaleMultiplier(upscale);
  telemetryRes.textContent = `${canvas.width}×${canvas.height}`;
});

cropSlider.addEventListener('input', () => {
  const val = parseFloat(cropSlider.value);
  cropVal.textContent = val.toFixed(2);
  viewer.setWindowCrop(val);
});

maxdispSlider.addEventListener('input', () => {
  const val = parseInt(maxdispSlider.value, 10);
  maxdispVal.textContent = String(val);
  viewer.setMaxDisparity(val);
});

freqSlider.addEventListener('input', () => {
  const val = parseFloat(freqSlider.value);
  freqVal.textContent = `${val.toFixed(2)} Hz`;
  motionController.attach(viewportContainer); // updates options if needed
});

ampSlider.addEventListener('input', () => {
  const val = parseFloat(ampSlider.value);
  ampVal.textContent = val.toFixed(2);
});

motionModeSelect.addEventListener('change', () => {
  motionMode = motionModeSelect.value as 'auto' | 'interactive' | 'manual';
  if (motionMode === 'manual') {
    manualShiftGroup.style.display = 'flex';
    motionController.detach();
  } else if (motionMode === 'interactive') {
    manualShiftGroup.style.display = 'none';
    motionController.attach(viewportContainer);
  } else {
    manualShiftGroup.style.display = 'none';
    motionController.detach();
  }
});

manualShiftSlider.addEventListener('input', () => {
  const val = parseFloat(manualShiftSlider.value);
  manualShiftVal.textContent = val.toFixed(2);
  motionController.setManualShift(val);
});

// Hardware GPU Video Export Wiring
async function triggerExport(format: ExportFormat) {
  if (isExporting) return;
  isExporting = true;
  exportMp4Btn.disabled = true;
  exportWebmBtn.disabled = true;
  exportStatus.textContent = `Starting ${format.toUpperCase()} hardware GPU export...`;

  const mode = exportModeSelect.value as ExportMode;

  try {
    if (mode === 'offline') {
      const result = await exportParallaxVideo(format, 'offline', viewer, sourceVideo, {
        fps: 60,
        maxDurationSec: 30,
        triggerDownload: true,
        onProgress: (p, frame, total) => {
          const pct = Math.round(p * 100);
          exportStatus.textContent = `Exporting ${format.toUpperCase()}: frame ${frame}/${total} (${pct}%)...`;
        },
      });
      exportStatus.textContent = `Done! Exported ${(result as any).filename} (${formatBytes((result as any).blob.size)})`;
    } else {
      exportStatus.textContent = `Recording ${format.toUpperCase()} in real-time (press button again to stop)...`;
      const session = (await exportParallaxVideo(format, 'realtime', viewer, sourceVideo, {
        triggerDownload: true,
      })) as any;

      // Stop on next click or after video finishes
      const onStopClick = async () => {
        exportStatus.textContent = 'Finalizing real-time recording...';
        const res = await session.stop();
        exportStatus.textContent = `Saved ${res.filename} (${formatBytes(res.blob.size)})`;
        cleanup();
      };

      const cleanup = () => {
        isExporting = false;
        exportMp4Btn.disabled = false;
        exportWebmBtn.disabled = false;
        exportMp4Btn.removeEventListener('click', onStopClick);
        exportWebmBtn.removeEventListener('click', onStopClick);
      };

      exportMp4Btn.addEventListener('click', onStopClick, { once: true });
      exportWebmBtn.addEventListener('click', onStopClick, { once: true });
      return;
    }
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error('[Export Failed]', err);
    exportStatus.textContent = `Export failed: ${msg}`;
  } finally {
    isExporting = false;
    exportMp4Btn.disabled = false;
    exportWebmBtn.disabled = false;
  }
}

exportMp4Btn.addEventListener('click', () => triggerExport('mp4'));
exportWebmBtn.addEventListener('click', () => triggerExport('webm'));

// Catalog Populator
async function populateCatalog() {
  // 1. Bundled Samples
  const samples = await fetchBundledManifest();
  bundledSamplesList.innerHTML = '';
  if (samples.length === 0) {
    bundledSamplesList.innerHTML = '<div style="font-size: 0.72rem; color: var(--text-muted); padding: 0.35rem;">No bundled samples found.</div>';
  } else {
    for (const s of samples) {
      const itemEl = document.createElement('div');
      itemEl.className = 'dcim-item';
      itemEl.innerHTML = `
        <div class="dcim-info">
          <span class="dcim-name">${s.name}</span>
          <span class="dcim-meta">${s.description || ''}</span>
        </div>
        <button class="load-btn" data-sample-url="${s.url}" data-testid="load-sample-${s.id}">Load</button>
      `;
      itemEl.querySelector('.load-btn')?.addEventListener('click', () => {
        loadVideo({ url: s.url, name: s.name });
      });
      bundledSamplesList.appendChild(itemEl);
    }
  }

  // 2. Local DCIM Recordings
  const dcimFiles = await fetchDcimList();
  dcimRecordingsList.innerHTML = '';
  if (dcimFiles.length === 0) {
    dcimRecordingsList.innerHTML = '<div style="font-size: 0.72rem; color: var(--text-muted); padding: 0.35rem;">(Local /DCIM/ catalog available when running Vite dev/preview server)</div>';
  } else {
    for (const f of dcimFiles) {
      const itemEl = document.createElement('div');
      itemEl.className = 'dcim-item';
      itemEl.innerHTML = `
        <div class="dcim-info">
          <span class="dcim-name">${f.folder ? f.folder + '/' : ''}${f.name}</span>
          <span class="dcim-meta">${formatBytes(f.sizeBytes)}</span>
        </div>
        <button class="load-btn" data-dcim-path="${f.relativePath}" data-testid="load-dcim-${f.name}">Load</button>
      `;
      itemEl.querySelector('.load-btn')?.addEventListener('click', () => {
        loadVideo({
          url: `/api/dcim/file?path=${encodeURIComponent(f.relativePath)}`,
          name: f.name,
        });
      });
      dcimRecordingsList.appendChild(itemEl);
    }
  }
}

refreshDcimBtn.addEventListener('click', populateCatalog);

let isLoopRunning = true;

// Main Animation & Parallax Render Loop (60 FPS)
function renderLoop(timestampMs: number) {
  if (!isLoopRunning) return;

  let viewShift = 0.5;

  if (motionMode === 'manual') {
    viewShift = parseFloat(manualShiftSlider.value);
  } else if (motionMode === 'interactive') {
    viewShift = motionController.update(timestampMs);
  } else {
    // Auto 0.28Hz Sine Wave
    const freq = parseFloat(freqSlider.value);
    const amp = parseFloat(ampSlider.value);
    const sine = Math.sin((timestampMs / 1000.0) * freq * Math.PI * 2.0);
    viewShift = Math.max(0.0, Math.min(1.0, 0.5 + amp * sine));
  }

  // Update Shift HUD Pin
  shiftPin.style.left = `${(viewShift * 100).toFixed(1)}%`;

  // Render on WebGL2 GPU
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !isExporting) {
    viewer.renderVideoFrame(sourceVideo, viewShift);
  }

  // Update Live Telemetry Pill
  const telemetry = viewer.getTelemetry();
  telemetryFps.textContent = String(telemetry.fps);
  telemetryDisp.textContent = `${telemetry.disparityMs.toFixed(1)}ms`;
  telemetryRender.textContent = `${telemetry.renderMs.toFixed(1)}ms`;
  telemetryRes.textContent = `${canvas.width}×${canvas.height}`;
  telemetryShift.textContent = viewShift.toFixed(2);

  requestAnimationFrame(renderLoop);
}

// Initial Boot Sequence
populateCatalog();
requestAnimationFrame(renderLoop);

// Auto-load bundled SBS sample on startup so viewer is immediately active
loadVideo({
  url: 'samples/HNI_0070_sbs.mp4',
  name: 'HNI_0070 (Pre-converted SBS MP4)',
});

// Expose on window for automated Playwright E2E tests and debugging
(window as any).__viewerApp = {
  viewer,
  videoElement: sourceVideo,
  motionController,
  importVideo: loadVideo,
  exportVideo: triggerExport,
  exportOffline: (format: ExportFormat, options?: any) =>
    exportOfflineGPUVideo(format, viewer, sourceVideo, options),
  getTelemetry: () => viewer.getTelemetry(),
  setMotionMode: (mode: 'auto' | 'interactive' | 'manual') => {
    motionMode = mode;
    motionModeSelect.value = mode;
  },
  setManualShift: (shift: number) => {
    motionMode = 'manual';
    motionModeSelect.value = 'manual';
    manualShiftGroup.style.display = 'flex';
    manualShiftSlider.value = String(shift);
    manualShiftVal.textContent = shift.toFixed(2);
    motionController.setManualShift(shift);
  },
  pauseRenderLoop: () => {
    isLoopRunning = false;
  },
  resumeRenderLoop: () => {
    if (!isLoopRunning) {
      isLoopRunning = true;
      requestAnimationFrame(renderLoop);
    }
  },
};
