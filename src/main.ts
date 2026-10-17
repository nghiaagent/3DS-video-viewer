import './style.css';
import { WebGLParallaxViewer, type RenderMode, type DisparityBackend, type UpscaleMultiplier } from './renderer/webglViewer.js';
import { MotionController } from './renderer/motionController.js';
import { import3DSVideo, type VideoSource } from './importer/import3DSVideo.js';
import { exportParallaxVideo, exportOfflineGPUVideo, type ExportFormat, type ExportMode } from './exporter/index.js';

// DOM Elements
const canvas = document.getElementById('parallax-canvas') as HTMLCanvasElement;
const sourceVideo = document.getElementById('source-video') as HTMLVideoElement;
const viewportContainer = document.getElementById('viewport-container') as HTMLElement;
const spatialWindow = document.getElementById('spatial-window') as HTMLElement;
const ambientGlow = document.getElementById('ambient-glow') as HTMLElement;
const emptyStateOverlay = document.getElementById('empty-state-overlay') as HTMLElement;

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

// Controls
const exportMp4Btn = document.getElementById('export-mp4-btn') as HTMLButtonElement;
const exportWebmBtn = document.getElementById('export-webm-btn') as HTMLButtonElement;
const exportModeSelect = document.getElementById('export-mode-select') as HTMLSelectElement | null;
const exportStatus = document.getElementById('export-status') as HTMLElement | null;

const backendSelect = document.getElementById('backend-select') as HTMLSelectElement | null;
const renderModeSelect = document.getElementById('render-mode-select') as HTMLSelectElement;
const upscaleSelect = document.getElementById('upscale-select') as HTMLSelectElement | null;
const motionModeSelect = document.getElementById('motion-mode-select') as HTMLSelectElement;

const manualShiftGroup = document.getElementById('manual-shift-group') as HTMLElement | null;
const manualShiftSlider = document.getElementById('manual-shift-slider') as HTMLInputElement | null;
const manualShiftVal = document.getElementById('manual-shift-val') as HTMLElement | null;

const cropSlider = document.getElementById('crop-slider') as HTMLInputElement | null;
const maxdispSlider = document.getElementById('maxdisp-slider') as HTMLInputElement;
const maxdispVal = document.getElementById('maxdisp-val') as HTMLElement | null;
const freqSlider = document.getElementById('freq-slider') as HTMLInputElement | null;
const ampSlider = document.getElementById('amp-slider') as HTMLInputElement | null;

// Telemetry & Indicators (preserved for automated testing hooks)
const telemetryFps = document.getElementById('telemetry-fps');
const telemetryDisp = document.getElementById('telemetry-disp');
const telemetryRender = document.getElementById('telemetry-render');
const telemetryRes = document.getElementById('telemetry-res');
const telemetryShift = document.getElementById('telemetry-shift');
const shiftPin = document.getElementById('shift-pin');

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
let isLoopRunning = true;

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
    setImportProgress('Opening video...', 0.05);
    const result = await import3DSVideo(source, sourceVideo, (stage, ratio) => {
      setImportProgress(stage, ratio);
    });
    if (emptyStateOverlay) {
      emptyStateOverlay.classList.add('hidden');
    }
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
  playIcon.textContent = '⏸';
  if (emptyStateOverlay) {
    emptyStateOverlay.classList.add('hidden');
  }
});

sourceVideo.addEventListener('pause', () => {
  playIcon.textContent = '▶';
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
  muteIcon.textContent = sourceVideo.muted ? '🔇' : '🔊';
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

// Controls Wiring
if (backendSelect) {
  backendSelect.addEventListener('change', () => {
    viewer.setBackend(backendSelect.value as DisparityBackend);
  });
}

renderModeSelect.addEventListener('change', () => {
  viewer.setRenderMode(parseInt(renderModeSelect.value, 10) as RenderMode);
});

if (upscaleSelect) {
  upscaleSelect.addEventListener('change', () => {
    const upscale = parseInt(upscaleSelect.value, 10) as UpscaleMultiplier;
    viewer.setUpscaleMultiplier(upscale);
    if (telemetryRes) telemetryRes.textContent = `${canvas.width}×${canvas.height}`;
  });
}

if (cropSlider) {
  cropSlider.addEventListener('input', () => {
    const val = parseFloat(cropSlider.value);
    viewer.setWindowCrop(val);
  });
}

maxdispSlider.addEventListener('input', () => {
  const val = parseInt(maxdispSlider.value, 10);
  if (maxdispVal) maxdispVal.textContent = String(val);
  viewer.setMaxDisparity(val);
});

motionModeSelect.addEventListener('change', () => {
  motionMode = motionModeSelect.value as 'auto' | 'interactive' | 'manual';
  if (motionMode === 'manual') {
    if (manualShiftGroup) manualShiftGroup.style.display = 'flex';
    motionController.detach();
  } else if (motionMode === 'interactive') {
    if (manualShiftGroup) manualShiftGroup.style.display = 'none';
    motionController.attach(viewportContainer);
  } else {
    if (manualShiftGroup) manualShiftGroup.style.display = 'none';
    motionController.detach();
  }
});

if (manualShiftSlider) {
  manualShiftSlider.addEventListener('input', () => {
    const val = parseFloat(manualShiftSlider.value);
    if (manualShiftVal) manualShiftVal.textContent = val.toFixed(2);
    motionController.setManualShift(val);
  });
}

// Hardware GPU Video Export Wiring
async function triggerExport(format: ExportFormat) {
  if (isExporting) return;
  isExporting = true;
  exportMp4Btn.disabled = true;
  exportWebmBtn.disabled = true;
  if (exportStatus) exportStatus.textContent = `Exporting ${format.toUpperCase()}...`;

  const mode: ExportMode = (exportModeSelect?.value as ExportMode) || 'offline';

  try {
    if (mode === 'offline') {
      const result = await exportParallaxVideo(format, 'offline', viewer, sourceVideo, {
        fps: 60,
        maxDurationSec: 30,
        triggerDownload: true,
        onProgress: (p, frame, total) => {
          const pct = Math.round(p * 100);
          if (exportStatus) exportStatus.textContent = `Exporting ${format.toUpperCase()}: ${frame}/${total} (${pct}%)...`;
        },
      });
      if (exportStatus) exportStatus.textContent = `Saved ${(result as any).filename}`;
    } else {
      const session = (await exportParallaxVideo(format, 'realtime', viewer, sourceVideo, {
        triggerDownload: true,
      })) as any;

      const onStopClick = async () => {
        const res = await session.stop();
        if (exportStatus) exportStatus.textContent = `Saved ${res.filename}`;
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
    if (exportStatus) exportStatus.textContent = `Export failed: ${msg}`;
  } finally {
    isExporting = false;
    exportMp4Btn.disabled = false;
    exportWebmBtn.disabled = false;
  }
}

exportMp4Btn.addEventListener('click', () => triggerExport('mp4'));
exportWebmBtn.addEventListener('click', () => triggerExport('webm'));

// Main Animation & Parallax Render Loop (60 FPS) with 3D Spatial Window Movement
function renderLoop(timestampMs: number) {
  if (!isLoopRunning) return;

  let viewShift = 0.5;

  if (motionMode === 'manual') {
    viewShift = manualShiftSlider ? parseFloat(manualShiftSlider.value) : 0.5;
  } else if (motionMode === 'interactive') {
    viewShift = motionController.update(timestampMs);
  } else {
    // Auto 0.28Hz Sine Wave
    const freq = freqSlider ? parseFloat(freqSlider.value) : 0.28;
    const amp = ampSlider ? parseFloat(ampSlider.value) : 0.42;
    const sine = Math.sin((timestampMs / 1000.0) * freq * Math.PI * 2.0);
    viewShift = Math.max(0.0, Math.min(1.0, 0.5 + amp * sine));
  }

  // Apple Vision Pro Moving Spatial Window Motion
  if (spatialWindow) {
    const s = (viewShift - 0.5) * 2.0; // [-1.0, +1.0]
    const v = motionController.getVerticalTilt(); // [-1.0, +1.0]

    const tx = s * 18.0; // horizontal drift px
    const ty = v * 8.0;  // vertical drift px
    const ry = -s * 6.5; // Y-axis rotation in degrees
    const rx = v * 4.0;  // X-axis rotation in degrees

    spatialWindow.style.transform = `translateX(${tx.toFixed(2)}px) translateY(${ty.toFixed(2)}px) rotateY(${ry.toFixed(2)}deg) rotateX(${rx.toFixed(2)}deg)`;

    const shadowX = (-s * 28.0).toFixed(1);
    spatialWindow.style.boxShadow = `${shadowX}px 32px 80px rgba(0, 0, 0, 0.85), 0 0 90px rgba(56, 189, 248, 0.08), 0 0 0 1px rgba(255, 255, 255, 0.14)`;

    if (ambientGlow) {
      ambientGlow.style.transform = `translateX(${(-s * 45).toFixed(1)}px)`;
    }
  }

  if (shiftPin) {
    shiftPin.style.left = `${(viewShift * 100).toFixed(1)}%`;
  }

  // Render on WebGL2 GPU
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !isExporting) {
    viewer.renderVideoFrame(sourceVideo, viewShift);
  }

  // Update Telemetry if elements are queried by automated tests
  const telemetry = viewer.getTelemetry();
  if (telemetryFps) telemetryFps.textContent = String(telemetry.fps);
  if (telemetryDisp) telemetryDisp.textContent = `${telemetry.disparityMs.toFixed(1)}ms`;
  if (telemetryRender) telemetryRender.textContent = `${telemetry.renderMs.toFixed(1)}ms`;
  if (telemetryRes) telemetryRes.textContent = `${canvas.width}×${canvas.height}`;
  if (telemetryShift) telemetryShift.textContent = viewShift.toFixed(2);

  requestAnimationFrame(renderLoop);
}

// Initial Boot Sequence
requestAnimationFrame(renderLoop);

// Expose on window for automated Playwright E2E tests and debugging
(window as any).__viewerApp = {
  viewer,
  videoElement: sourceVideo,
  motionController,
  spatialWindow,
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
    if (manualShiftGroup) manualShiftGroup.style.display = 'flex';
    if (manualShiftSlider) manualShiftSlider.value = String(shift);
    if (manualShiftVal) manualShiftVal.textContent = shift.toFixed(2);
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
