import './style.css';
import { WebGLParallaxViewer, type RenderMode, type DisparityBackend, type UpscaleMultiplier } from './renderer/webglViewer.js';
import { MotionController } from './renderer/motionController.js';
import { import3DSVideo, type VideoSource } from './importer/import3DSVideo.js';
import { exportParallaxVideo, exportOfflineGPUVideo, type ExportFormat, type ExportMode } from './exporter/index.js';
import { DepthAnythingController, VideoDepthTimeline } from './depth/stereoGuidedDepth.js';

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

const backendSelect = document.getElementById('backend-select') as HTMLSelectElement | null;
const renderModeSelect = document.getElementById('render-mode-select') as HTMLSelectElement;
const modelV2Btn = document.getElementById('model-v2-btn') as HTMLButtonElement | null;
const modelV3Btn = document.getElementById('model-v3-btn') as HTMLButtonElement | null;
const upscaleSelect = document.getElementById('upscale-select') as HTMLSelectElement | null;
const motionModeSelect = document.getElementById('motion-mode-select') as HTMLSelectElement;

const speedDockControl = document.getElementById('speed-dock-control') as HTMLElement | null;
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

const depthController = new DepthAnythingController(
  { modelVersion: 'v3', inputSize: 504, expandRadius: 1, dilateRadius: 1 },
  (calibratedDepth, width, height, _telemetry) => {
    viewer.uploadExternalDepthMap(calibratedDepth, width, height);
  }
);
depthController.init().catch((err) => {
  console.warn('[DepthController] Init warning:', err);
});

let videoDepthTimeline: VideoDepthTimeline = new VideoDepthTimeline({
  fps: 20,
  inputSize: 336,
});
depthController.setTimeline(videoDepthTimeline);

// Register frame-accurate synchronous depth provider (0.0 ms desync)
viewer.setDepthProvider((timeSec: number) => {
  if (!videoDepthTimeline) return null;
  const depth = videoDepthTimeline.getDepthAtTime(timeSec, depthController.getModelVersion());
  if (depth) {
    return { depth, width: 480, height: 240 };
  }
  return null;
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

export type LoadVideoSource = File | Blob | string | VideoSource;

// Load Video Helper
async function loadVideo(source: LoadVideoSource, displayName?: string) {
  try {
    viewer.clearExternalDepthMap();
    setImportProgress('Opening video...', 0.05);

    let sourceInput: VideoSource;
    if (typeof source === 'string') {
      sourceInput = {
        url: source,
        name: displayName || source.split('/').pop() || '3ds_recording.avi',
      };
    } else if (source instanceof File) {
      sourceInput = source;
    } else if (source instanceof Blob) {
      sourceInput = new File([source], displayName || '3ds_recording.avi', { type: source.type });
    } else {
      sourceInput = source;
    }

    const result = await import3DSVideo(sourceInput, sourceVideo, (stage, ratio) => {
      setImportProgress(stage, ratio);
    });
    sourceVideo.pause();
    sourceVideo.currentTime = 0;

    if (result.metadata && result.metadata.found) {
      viewer.setThreeDSMetadata(result.metadata.parallaxOffset, result.metadata.estimatedBaseDisp);
      if (result.metadata.estimatedBaseDisp !== null && Number.isFinite(result.metadata.estimatedBaseDisp)) {
        viewer.setBaseDisparity(result.metadata.estimatedBaseDisp);
      }
    } else {
      viewer.setThreeDSMetadata(0, null as any);
    }
    if (emptyStateOverlay) {
      emptyStateOverlay.classList.add('hidden');
    }
    console.log('[ViewerApp] Loaded video:', result.filename, result.format, result.metadata);

    // Initial timeline pre-computation across video stream for BOTH V3 and V2
    if (videoDepthTimeline) {
      videoDepthTimeline.destroy();
    }
    videoDepthTimeline = new VideoDepthTimeline({
      fps: 20,
      inputSize: 336,
      baseDisp: viewer.getBaseDisparity() ?? 0,
    });
    depthController.setTimeline(videoDepthTimeline);

    const isE2E =
      typeof navigator !== 'undefined' &&
      (Boolean((navigator as any).webdriver) || /HeadlessChrome/.test(navigator.userAgent));
    const initialKeyframes = isE2E ? 2 : 6;
    const timelineTimeoutMs = isE2E ? 12_000 : 30_000;

    await Promise.race([
      videoDepthTimeline.buildInitialTimeline(
        sourceVideo,
        viewer.getBaseDisparity(),
        (stage, ratio) => {
          setImportProgress(stage, ratio);
        },
        {
          initialKeyframes,
          runFullStreamInBackground: true,
        }
      ),
      new Promise<void>((resolve) => setTimeout(resolve, timelineTimeoutMs)),
    ]);

    // Initial pass complete; hide progress banner and start playback
    setImportProgress('Ready', 1.0);

    try {
      await sourceVideo.play();
    } catch {
      sourceVideo.muted = true;
      await sourceVideo.play().catch(() => {});
    }
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

async function setDepthModelVersion(version: 'v2' | 'v3') {
  if (modelV2Btn && modelV3Btn) {
    if (version === 'v2') {
      modelV2Btn.classList.add('active');
      modelV3Btn.classList.remove('active');
    } else {
      modelV3Btn.classList.add('active');
      modelV2Btn.classList.remove('active');
    }
  }

  // 1. Update model version on depth controller
  const setModelPromise = depthController.setModelVersion(version);

  // 2. Immediately pull current timestamp's depth map from videoDepthTimeline (0 ms instant switch!)
  if (videoDepthTimeline && sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    const depth = videoDepthTimeline.getDepthAtTime(sourceVideo.currentTime, version);
    if (depth) {
      viewer.uploadExternalDepthMap(depth, 480, 240);
      const verticalTilt = motionController.getVerticalTilt();
      viewer.renderVideoFrame(sourceVideo, currentViewShift, verticalTilt);
    }
  }

  await setModelPromise;
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    depthController.processVideoFrame(sourceVideo, viewer.getBaseDisparity(), 16).catch(() => {});
  }
}

if (modelV2Btn) {
  modelV2Btn.addEventListener('click', () => {
    setDepthModelVersion('v2');
  });
}

if (modelV3Btn) {
  modelV3Btn.addEventListener('click', () => {
    setDepthModelVersion('v3');
  });
}

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

function updateMotionModeUI(mode: 'auto' | 'interactive' | 'manual') {
  if (mode === 'manual') {
    if (manualShiftGroup) manualShiftGroup.style.display = 'flex';
    if (speedDockControl) speedDockControl.style.display = 'none';
    motionController.detach();
  } else if (mode === 'interactive') {
    if (manualShiftGroup) manualShiftGroup.style.display = 'none';
    if (speedDockControl) speedDockControl.style.display = 'none';
    motionController.attach(viewportContainer);
  } else {
    // Gentle Sway mode
    if (manualShiftGroup) manualShiftGroup.style.display = 'none';
    if (speedDockControl) speedDockControl.style.display = 'flex';
    motionController.detach();
  }
}

motionModeSelect.addEventListener('change', () => {
  motionMode = motionModeSelect.value as 'auto' | 'interactive' | 'manual';
  updateMotionModeUI(motionMode);
});

if (manualShiftSlider) {
  manualShiftSlider.addEventListener('input', () => {
    const val = parseFloat(manualShiftSlider.value);
    if (manualShiftVal) manualShiftVal.textContent = val.toFixed(2);
    motionController.setManualShift(val);
  });
}

// Circular Export Progress Helper
function setExportProgress(btn: HTMLButtonElement, progress: number) {
  const circle = btn.querySelector<SVGCircleElement>('.progress-ring-fill');
  if (circle) {
    const circumference = 2 * Math.PI * 8; // ~50.2655
    const clampedProgress = Math.max(0, Math.min(1, progress));
    const offset = circumference * (1 - clampedProgress);
    circle.style.strokeDashoffset = `${offset.toFixed(2)}px`;
  }
}

// Hardware GPU Video Export Wiring (Circular Progress Rings, Zero Text Clutter)
async function triggerExport(format: ExportFormat) {
  if (isExporting) return;
  isExporting = true;
  exportMp4Btn.disabled = true;
  exportWebmBtn.disabled = true;

  const activeBtn = format === 'mp4' ? exportMp4Btn : exportWebmBtn;
  activeBtn.classList.add('is-exporting');
  setExportProgress(activeBtn, 0);

  const mode: ExportMode = (exportModeSelect?.value as ExportMode) || 'offline';

  try {
    if (mode === 'offline') {
      await exportParallaxVideo(format, 'offline', viewer, sourceVideo, {
        fps: 60,
        maxDurationSec: 30,
        triggerDownload: true,
        onProgress: (p) => {
          setExportProgress(activeBtn, p);
        },
      });
    } else {
      const session = (await exportParallaxVideo(format, 'realtime', viewer, sourceVideo, {
        triggerDownload: true,
      })) as any;

      const onStopClick = async () => {
        await session.stop();
        cleanup();
      };

      const cleanup = () => {
        isExporting = false;
        exportMp4Btn.disabled = false;
        exportWebmBtn.disabled = false;
        activeBtn.classList.remove('is-exporting');
        setExportProgress(activeBtn, 0);
        exportMp4Btn.removeEventListener('click', onStopClick);
        exportWebmBtn.removeEventListener('click', onStopClick);
      };

      exportMp4Btn.addEventListener('click', onStopClick, { once: true });
      exportWebmBtn.addEventListener('click', onStopClick, { once: true });
      return;
    }
  } catch (err: unknown) {
    console.error('[Export Failed]', err);
  } finally {
    isExporting = false;
    exportMp4Btn.disabled = false;
    exportWebmBtn.disabled = false;
    activeBtn.classList.remove('is-exporting');
    setExportProgress(activeBtn, 0);
  }
}

exportMp4Btn.addEventListener('click', () => triggerExport('mp4'));
exportWebmBtn.addEventListener('click', () => triggerExport('webm'));

// Phase accumulator for smooth frequency transitions without teleporting
let lastTimestampMs = 0;
let currentPhase = 0;
let currentViewShift = 0.5;

// Main Animation & Parallax Render Loop (60 FPS) with Apple Vision Pro 3D Window Movement
function renderLoop(timestampMs: number) {
  if (!isLoopRunning) return;

  if (!lastTimestampMs) {
    lastTimestampMs = timestampMs;
  }
  const dtSeconds = Math.min(Math.max(0, (timestampMs - lastTimestampMs) / 1000.0), 0.1);
  lastTimestampMs = timestampMs;

  let viewShift = 0.5;

  if (motionMode === 'manual') {
    viewShift = manualShiftSlider ? parseFloat(manualShiftSlider.value) : 0.5;
  } else if (motionMode === 'interactive') {
    viewShift = motionController.update(timestampMs);
  } else {
    // Gentle Sway mode: accumulate phase based on frequency slider to eliminate teleporting jumps
    const freq = freqSlider ? parseFloat(freqSlider.value) : 0.28;
    const amp = ampSlider ? parseFloat(ampSlider.value) : 0.42;
    currentPhase = (currentPhase + dtSeconds * freq * Math.PI * 2.0) % (Math.PI * 2.0);
    const sine = Math.sin(currentPhase);
    viewShift = Math.max(0.0, Math.min(1.0, 0.5 + amp * sine));
  }
  currentViewShift = viewShift;

  // Enhanced Apple Vision Pro Spatial Window Movement & Reactive Specular Border
  if (spatialWindow) {
    const s = (viewShift - 0.5) * 2.0; // [-1.0, +1.0]
    const v = motionController.getVerticalTilt(); // [-1.0, +1.0]

    const tx = s * 26.0; // horizontal drift px
    const ty = v * 10.0; // vertical drift px
    const ry = -s * 8.0; // Y-axis rotation in degrees
    const rx = v * 5.0;  // X-axis rotation in degrees

    spatialWindow.style.transform = `translate3d(${tx.toFixed(2)}px, ${ty.toFixed(2)}px, 0px) rotateY(${ry.toFixed(2)}deg) rotateX(${rx.toFixed(2)}deg)`;
    spatialWindow.style.setProperty('--window-shift', s.toFixed(3));

    const shadowX = (-s * 32.0).toFixed(1);
    spatialWindow.style.boxShadow = `${shadowX}px 32px 80px rgba(0, 0, 0, 0.85), 0 0 90px rgba(56, 189, 248, 0.08), 0 0 0 1px rgba(255, 255, 255, 0.14)`;

    if (ambientGlow) {
      ambientGlow.style.transform = `translate3d(${(-s * 50).toFixed(1)}px, 0px, 0px)`;
    }
  }

  if (shiftPin) {
    shiftPin.style.left = `${(viewShift * 100).toFixed(1)}%`;
  }

  // Render on WebGL2 GPU
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !isExporting) {
    const verticalTilt = motionController.getVerticalTilt();
    viewer.renderVideoFrame(sourceVideo, viewShift, verticalTilt);

    if (!depthController.isBusy()) {
      depthController.processVideoFrame(sourceVideo, viewer.getBaseDisparity(), 16).catch(() => {});
    }
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
  depthController,
  get videoDepthTimeline() {
    return videoDepthTimeline;
  },
  getDepthTimelineProgress: () => videoDepthTimeline.getProgress(),
  videoElement: sourceVideo,
  motionController,
  spatialWindow,
  importVideo: loadVideo,
  exportVideo: triggerExport,
  exportOffline: (format: ExportFormat, options?: any) =>
    exportOfflineGPUVideo(format, viewer, sourceVideo, options),
  getTelemetry: () => viewer.getTelemetry(),
  get lastTelemetry() {
    return depthController.lastTelemetry;
  },
  get sbsReconstructionMae() {
    return depthController.sbsReconstructionMae;
  },
  get scaleA() {
    return depthController.scaleA;
  },
  get shiftB() {
    return depthController.shiftB;
  },
  get anchorCount() {
    return depthController.anchorCount;
  },
  get inferenceTimeMs() {
    return depthController.inferenceTimeMs;
  },
  hasExternalDepthMap: () => viewer.hasExternalDepthMap(),
  setDepthModel: (v: 'v3' | 'v2') => setDepthModelVersion(v),
  setMotionMode: (mode: 'auto' | 'interactive' | 'manual') => {
    motionMode = mode;
    motionModeSelect.value = mode;
    updateMotionModeUI(mode);
  },
  setManualShift: (shift: number) => {
    motionMode = 'manual';
    motionModeSelect.value = 'manual';
    updateMotionModeUI('manual');
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

// Auto-load DCIM video via query param on localhost e.g. ?dcim=101NIN03/HNI_0062.AVI
if (typeof window !== 'undefined') {
  const urlParams = new URLSearchParams(window.location.search);
  const dcimParam = urlParams.get('dcim');
  if (dcimParam && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    const filename = dcimParam.split('/').pop() || 'HNI_0062.AVI';
    loadVideo({ url: `/api/dcim/file?path=${encodeURIComponent(dcimParam)}`, name: filename });
  }
}
