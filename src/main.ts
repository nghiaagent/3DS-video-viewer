import './style.css';
import { WebGLParallaxViewer, type RenderMode, type DisparityBackend, type UpscaleMultiplier } from './renderer/webglViewer.js';
import { MotionController } from './renderer/motionController.js';
import { import3DSVideo, type VideoSource } from './importer/import3DSVideo.js';
import {
  exportParallaxVideo,
  exportOfflineGPUVideo,
  type ExportFormat,
  type ExportFrame,
  type ExportMode,
} from './exporter/index.js';
import { StageRenderer } from './exporter/stageRenderer.js';
import { DepthAnythingController, VideoDepthTimeline } from './depth/stereoGuidedDepth.js';
import {
  DEFAULT_ORBIT,
  getOrbitOffsetFromPhase,
  getWindowPose,
  offsetToRenderParams,
  viewShiftToX,
  type MotionMode,
  type MotionOffset,
  type OrbitSettings,
} from './renderer/motionPath.js';
import {
  dcimPathFromUrl,
  fetchServerDepth,
  fetchServerDepthForFile,
  pollServerProgress,
  serverAvailable,
} from './depth/serverDepthSource.js';

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
const exportFrameSelect = document.getElementById('export-frame-select') as HTMLSelectElement | null;

const backendSelect = document.getElementById('backend-select') as HTMLSelectElement | null;
const renderModeSelect = document.getElementById('render-mode-select') as HTMLSelectElement;
const upscaleSelect = document.getElementById('upscale-select') as HTMLSelectElement | null;
const motionModeSelect = document.getElementById('motion-mode-select') as HTMLSelectElement;

const speedDockControl = document.getElementById('speed-dock-control') as HTMLElement | null;
const manualShiftGroup = document.getElementById('manual-shift-group') as HTMLElement | null;
const manualShiftSlider = document.getElementById('manual-shift-slider') as HTMLInputElement | null;
const manualShiftVal = document.getElementById('manual-shift-val') as HTMLElement | null;
const manualTiltSlider = document.getElementById('manual-tilt-slider') as HTMLInputElement | null;
const orbitSizeDockControl = document.getElementById('orbit-size-dock-control') as HTMLElement | null;

const cropSlider = document.getElementById('crop-slider') as HTMLInputElement | null;
const maxdispSlider = document.getElementById('maxdisp-slider') as HTMLInputElement;
const maxdispVal = document.getElementById('maxdisp-val') as HTMLElement | null;
const focusSlider = document.getElementById('focus-slider') as HTMLInputElement | null;
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
  { modelVersion: 'v3', inputSize: 392, expandRadius: 1, dilateRadius: 1 },
  (calibratedDepth, width, height, _telemetry) => {
    viewer.uploadExternalDepthMap(calibratedDepth, width, height);
  }
);
// No eager depthController.init(): with the local DA3 server supplying depth the in-browser model is never needed,
// and processVideoFrame() initializes it on demand when a frame has no depth (the ONNX fallback path).

let videoDepthTimeline: VideoDepthTimeline = new VideoDepthTimeline({
  fps: 20,
  inputSize: 392,
});
depthController.setTimeline(videoDepthTimeline);

// Register frame-accurate synchronous depth provider (0.0 ms desync)
viewer.setDepthProvider((timeSec: number) => {
  if (!videoDepthTimeline) return null;
  const depth = videoDepthTimeline.getDepthAtTime(timeSec, 'v3');
  if (depth) {
    return { depth, width: 480, height: 240 };
  }
  return null;
});

// True while the local DA3 server is computing depth for the clip being loaded; the in-browser depth controller must
// stay idle then (it would otherwise load its own model and run per-frame inference on an empty timeline).
let serverDepthPending = false;
let motionMode: MotionMode = 'interactive';
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

    // Initial timeline pre-computation across video stream for V3
    if (videoDepthTimeline) {
      videoDepthTimeline.destroy();
    }
    videoDepthTimeline = new VideoDepthTimeline({
      fps: 20,
      inputSize: 392,
      baseDisp: viewer.getBaseDisparity() ?? 0,
    });
    depthController.setTimeline(videoDepthTimeline);

    // Preferred: local DA3 server (all frames, both eyes, joint passes, one global normalization).
    // DCIM clips are read by the server directly and picked files are uploaded to it; anything else, or a server
    // failure, uses the in-browser ONNX pipeline below.
    let usedServerDepth = false;
    const sourceUrl = 'url' in sourceInput ? sourceInput.url : null;
    const dcimPath = sourceUrl ? dcimPathFromUrl(sourceUrl) : null;
    const uploadFile = sourceInput instanceof File ? sourceInput : null;
    if ((dcimPath || uploadFile) && (await serverAvailable())) {
      serverDepthPending = true;
      let stopProgressPolling = () => {};
      try {
        setImportProgress('Computing depth on local server (DA3, all frames)...', 0.5);
        stopProgressPolling = pollServerProgress(({ phase, done, total }) => {
          if (phase === 'inferring' && total > 0) {
            setImportProgress(`Computing depth on local server (window ${done}/${total})...`, 0.5 + 0.45 * (done / total));
          } else if (phase === 'decoding' || phase === 'finishing') {
            setImportProgress(`Computing depth on local server (${phase})...`, 0.5);
          }
        });
        const serverDepth = dcimPath ? await fetchServerDepth(dcimPath) : await fetchServerDepthForFile(uploadFile!);
        videoDepthTimeline.loadPrecomputed(
          serverDepth.frames,
          sourceVideo.duration,
          serverDepth.width,
          serverDepth.height
        );
        usedServerDepth = true;
      } catch (err) {
        console.warn('[ViewerApp] Local depth server failed, falling back to in-browser ONNX:', err);
      } finally {
        stopProgressPolling();
        serverDepthPending = false;
      }
    }

    const isE2E =
      typeof navigator !== 'undefined' &&
      (Boolean((navigator as any).webdriver) || /HeadlessChrome/.test(navigator.userAgent));
    const initialKeyframes = isE2E ? 2 : 12;
    const timelineTimeoutMs = isE2E ? 12_000 : 30_000;

    if (!usedServerDepth) {
      await Promise.race([
        videoDepthTimeline.buildInitialTimeline(
          sourceVideo,
          viewer.getBaseDisparity(),
          (stage, ratio) => {
            const displayStage = stage.includes('Analyzing') ? 'Analyzing 3D stereo depth (V3)...' : stage;
            setImportProgress(displayStage, ratio);
          },
          {
            initialKeyframes,
            runFullStreamInBackground: true,
          }
        ),
        new Promise<void>((resolve) => setTimeout(resolve, timelineTimeoutMs)),
      ]);
    }

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

// Depth slider = parallax strength (how far layers move against each other)
maxdispSlider.addEventListener('input', () => {
  const val = parseFloat(maxdispSlider.value);
  if (maxdispVal) maxdispVal.textContent = val.toFixed(2);
  viewer.setSensitivity(val);
});

// Focus slider = the depth that stays still while everything else moves
focusSlider?.addEventListener('input', () => viewer.setFocus(parseFloat(focusSlider.value)));

function updateMotionModeUI(mode: MotionMode) {
  const show = (el: HTMLElement | null, on: boolean) => {
    if (el) el.style.display = on ? 'flex' : 'none';
  };
  show(manualShiftGroup, mode === 'manual');
  show(speedDockControl, mode === 'orbit');
  show(orbitSizeDockControl, mode === 'orbit');
  if (mode === 'interactive') {
    motionController.attach(viewportContainer);
  } else {
    motionController.detach();
  }
  if (mode === 'orbit') {
    orbitAmpCurrent = 0; // the circle grows out of the centre instead of snapping to a point on it
  }
}

motionModeSelect.addEventListener('change', () => {
  motionMode = motionModeSelect.value as MotionMode;
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
// A real-time recording is in progress (the export buttons act as "stop" while it runs)
let realtimeActive = false;

/** Orbit used for the exported camera: the current Orbit settings when in Orbit mode, otherwise the defaults. */
function exportOrbitSettings(): OrbitSettings {
  if (motionMode === 'orbit') {
    return {
      freqHz: freqSlider ? parseFloat(freqSlider.value) : DEFAULT_ORBIT.freqHz,
      amp: ampSlider ? parseFloat(ampSlider.value) : DEFAULT_ORBIT.amp,
      yRatio: DEFAULT_ORBIT.yRatio,
    };
  }
  return DEFAULT_ORBIT;
}

async function triggerExport(format: ExportFormat) {
  if (isExporting || realtimeActive) return;

  const activeBtn = format === 'mp4' ? exportMp4Btn : exportWebmBtn;
  const mode: ExportMode = (exportModeSelect?.value as ExportMode) || 'offline';

  if (mode === 'realtime') {
    // Records the live canvas (content only). The live render loop must keep running, so isExporting stays false.
    realtimeActive = true;
    activeBtn.classList.add('is-exporting');
    setExportProgress(activeBtn, 1);
    try {
      const session = (await exportParallaxVideo(format, 'realtime', viewer, sourceVideo, {
        triggerDownload: true,
      })) as any;

      const onStopClick = async () => {
        await session.stop();
        cleanup();
      };
      const cleanup = () => {
        realtimeActive = false;
        activeBtn.classList.remove('is-exporting');
        setExportProgress(activeBtn, 0);
        exportMp4Btn.removeEventListener('click', onStopClick);
        exportWebmBtn.removeEventListener('click', onStopClick);
      };
      exportMp4Btn.addEventListener('click', onStopClick, { once: true });
      exportWebmBtn.addEventListener('click', onStopClick, { once: true });
    } catch (err: unknown) {
      console.error('[Export Failed]', err);
      realtimeActive = false;
      activeBtn.classList.remove('is-exporting');
      setExportProgress(activeBtn, 0);
    }
    return;
  }

  // Offline export renders every frame itself, so the live loop is paused for its duration
  isExporting = true;
  exportMp4Btn.disabled = true;
  exportWebmBtn.disabled = true;
  activeBtn.classList.add('is-exporting');
  setExportProgress(activeBtn, 0);

  try {
    await exportParallaxVideo(format, 'offline', viewer, sourceVideo, {
      fps: 60,
      maxDurationSec: 30,
      triggerDownload: true,
      frame: (exportFrameSelect?.value as ExportFrame) || 'stage',
      motion: { settings: exportOrbitSettings() },
      onProgress: (p) => {
        setExportProgress(activeBtn, p);
      },
    });
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

// Orbit state: integrate the phase (so changing speed never makes the view jump) and ease the radius in
// (so switching modes grows the circle smoothly).
let lastTimestampMs = 0;
let orbitPhase = 0;
let orbitAmpCurrent = 0;
const ORBIT_AMP_EASE_SEC = 0.4;

// Test/debug hook: pin the view direction (null = normal behaviour)
let motionOverride: MotionOffset | null = null;

/** Current view direction for the selected motion mode: x, y in [-1, 1]. */
function resolveMotion(timestampMs: number, dtSeconds: number): MotionOffset {
  if (motionOverride) return motionOverride;
  if (motionMode === 'manual') {
    return {
      x: viewShiftToX(manualShiftSlider ? parseFloat(manualShiftSlider.value) : 0.5),
      y: manualTiltSlider ? parseFloat(manualTiltSlider.value) : 0,
    };
  }
  if (motionMode === 'interactive') {
    const shift = motionController.update(timestampMs);
    return { x: viewShiftToX(shift), y: motionController.getVerticalTilt() };
  }
  const freq = freqSlider ? parseFloat(freqSlider.value) : DEFAULT_ORBIT.freqHz;
  const targetAmp = ampSlider ? parseFloat(ampSlider.value) : DEFAULT_ORBIT.amp;
  orbitPhase = (orbitPhase + dtSeconds * freq * Math.PI * 2.0) % (Math.PI * 2.0);
  orbitAmpCurrent += (targetAmp - orbitAmpCurrent) * (1 - Math.exp(-dtSeconds / ORBIT_AMP_EASE_SEC));
  return getOrbitOffsetFromPhase(orbitPhase, orbitAmpCurrent, DEFAULT_ORBIT.yRatio);
}

// Main Animation & Parallax Render Loop (60 FPS) with Apple Vision Pro 3D Window Movement
function renderLoop(timestampMs: number) {
  if (!isLoopRunning) return;

  if (!lastTimestampMs) {
    lastTimestampMs = timestampMs;
  }
  const dtSeconds = Math.min(Math.max(0, (timestampMs - lastTimestampMs) / 1000.0), 0.1);
  lastTimestampMs = timestampMs;

  const offset = resolveMotion(timestampMs, dtSeconds);
  const { viewShift, verticalTilt } = offsetToRenderParams(offset);

  // Spatial window movement and reactive specular border (one pose function drives both axes)
  if (spatialWindow) {
    const pose = getWindowPose(offset);
    spatialWindow.style.transform =
      `translate3d(${pose.tx.toFixed(2)}px, ${pose.ty.toFixed(2)}px, 0px) ` +
      `rotateY(${pose.rotateYDeg.toFixed(2)}deg) rotateX(${pose.rotateXDeg.toFixed(2)}deg)`;
    spatialWindow.style.setProperty('--window-shift', pose.x.toFixed(3));
    spatialWindow.style.setProperty('--window-shift-y', pose.y.toFixed(3));
    spatialWindow.style.boxShadow =
      `${pose.shadowX.toFixed(1)}px ${pose.shadowY.toFixed(1)}px 80px rgba(0, 0, 0, 0.85), ` +
      '0 0 90px rgba(56, 189, 248, 0.08), 0 0 0 1px rgba(255, 255, 255, 0.14)';

    if (ambientGlow) {
      ambientGlow.style.transform = `translate3d(${pose.glowX.toFixed(1)}px, ${pose.glowY.toFixed(1)}px, 0px)`;
    }
  }

  if (shiftPin) {
    shiftPin.style.left = `${(viewShift * 100).toFixed(1)}%`;
  }

  // Render on WebGL2 GPU
  if (sourceVideo.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && !isExporting) {
    viewer.renderVideoFrame(sourceVideo, viewShift, verticalTilt);

    if (!serverDepthPending && !depthController.isBusy()) {
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
motionModeSelect.value = motionMode;
updateMotionModeUI(motionMode); // show only the controls that belong to the default mode
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
  setMotionOverride: (offset: MotionOffset | null) => {
    motionOverride = offset;
  },
  /** Renders the export stage for the pinned view direction at the given size; returns a PNG data URL. */
  renderStageSnapshot: (width: number, height: number, scale: number): string => {
    const stage = new StageRenderer(width, height, scale);
    const canvasOut = stage.render(canvas, getWindowPose(motionOverride ?? { x: 0, y: 0 }));
    const url = canvasOut.toDataURL('image/png');
    stage.destroy();
    return url;
  },
  hasExternalDepthMap: () => viewer.hasExternalDepthMap(),
  setDepthModel: (v: 'v3') => depthController.setModelVersion(v),
  setMotionMode: (mode: MotionMode) => {
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
