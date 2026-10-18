import * as Mp4Muxer from 'mp4-muxer';
import * as WebmMuxer from 'webm-muxer';
import type { WebGLParallaxViewer } from '../renderer/webglViewer.js';
import {
  DEFAULT_ORBIT,
  getOrbitOffset,
  getWindowPose,
  loopedOrbit,
  offsetToRenderParams,
  type MotionOffset,
  type OrbitSettings,
} from '../renderer/motionPath.js';
import { glowColors, smoothGlow, type GlowState } from '../renderer/contentGlow.js';
import { grainSeedForTime } from '../renderer/grain.js';
import { StageRenderer } from './stageRenderer.js';

export type ExportFormat = 'mp4' | 'webm';

/** 'stage': the whole framed window on the dark stage (1920x1080). 'content': only the viewer canvas, as before. */
export type ExportFrame = 'stage' | 'content';

export interface ExportProgress {
  progress: number;
  currentFrame: number;
  totalFrames: number;
}

export interface ExportMotion {
  /** Orbit radius, speed and shape. Defaults to DEFAULT_ORBIT. */
  settings?: OrbitSettings;
  /** Starting angle of the orbit in radians. */
  phase0?: number;
  /** Round the orbit speed to a whole number of orbits per clip so the first and last frames match (seamless loop), when that
   * stays close to the requested speed. Default true. */
  loop?: boolean;
  /** Hold one view direction instead of orbiting. */
  fixed?: MotionOffset;
}

export interface OfflineExportOptions {
  /** Output frame rate. Default 60: the camera moves every frame even though the 3DS footage is about 20 fps. */
  fps?: number;
  maxDurationSec?: number;
  bitrate?: number;
  onProgress?: (progress: number, currentFrame: number, totalFrames: number) => void;
  triggerDownload?: boolean;
  /** Default 'stage'. */
  frame?: ExportFrame;
  /**
   * Camera path. The exported clip always moves (an orbit unless a fixed view is given): a still camera would produce a
   * flat 2D video, which is not what this export is for.
   */
  motion?: ExportMotion;
  /** Frame rate of the source video; the video is only re-seeked when the source frame changes. Default 20. */
  sourceFps?: number;
}

export interface ExportResult {
  blob: Blob;
  filename: string;
  mimeType: string;
  codec: string;
  width: number;
  height: number;
  durationSec: number;
  fps: number;
}

/** Stage export geometry: a 1920x1080 stage with the window at 1.4x its reference size (1344 x 672), centred. */
export const STAGE_WIDTH = 1920;
export const STAGE_HEIGHT = 1080;
/** 2.25: the 960 x 480 window becomes 2160 x 1080, as tall as the video (the sides crop a little at rest, more when tilted). */
export const STAGE_SCALE = 2.25;

/** Codecs to try, best first. 1080p60 needs H.264 level 4.2 / VP9 level 4.1; the older strings are the fallbacks. */
const STAGE_CODECS: Record<ExportFormat, string[]> = {
  mp4: ['avc1.64002a', 'avc1.4d002a', 'avc1.42002a'],
  webm: ['vp09.00.41.08', 'vp09.00.40.08', 'vp09.00.31.08'],
};
/** The canvas-only export is 960x480 at 60 fps, which fits the original levels. */
const CONTENT_CODECS: Record<ExportFormat, string[]> = {
  mp4: ['avc1.42001f'],
  webm: ['vp09.00.10.08'],
};

/**
 * Hardware-accelerated offline WebCodecs video export (.mp4 H.264 via mp4-muxer, .webm VP9 via webm-muxer).
 * By default every frame is the whole stage: glow, shadow, rim and the tilted window around the live 3D picture.
 */
export async function exportOfflineGPUVideo(
  format: ExportFormat,
  viewer: WebGLParallaxViewer,
  sourceVideo: HTMLVideoElement,
  options?: OfflineExportOptions
): Promise<ExportResult> {
  if (typeof VideoEncoder === 'undefined') {
    throw new Error('WebCodecs VideoEncoder is not supported in this browser. Please use Real-Time export.');
  }

  const frameMode: ExportFrame = options?.frame ?? 'stage';
  const fps = options?.fps ?? 60;
  const sourceFps = options?.sourceFps ?? 20;
  // 16 Mbps for the stage: the fine film grain costs bits at 1080p60
  const bitrate = options?.bitrate ?? (frameMode === 'stage' ? 16_000_000 : 6_000_000);

  // Determine export duration (default to video duration or max 30s)
  const durationSec = Math.min(
    sourceVideo.duration && isFinite(sourceVideo.duration) && sourceVideo.duration > 0 ? sourceVideo.duration : 5.0,
    options?.maxDurationSec ?? 30.0
  );
  const totalFrames = Math.max(1, Math.floor(durationSec * fps));

  // Camera path
  const motion = options?.motion;
  const baseOrbit = motion?.settings ?? DEFAULT_ORBIT;
  const orbit: OrbitSettings = motion?.loop === false ? baseOrbit : loopedOrbit(baseOrbit, durationSec);
  const viewDirectionAt = (tSec: number): MotionOffset => motion?.fixed ?? getOrbitOffset(tSec, orbit, motion?.phase0 ?? 0);

  // For the stage the window is 2160 px wide, so render the picture at the viewer's highest upscale
  // (1440 px) instead of stretching the usual 960 px canvas.
  const previousUpscale = viewer.getUpscaleMultiplier();
  if (frameMode === 'stage') viewer.setUpscaleMultiplier(3);
  const canvas = viewer.getCanvas();
  const stage = frameMode === 'stage' ? new StageRenderer(STAGE_WIDTH, STAGE_HEIGHT, STAGE_SCALE) : null;
  const width = stage ? STAGE_WIDTH : canvas.width;
  const height = stage ? STAGE_HEIGHT : canvas.height;

  const codecString = await pickCodec(frameMode === 'stage' ? STAGE_CODECS[format] : CONTENT_CODECS[format], width, height, bitrate, fps);
  const hardwareAccel = await pickHardwareAcceleration(codecString, width, height, bitrate, fps);

  let muxer: Mp4Muxer.Muxer<Mp4Muxer.ArrayBufferTarget> | WebmMuxer.Muxer<WebmMuxer.ArrayBufferTarget>;
  let filename: string;
  let mimeType: string;
  if (format === 'mp4') {
    filename = `3ds_parallax_${frameMode === 'stage' ? 'stage' : '2d'}_${Date.now()}.mp4`;
    mimeType = 'video/mp4';
    muxer = new Mp4Muxer.Muxer({
      target: new Mp4Muxer.ArrayBufferTarget(),
      video: { codec: 'avc', width, height },
      fastStart: 'in-memory',
    });
  } else {
    filename = `3ds_parallax_${frameMode === 'stage' ? 'stage' : '2d'}_${Date.now()}.webm`;
    mimeType = 'video/webm';
    muxer = new WebmMuxer.Muxer({
      target: new WebmMuxer.ArrayBufferTarget(),
      video: { codec: 'V_VP9', width, height, frameRate: fps },
    });
  }

  let encodeError: Error | null = null;
  const encoder = new VideoEncoder({
    output: (chunk, meta) => {
      if (format === 'mp4') {
        (muxer as Mp4Muxer.Muxer<Mp4Muxer.ArrayBufferTarget>).addVideoChunk(chunk, meta);
      } else {
        (muxer as WebmMuxer.Muxer<WebmMuxer.ArrayBufferTarget>).addVideoChunk(chunk, meta);
      }
    },
    error: (e) => {
      console.error('[WebCodecs Encoder Error]', e);
      encodeError = e;
    },
  });
  encoder.configure({ codec: codecString, width, height, bitrate, framerate: fps, hardwareAcceleration: hardwareAccel });

  // Preserve original playback position and pause during offline export
  const originalTime = sourceVideo.currentTime;
  const originalPaused = sourceVideo.paused;
  sourceVideo.pause();

  const dtSec = 1.0 / fps;
  const dtMicros = Math.round(dtSec * 1_000_000);
  const lastSourceFrame = Math.max(0, Math.floor((sourceVideo.duration || 1) * sourceFps) - 1);
  let currentSourceFrame = -1;
  let glow: GlowState | null = null; // follows the picture's colour, smoothed in export time

  try {
    for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
      if (encodeError) throw encodeError;

      const tSec = frameIdx * dtSec;

      // Seek only when the source frame changes (the footage is ~20 fps, the export 60)
      const sourceFrame = Math.min(lastSourceFrame, Math.round(tSec * sourceFps));
      if (sourceFrame !== currentSourceFrame) {
        sourceVideo.currentTime = Math.min(sourceFrame / sourceFps, (sourceVideo.duration || 1.0) - 0.001);
        await waitForSeek(sourceVideo);
        currentSourceFrame = sourceFrame;
      }

      // Camera for this exact timestamp, from the same motion path the live viewer uses
      const direction = viewDirectionAt(tSec);
      const { viewShift, verticalTilt } = offsetToRenderParams(direction);
      viewer.setGrainSeed(grainSeedForTime(sourceFrame / sourceFps, sourceFps));
      await viewer.renderVideoFrame(sourceVideo, viewShift, verticalTilt);

      const glowTarget = glowColors(viewer.getContentAverage());
      glow = glow ? smoothGlow(glow, glowTarget, dtSec) : glowTarget;
      const source: HTMLCanvasElement = stage ? stage.render(canvas, getWindowPose(direction), glow) : canvas;
      const videoFrame = new VideoFrame(source, {
        timestamp: Math.round(tSec * 1_000_000),
        duration: dtMicros,
      });

      const isKeyFrame = frameIdx % fps === 0;
      encoder.encode(videoFrame, { keyFrame: isKeyFrame });
      videoFrame.close();

      options?.onProgress?.((frameIdx + 1) / totalFrames, frameIdx + 1, totalFrames);
    }

    await encoder.flush();
    encoder.close();
    muxer.finalize();

    const buffer = muxer.target.buffer;
    const blob = new Blob([buffer], { type: mimeType });

    if (options?.triggerDownload) {
      downloadBlob(blob, filename);
    }

    return { blob, filename, mimeType, codec: codecString, width, height, durationSec, fps };
  } finally {
    stage?.destroy();
    if (frameMode === 'stage') viewer.setUpscaleMultiplier(previousUpscale);
    // Restore video state
    sourceVideo.currentTime = originalTime;
    if (!originalPaused) {
      sourceVideo.play().catch(() => {});
    }
  }
}

/** First codec string the encoder accepts for this size and rate. */
async function pickCodec(candidates: string[], width: number, height: number, bitrate: number, fps: number): Promise<string> {
  for (const codec of candidates) {
    for (const hardwareAcceleration of ['prefer-hardware', 'no-preference'] as HardwareAcceleration[]) {
      try {
        const check = await VideoEncoder.isConfigSupported({ codec, width, height, bitrate, framerate: fps, hardwareAcceleration });
        if (check.supported) return codec;
      } catch {
        // try the next one
      }
    }
  }
  throw new Error(`No supported video codec for ${width}x${height} at ${fps} fps (tried ${candidates.join(', ')})`);
}

async function pickHardwareAcceleration(codec: string, width: number, height: number, bitrate: number, fps: number): Promise<HardwareAcceleration> {
  try {
    const check = await VideoEncoder.isConfigSupported({ codec, width, height, bitrate, framerate: fps, hardwareAcceleration: 'prefer-hardware' });
    return check.supported ? 'prefer-hardware' : 'no-preference';
  } catch {
    return 'no-preference';
  }
}

function waitForSeek(video: HTMLVideoElement): Promise<void> {
  return new Promise<void>((resolve) => {
    if (video.seeking) {
      const onSeeked = () => {
        video.removeEventListener('seeked', onSeeked);
        resolve();
      };
      video.addEventListener('seeked', onSeeked);
    } else {
      resolve();
    }
  });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 1000);
}
