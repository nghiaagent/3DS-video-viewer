import * as Mp4Muxer from 'mp4-muxer';
import * as WebmMuxer from 'webm-muxer';
import type { WebGLParallaxViewer } from '../renderer/webglViewer.js';
import { getPerspectiveShift } from '../renderer/motionController.js';

export type ExportFormat = 'mp4' | 'webm';

export interface ExportProgress {
  progress: number;
  currentFrame: number;
  totalFrames: number;
}

export interface OfflineExportOptions {
  fps?: number;
  maxDurationSec?: number;
  bitrate?: number;
  onProgress?: (progress: number, currentFrame: number, totalFrames: number) => void;
  triggerDownload?: boolean;
}

export interface ExportResult {
  blob: Blob;
  filename: string;
  mimeType: string;
  codec: string;
  width: number;
  height: number;
  durationSec: number;
}

/**
 * Hardware-accelerated offline WebCodecs video export.
 * Hardcoded strictly to:
 * 1. .mp4 (H.264 / 'avc1.42001f') with mp4-muxer
 * 2. .webm (VP9 / 'vp09.00.10.08') with webm-muxer
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

  const canvas = viewer.getCanvas();
  const width = canvas.width;
  const height = canvas.height;
  const fps = options?.fps ?? 60;
  const bitrate = options?.bitrate ?? 6_000_000;

  // Determine export duration (default to video duration or max 30s)
  const durationSec = Math.min(
    sourceVideo.duration && isFinite(sourceVideo.duration) && sourceVideo.duration > 0
      ? sourceVideo.duration
      : 5.0,
    options?.maxDurationSec ?? 30.0
  );
  const totalFrames = Math.max(1, Math.floor(durationSec * fps));

  let muxer: Mp4Muxer.Muxer<Mp4Muxer.ArrayBufferTarget> | WebmMuxer.Muxer<WebmMuxer.ArrayBufferTarget>;
  let codecString: string;
  let filename: string;
  let mimeType: string;

  if (format === 'mp4') {
    codecString = 'avc1.42001f'; // H.264 Baseline Level 3.1
    filename = `3ds_parallax_2d_${Date.now()}.mp4`;
    mimeType = 'video/mp4';

    muxer = new Mp4Muxer.Muxer({
      target: new Mp4Muxer.ArrayBufferTarget(),
      video: {
        codec: 'avc',
        width,
        height,
      },
      fastStart: 'in-memory',
    });
  } else {
    codecString = 'vp09.00.10.08'; // VP9 Profile 0 Level 1.0
    filename = `3ds_parallax_2d_${Date.now()}.webm`;
    mimeType = 'video/webm';

    muxer = new WebmMuxer.Muxer({
      target: new WebmMuxer.ArrayBufferTarget(),
      video: {
        codec: 'V_VP9',
        width,
        height,
        frameRate: fps,
      },
    });
  }

  // Probe hardware acceleration support with fallback
  let hardwareAccel: HardwareAcceleration = 'prefer-hardware';
  try {
    const configCheck = await VideoEncoder.isConfigSupported({
      codec: codecString,
      width,
      height,
      bitrate,
      framerate: fps,
      hardwareAcceleration: 'prefer-hardware',
    });
    if (!configCheck.supported) {
      hardwareAccel = 'no-preference';
    }
  } catch {
    hardwareAccel = 'no-preference';
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

  encoder.configure({
    codec: codecString,
    width,
    height,
    bitrate,
    framerate: fps,
    hardwareAcceleration: hardwareAccel,
  });

  // Preserve original playback position and pause during offline export
  const originalTime = sourceVideo.currentTime;
  const originalPaused = sourceVideo.paused;
  sourceVideo.pause();

  const dtSec = 1.0 / fps;
  const dtMicros = Math.round(dtSec * 1_000_000);

  try {
    for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
      if (encodeError) throw encodeError;

      const tSec = frameIdx * dtSec;
      sourceVideo.currentTime = Math.min(tSec, (sourceVideo.duration || 1.0) - 0.001);

      await waitForSeek(sourceVideo);

      // Camera shift calculation: 0.28Hz auto-oscillation at this exact timestamp
      const shift = getPerspectiveShift(tSec * 1000.0);

      // Render perspective parallax frame directly on GPU
      await viewer.renderVideoFrame(sourceVideo, shift);

      // Zero-copy capture from WebGL2 canvas backbuffer
      const videoFrame = new VideoFrame(canvas, {
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

    return {
      blob,
      filename,
      mimeType,
      codec: codecString,
      width,
      height,
      durationSec,
    };
  } finally {
    // Restore video state
    sourceVideo.currentTime = originalTime;
    if (!originalPaused) {
      sourceVideo.play().catch(() => {});
    }
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
