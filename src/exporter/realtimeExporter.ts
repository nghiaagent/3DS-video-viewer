import type { ExportFormat } from './offlineExporter.js';
import { downloadBlob } from './offlineExporter.js';

export interface RealtimeExportSession {
  stop: () => Promise<{ blob: Blob; filename: string; mimeType: string }>;
  isRecording: () => boolean;
}

export function startRealtimeGPUVideo(
  format: ExportFormat,
  webglCanvas: HTMLCanvasElement,
  sourceVideoElement?: HTMLVideoElement,
  options?: { triggerDownload?: boolean; bitrate?: number }
): RealtimeExportSession {
  // Capture 60fps stream straight from WebGL2 backbuffer
  const canvasStream = webglCanvas.captureStream(60);

  // If source video has audio, combine audio track into stream
  if (sourceVideoElement) {
    try {
      const videoStream = (sourceVideoElement as any).captureStream
        ? (sourceVideoElement as any).captureStream()
        : (sourceVideoElement as any).mozCaptureStream
        ? (sourceVideoElement as any).mozCaptureStream()
        : null;

      if (videoStream) {
        const audioTracks = videoStream.getAudioTracks();
        for (const track of audioTracks) {
          canvasStream.addTrack(track);
        }
      }
    } catch {
      // Audio capture stream might not be permitted for cross-origin or certain elements
    }
  }

  // Determine best supported MIME type
  let mimeType = '';
  let filename = '';

  if (format === 'mp4') {
    filename = `3ds_realtime_parallax_${Date.now()}.mp4`;
    const candidates = [
      'video/mp4;codecs=avc1.42001f,mp4a.40.2',
      'video/mp4;codecs=avc1',
      'video/mp4',
    ];
    mimeType = candidates.find((c) => MediaRecorder.isTypeSupported(c)) || '';
    if (!mimeType) {
      console.warn('Browser MediaRecorder does not support video/mp4, falling back to video/webm');
      mimeType = 'video/webm;codecs=vp9';
      filename = `3ds_realtime_parallax_${Date.now()}.webm`;
    }
  } else {
    filename = `3ds_realtime_parallax_${Date.now()}.webm`;
    const candidates = [
      'video/webm;codecs=vp09.00.10.08,opus',
      'video/webm;codecs=vp9,opus',
      'video/webm;codecs=vp9',
      'video/webm',
    ];
    mimeType = candidates.find((c) => MediaRecorder.isTypeSupported(c)) || 'video/webm';
  }

  const recordedChunks: Blob[] = [];
  const recorder = new MediaRecorder(canvasStream, {
    mimeType,
    videoBitsPerSecond: options?.bitrate ?? 6_000_000,
  });

  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) {
      recordedChunks.push(e.data);
    }
  };

  recorder.start(100); // 100ms slice interval

  return {
    isRecording: () => recorder.state === 'recording',
    stop: () => {
      return new Promise<{ blob: Blob; filename: string; mimeType: string }>((resolve, reject) => {
        if (recorder.state === 'inactive') {
          const blob = new Blob(recordedChunks, { type: mimeType });
          resolve({ blob, filename, mimeType });
          return;
        }

        recorder.onstop = () => {
          const blob = new Blob(recordedChunks, { type: mimeType });
          if (options?.triggerDownload) {
            downloadBlob(blob, filename);
          }
          resolve({ blob, filename, mimeType });
        };

        recorder.onerror = (e) => {
          reject(e);
        };

        recorder.stop();
      });
    },
  };
}
