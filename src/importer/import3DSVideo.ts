import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';
import { parse3DSMetadata, type ThreeDSMetadata } from './parse3DSMetadata.js';

export interface VideoSourceInput {
  url: string;
  name: string;
}

export type VideoSource = File | VideoSourceInput;

let ffmpegInstance: FFmpeg | null = null;
let ffmpegLoadingPromise: Promise<FFmpeg> | null = null;

export async function getFFmpeg(onProgress?: (progress: number) => void): Promise<FFmpeg> {
  if (ffmpegInstance && ffmpegInstance.loaded) {
    return ffmpegInstance;
  }
  if (ffmpegLoadingPromise) {
    return ffmpegLoadingPromise;
  }

  ffmpegLoadingPromise = (async () => {
    try {
      const ffmpeg = new FFmpeg();
      ffmpeg.on('progress', ({ progress }) => {
        onProgress?.(progress);
      });
      ffmpeg.on('log', ({ message }) => {
        if (message.includes('error') || message.includes('Error')) {
          console.warn('[FFmpeg]', message);
        }
      });

      // Robust URL resolution: base on window.location.href so subpaths (e.g. GitHub Pages /3DS-video-viewer/) work
      const coreUrl = new URL('./ffmpeg/ffmpeg-core.js', window.location.href).href;
      const wasmUrl = new URL('./ffmpeg/ffmpeg-core.wasm', window.location.href).href;

      // Safe fetch helper verifying res.ok before creating Blob URL to prevent converting 404 HTML pages into JS blobs
      async function safeToBlobURL(url: string, mimeType: string): Promise<string> {
        const res = await fetch(url);
        if (!res.ok) {
          throw new Error(`Failed to load ${url}: HTTP ${res.status} ${res.statusText}`);
        }
        const blob = await res.blob();
        return URL.createObjectURL(new Blob([await blob.arrayBuffer()], { type: mimeType }));
      }

      const [coreBlobUrl, wasmBlobUrl] = await Promise.all([
        safeToBlobURL(coreUrl, 'text/javascript'),
        safeToBlobURL(wasmUrl, 'application/wasm'),
      ]);

      await ffmpeg.load({
        coreURL: coreBlobUrl,
        wasmURL: wasmBlobUrl,
      });

      ffmpegInstance = ffmpeg;
      return ffmpeg;
    } catch (err) {
      ffmpegLoadingPromise = null;
      throw err;
    }
  })();

  return ffmpegLoadingPromise;
}

/**
 * Imports a Nintendo 3DS stereoscopic recording (.AVI dual-stream MJPEG)
 * and connects it to the target HTML5 <video> element.
 */
export async function import3DSVideo(
  source: VideoSource,
  videoElement: HTMLVideoElement,
  onProgress?: (stage: string, ratio: number) => void
): Promise<{ filename: string; format: 'transcoded_avi'; objectUrl: string; metadata: ThreeDSMetadata }> {
  let fileName = '';
  let ext = '';

  const isFileInput = (typeof File !== 'undefined' && source instanceof File) || ('name' in source && !('url' in source));
  if (isFileInput) {
    fileName = source.name;
    ext = (source.name || '').toLowerCase().split('.').pop() || '';
  } else {
    fileName = source.name || '';
    const rawUrl = (source.url || '').split('?')[0].toLowerCase();
    const urlExt = rawUrl.split('.').pop() || '';
    const nameExt = fileName.toLowerCase().split('.').pop() || '';
    if (urlExt === 'avi' || nameExt === 'avi') {
      ext = 'avi';
    } else {
      ext = urlExt || nameExt;
    }
  }

  // Strictly enforce .AVI source format (no scope creep)
  if (ext !== 'avi') {
    throw new Error('Only Nintendo 3DS .AVI recordings are supported.');
  }

  // Dual-stream Left/Right MJPEG 3DS .AVI transcoding path
  onProgress?.('Initializing FFmpeg...', 0.1);
  const ffmpeg = await getFFmpeg((ratio) => {
    onProgress?.('Transcoding 3DS dual-stream AVI...', 0.3 + Math.min(1.0, Math.max(0.0, ratio)) * 0.6);
  });

  onProgress?.('Fetching AVI file...', 0.2);
  const fileSource = source instanceof File ? source : (typeof window !== 'undefined' ? new URL(source.url, window.location.href).href : source.url);
  const fileData = await fetchFile(fileSource);
  const metadata = parse3DSMetadata(fileData);

  const inputName = `input_${Date.now()}.avi`;
  const outputName = `output_${Date.now()}.mp4`;

  await ffmpeg.writeFile(inputName, fileData);

  onProgress?.('Transcoding dual MJPEG streams to SBS MP4...', 0.3);
  const exitCode = await ffmpeg.exec([
    '-i',
    inputName,
    '-filter_complex',
    '[0:v:0][0:v:1]hstack=inputs=2[v]',
    '-map',
    '[v]',
    '-map',
    '0:a:0?',
    '-c:v',
    'libx264',
    '-pix_fmt',
    'yuv420p',
    '-preset',
    'ultrafast',
    '-crf',
    '18',
    '-c:a',
    'aac',
    outputName,
  ]);

  if (exitCode !== 0) {
    throw new Error(`FFmpeg transcoding failed with exit code ${exitCode}`);
  }

  const outputData = await ffmpeg.readFile(outputName);
  const rawBytes = outputData as Uint8Array;
  const copy = new Uint8Array(rawBytes.length);
  copy.set(rawBytes);
  const blob = new Blob([copy.buffer], { type: 'video/mp4' });
  const objectUrl = URL.createObjectURL(blob);

  // Clean up FFmpeg virtual filesystem
  try {
    await ffmpeg.deleteFile(inputName);
    await ffmpeg.deleteFile(outputName);
  } catch (e) {
    console.warn('Error cleaning temporary FFmpeg files:', e);
  }

  videoElement.src = objectUrl;
  videoElement.loop = true;

  await waitForVideoLoaded(videoElement);

  try {
    await videoElement.play();
  } catch {
    videoElement.muted = true;
    await videoElement.play();
  }

  onProgress?.('Ready', 1.0);
  return {
    filename: fileName,
    format: 'transcoded_avi',
    objectUrl,
    metadata,
  };
}

function waitForVideoLoaded(video: HTMLVideoElement): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      resolve();
      return;
    }

    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(new Error(`Failed to load video: ${video.error?.message || 'unknown error'}`));
    };

    const cleanup = () => {
      video.removeEventListener('loadeddata', onLoaded);
      video.removeEventListener('error', onError);
    };

    video.addEventListener('loadeddata', onLoaded);
    video.addEventListener('error', onError);
    video.load();
  });
}
