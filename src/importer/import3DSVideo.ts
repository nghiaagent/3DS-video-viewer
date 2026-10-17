import { FFmpeg } from '@ffmpeg/ffmpeg';
import { fetchFile } from '@ffmpeg/util';

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
    const ffmpeg = new FFmpeg();
    ffmpeg.on('progress', ({ progress }) => {
      onProgress?.(progress);
    });
    ffmpeg.on('log', ({ message }) => {
      if (message.includes('error') || message.includes('Error')) {
        console.warn('[FFmpeg]', message);
      }
    });

    // Core files placed in public/ffmpeg/ for reliable same-origin loading
    await ffmpeg.load({
      coreURL: './ffmpeg/ffmpeg-core.js',
      wasmURL: './ffmpeg/ffmpeg-core.wasm',
    });

    ffmpegInstance = ffmpeg;
    return ffmpeg;
  })();

  return ffmpegLoadingPromise;
}

/**
 * Imports any 3DS stereoscopic recording (.mp4, .webm, or raw dual-stream .AVI)
 * and connects it to the target HTML5 <video> element.
 */
export async function import3DSVideo(
  source: VideoSource,
  videoElement: HTMLVideoElement,
  onProgress?: (stage: string, ratio: number) => void
): Promise<{ filename: string; format: 'sbs_mp4' | 'sbs_webm' | 'transcoded_avi'; objectUrl: string }> {
  const fileName = source instanceof File ? source.name : source.name;
  const ext = fileName.toLowerCase().split('.').pop() || '';

  // 1. FAST PATH: Standard Side-by-Side MP4 or WebM
  if (ext === 'mp4' || ext === 'webm') {
    onProgress?.('Loading video...', 0.3);
    const objectUrl = source instanceof File ? URL.createObjectURL(source) : source.url;

    videoElement.src = objectUrl;
    videoElement.loop = true;

    await waitForVideoLoaded(videoElement);

    try {
      await videoElement.play();
    } catch {
      // Autoplay with sound might be blocked by browser policy
      videoElement.muted = true;
      await videoElement.play();
    }

    onProgress?.('Ready', 1.0);
    return {
      filename: fileName,
      format: ext === 'mp4' ? 'sbs_mp4' : 'sbs_webm',
      objectUrl,
    };
  }

  // 2. RAW 3DS .AVI PATH (Dual-stream Left/Right MJPEG)
  if (ext === 'avi') {
    onProgress?.('Initializing FFmpeg...', 0.1);
    const ffmpeg = await getFFmpeg((ratio) => {
      onProgress?.('Transcoding 3DS dual-stream AVI...', 0.3 + Math.min(1.0, Math.max(0.0, ratio)) * 0.6);
    });

    onProgress?.('Fetching AVI file...', 0.2);
    const fileSource = source instanceof File ? source : source.url;
    const fileData = await fetchFile(fileSource);

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
    };
  }

  throw new Error(`Unsupported video format: .${ext}. Please provide .AVI, .MP4, or .WEBM.`);
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
