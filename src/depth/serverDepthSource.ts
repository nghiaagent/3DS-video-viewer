/**
 * Client for the local DA3 depth server (server/serve.py).
 *
 * The server runs every frame of the clip, both eyes, through DepthAnything3 in windowed joint passes with one
 * clip-global normalization, so depth is consistent over time. It only knows files under its DCIM root, so this
 * path is used for DCIM URLs (`?dcim=`) and for files picked in the browser, which are uploaded; anything else
 * (or any server failure) falls back to the in-browser ONNX pipeline.
 */

export interface ServerDepth {
  frames: Uint8Array[];
  fps: number;
  width: number;
  height: number;
}

const DCIM_FILE_PATH = '/api/dcim/file';

/** Extracts the DCIM-relative path from an `/api/dcim/file?path=...` URL, or null for any other source. */
export function dcimPathFromUrl(url: string): string | null {
  try {
    const parsed = new URL(url, window.location.origin);
    if (parsed.pathname !== DCIM_FILE_PATH) return null;
    return parsed.searchParams.get('path');
  } catch {
    return null;
  }
}

/** True when the local DA3 server answers /api/health (a SPA fallback page or a 404 does not count). */
export async function serverAvailable(timeoutMs = 800): Promise<boolean> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch('/api/health', { signal: ctrl.signal });
    if (!res.ok) return false;
    const body = (await res.json()) as { model?: string };
    return typeof body.model === 'string';
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

async function parseDepthResponse(res: Response): Promise<ServerDepth> {
  if (!res.ok) {
    throw new Error(`Depth server returned ${res.status}`);
  }
  const buf = await res.arrayBuffer();
  const headerLen = new DataView(buf).getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, headerLen))) as {
    frames: number;
    width: number;
    height: number;
    fps: number;
  };
  const frameBytes = header.width * header.height;
  const base = 4 + headerLen;
  if (buf.byteLength < base + header.frames * frameBytes) {
    throw new Error('Depth server response is truncated');
  }
  const frames: Uint8Array[] = [];
  for (let i = 0; i < header.frames; i++) {
    frames.push(new Uint8Array(buf, base + i * frameBytes, frameBytes));
  }
  return { frames, fps: header.fps, width: header.width, height: header.height };
}

/** Fetches all depth frames for a DCIM clip: [u32 LE header length][JSON header][frames * width * height uint8]. */
export async function fetchServerDepth(dcimPath: string): Promise<ServerDepth> {
  return parseDepthResponse(await fetch(`/api/depth?path=${encodeURIComponent(dcimPath)}`));
}

/** Uploads a user-picked .AVI to the server (streamed to disk there) and returns its depth sequence. */
export async function fetchServerDepthForFile(file: Blob): Promise<ServerDepth> {
  return parseDepthResponse(
    await fetch('/api/depth/upload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: file,
    })
  );
}

export interface ServerProgress {
  phase: string;
  done: number;
  total: number;
}

/** Polls /api/progress while a depth request is running; returns a function that stops the polling. */
export function pollServerProgress(onProgress: (p: ServerProgress) => void, intervalMs = 700): () => void {
  let stopped = false;
  const tick = async () => {
    if (stopped) return;
    try {
      const res = await fetch('/api/progress');
      if (res.ok && !stopped) onProgress((await res.json()) as ServerProgress);
    } catch {
      // the server may be busy or restarting; keep polling quietly
    }
    if (!stopped) setTimeout(tick, intervalMs);
  };
  void tick();
  return () => {
    stopped = true;
  };
}
