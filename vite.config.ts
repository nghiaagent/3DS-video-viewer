import { defineConfig, type Plugin } from 'vite';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';

// Folder holding the 3DS DCIM recordings; override with the DCIM_ROOT environment variable.
const DCIM_ROOT = path.resolve(process.env.DCIM_ROOT ?? path.join(os.homedir(), 'Downloads/3DS vids/DCIM'));

interface DcimFileItem {
  name: string;
  relativePath: string;
  folder: string;
  sizeBytes: number;
  mtime: number;
}

function scanDcimDirectory(dir: string, baseDir: string = dir): DcimFileItem[] {
  const results: DcimFileItem[] = [];
  if (!fs.existsSync(dir)) return results;

  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...scanDcimDirectory(fullPath, baseDir));
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name).toUpperCase();
      if (ext === '.AVI' || ext === '.MPO') {
        const stats = fs.statSync(fullPath);
        const relativePath = path.relative(baseDir, fullPath);
        const folder = path.dirname(relativePath);
        results.push({
          name: entry.name,
          relativePath,
          folder: folder === '.' ? '' : folder,
          sizeBytes: stats.size,
          mtime: stats.mtimeMs,
        });
      }
    }
  }
  return results;
}

function getMimeType(filePath: string): string {
  const ext = path.extname(filePath).toUpperCase();
  switch (ext) {
    case '.AVI':
      return 'video/x-msvideo';
    case '.MPO':
      return 'image/mpo';
    case '.MP4':
      return 'video/mp4';
    case '.WEBM':
      return 'video/webm';
    case '.JPG':
    case '.JPEG':
      return 'image/jpeg';
    default:
      return 'application/octet-stream';
  }
}

function handleDcimMiddleware(req: IncomingMessage, res: ServerResponse): boolean {
  if (!req.url) return false;
  const parsedUrl = new URL(req.url, 'http://localhost');

  if (parsedUrl.pathname === '/api/dcim/list') {
    res.setHeader('Content-Type', 'application/json');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    try {
      const files = scanDcimDirectory(DCIM_ROOT);
      // Sort alphabetically by relativePath
      files.sort((a, b) => a.relativePath.localeCompare(b.relativePath));
      res.statusCode = 200;
      res.end(JSON.stringify(files));
    } catch (err: unknown) {
      res.statusCode = 500;
      const message = err instanceof Error ? err.message : 'Unknown error';
      res.end(JSON.stringify({ error: message }));
    }
    return true;
  }

  if (parsedUrl.pathname === '/api/dcim/file') {
    const rawPath = parsedUrl.searchParams.get('path');
    if (!rawPath) {
      res.statusCode = 400;
      res.end('Missing path query parameter');
      return true;
    }

    const safeRelativePath = path.normalize(rawPath).replace(/^(\.\.(\/|\\|$))+/, '');
    const absolutePath = path.resolve(DCIM_ROOT, safeRelativePath);

    // Guard against path traversal
    if (!absolutePath.startsWith(DCIM_ROOT) || !fs.existsSync(absolutePath)) {
      res.statusCode = 404;
      res.end('File not found or forbidden');
      return true;
    }

    const stat = fs.statSync(absolutePath);
    if (!stat.isFile()) {
      res.statusCode = 400;
      res.end('Requested target is not a file');
      return true;
    }

    const mime = getMimeType(absolutePath);
    const range = req.headers.range;

    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');

    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
      const chunkSize = end - start + 1;

      res.statusCode = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${stat.size}`);
      res.setHeader('Content-Length', chunkSize);
      res.setHeader('Content-Type', mime);

      const stream = fs.createReadStream(absolutePath, { start, end });
      stream.pipe(res);
    } else {
      res.statusCode = 200;
      res.setHeader('Content-Length', stat.size);
      res.setHeader('Content-Type', mime);

      const stream = fs.createReadStream(absolutePath);
      stream.pipe(res);
    }

    return true;
  }

  return false;
}

function dcimLocalServerPlugin(): Plugin {
  return {
    name: 'dcim-local-server',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!handleDcimMiddleware(req, res)) {
          next();
        }
      });
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!handleDcimMiddleware(req, res)) {
          next();
        }
      });
    },
  };
}

export default defineConfig({
  base: './',
  plugins: [dcimLocalServerPlugin()],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  optimizeDeps: {
    exclude: ['@ffmpeg/ffmpeg', '@ffmpeg/util'],
  },
});
