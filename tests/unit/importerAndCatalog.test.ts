import { describe, it, expect } from 'vitest';
import { formatBytes } from '../../src/importer/dcimCatalog.js';
import fs from 'node:fs';
import path from 'node:path';

describe('DCIM Catalog & Helper Functions', () => {
  it('formats byte quantities into readable strings', () => {
    expect(formatBytes(500)).toBe('500 B');
    expect(formatBytes(1024)).toBe('1.0 KB');
    expect(formatBytes(68648)).toBe('67.0 KB');
    expect(formatBytes(1024 * 1024 * 6.5)).toBe('6.5 MB');
  });


  it('verifies ffmpeg-core files exist in public/ffmpeg for same-origin loading', () => {
    const coreJs = path.resolve(__dirname, '../../public/ffmpeg/ffmpeg-core.js');
    const coreWasm = path.resolve(__dirname, '../../public/ffmpeg/ffmpeg-core.wasm');

    expect(fs.existsSync(coreJs)).toBe(true);
    expect(fs.existsSync(coreWasm)).toBe(true);
    expect(fs.statSync(coreJs).size).toBeGreaterThan(100_000);
    expect(fs.statSync(coreWasm).size).toBeGreaterThan(30_000_000);
  });

  it('correctly resolves subpath URLs for GitHub Pages deployment', () => {
    const ghPagesBase = 'https://nghiaagent.github.io/3DS-video-viewer/';
    const ghPagesIndex = 'https://nghiaagent.github.io/3DS-video-viewer/index.html';

    const coreUrlFromBase = new URL('./ffmpeg/ffmpeg-core.js', ghPagesBase).href;
    const wasmUrlFromBase = new URL('./ffmpeg/ffmpeg-core.wasm', ghPagesBase).href;
    const manifestFromBase = new URL('./samples/manifest.json', ghPagesBase).href;

    const coreUrlFromIndex = new URL('./ffmpeg/ffmpeg-core.js', ghPagesIndex).href;

    expect(coreUrlFromBase).toBe('https://nghiaagent.github.io/3DS-video-viewer/ffmpeg/ffmpeg-core.js');
    expect(wasmUrlFromBase).toBe('https://nghiaagent.github.io/3DS-video-viewer/ffmpeg/ffmpeg-core.wasm');
    expect(manifestFromBase).toBe('https://nghiaagent.github.io/3DS-video-viewer/samples/manifest.json');
    expect(coreUrlFromIndex).toBe('https://nghiaagent.github.io/3DS-video-viewer/ffmpeg/ffmpeg-core.js');
  });

  it('strictly rejects non-AVI video files with clear error message', async () => {
    const { import3DSVideo } = await import('../../src/importer/import3DSVideo.js');
    const fakeVideo = {} as HTMLVideoElement;

    // Test rejection of .mp4
    await expect(
      import3DSVideo({ name: 'HNI_0001.mp4', url: 'https://example.com/HNI_0001.mp4' }, fakeVideo)
    ).rejects.toThrow('Only Nintendo 3DS .AVI recordings are supported.');

    // Test rejection of .webm
    await expect(
      import3DSVideo({ name: 'HNI_0002.webm', url: 'https://example.com/HNI_0002.webm' }, fakeVideo)
    ).rejects.toThrow('Only Nintendo 3DS .AVI recordings are supported.');

    // Test rejection of .mov
    await expect(
      import3DSVideo({ name: 'HNI_0003.mov', url: 'https://example.com/HNI_0003.mov' }, fakeVideo)
    ).rejects.toThrow('Only Nintendo 3DS .AVI recordings are supported.');

    // Test rejection of File object with non-avi extension
    const nonAviFile = { name: 'sample.mp4' } as File;
    await expect(
      import3DSVideo(nonAviFile, fakeVideo)
    ).rejects.toThrow('Only Nintendo 3DS .AVI recordings are supported.');
  });
});
