import { describe, it, expect } from 'vitest';
import { formatBytes } from '../../src/importer/dcimCatalog.js';
import { parse3DSMetadata } from '../../src/importer/parse3DSMetadata.js';
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

describe('parse3DSMetadata (3DS1 MakerNote Parallax)', () => {
  function createSynthetic3DS1(floatBytes: number[], category = 516, prefixLength = 0): Uint8Array {
    const buf = new Uint8Array(prefixLength + 64);
    // '3DS1'
    buf[prefixLength + 0] = 0x33;
    buf[prefixLength + 1] = 0x44;
    buf[prefixLength + 2] = 0x53;
    buf[prefixLength + 3] = 0x31;
    // Category uint16 LE at +16
    const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    view.setUint16(prefixLength + 16, category, true);
    // Float32 LE at +40
    for (let i = 0; i < 4; i++) {
      buf[prefixLength + 40 + i] = floatBytes[i];
    }
    return buf;
  }

  it('parses synthetic 3DS1 buffer with float32 -40.0 (00 00 20 C2)', () => {
    const buf = createSynthetic3DS1([0x00, 0x00, 0x20, 0xc2]);
    const metadata = parse3DSMetadata(buf);
    expect(metadata.found).toBe(true);
    expect(metadata.parallaxOffset).toBe(-40.0);
    expect(metadata.estimatedBaseDisp).toBe(-70);
    expect(metadata.category).toBe(516);
    expect(metadata.activeWidth).toBe(400);
    expect(metadata.activeHeight).toBe(240);
  });

  it('parses synthetic 3DS1 buffer with float32 ≈ -27.676 (1B 69 DD C1)', () => {
    const buf = createSynthetic3DS1([0x1b, 0x69, 0xdd, 0xc1], 516, 128);
    const metadata = parse3DSMetadata(buf);
    expect(metadata.found).toBe(true);
    expect(metadata.parallaxOffset).toBeCloseTo(-27.676, 2);
    expect(metadata.activeWidth).toBe(400);
    expect(metadata.activeHeight).toBe(240);
  });

  it('parses synthetic 3DS1 buffer with float32 ≈ +3.598 (FB 3E 66 40)', () => {
    const buf = createSynthetic3DS1([0xfb, 0x3e, 0x66, 0x40]);
    const metadata = parse3DSMetadata(buf);
    expect(metadata.found).toBe(true);
    expect(metadata.parallaxOffset).toBeCloseTo(3.598, 2);
    expect(metadata.activeWidth).toBe(400);
    expect(metadata.activeHeight).toBe(240);
  });

  it('returns default fallback metadata when 3DS1 magic is absent', () => {
    const emptyBuf = new Uint8Array(256);
    const metadata = parse3DSMetadata(emptyBuf);
    expect(metadata.found).toBe(false);
    expect(metadata.parallaxOffset).toBe(0);
    expect(metadata.estimatedBaseDisp).toBe(-16);
    expect(metadata.category).toBe(0);
    expect(metadata.activeWidth).toBe(400);
    expect(metadata.activeHeight).toBe(240);
  });

  it('extracts correct metadata from real Nintendo 3DS AVI sample if present', () => {
    const samplePath = '/Users/nghiaagent/Downloads/3DS vids/DCIM/101NIN03/HNI_0062.AVI';
    if (fs.existsSync(samplePath)) {
      const fd = fs.openSync(samplePath, 'r');
      const buf = Buffer.alloc(64 * 1024);
      const bytesRead = fs.readSync(fd, buf, 0, 64 * 1024, 0);
      fs.closeSync(fd);

      const metadata = parse3DSMetadata(new Uint8Array(buf.buffer, buf.byteOffset, bytesRead));
      expect(metadata.found).toBe(true);
      expect(metadata.parallaxOffset).toBe(-40.0);
      expect(metadata.estimatedBaseDisp).toBe(-70);
      expect(metadata.category).toBe(516);
      expect(metadata.activeWidth).toBe(400);
      expect(metadata.activeHeight).toBe(240);
    }
  });
});

