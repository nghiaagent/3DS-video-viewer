import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const DCIM_ROOT = '/Users/nghiaagent/Downloads/3DS vids/DCIM';
const FFMPEG_BIN = fs.existsSync('/opt/homebrew/bin/ffmpeg')
  ? '/opt/homebrew/bin/ffmpeg'
  : 'ffmpeg';

const fixturesDir = path.join(rootDir, 'tests', 'fixtures');
const samplesDir = path.join(rootDir, 'public', 'samples');

fs.mkdirSync(fixturesDir, { recursive: true });
fs.mkdirSync(samplesDir, { recursive: true });

console.log('--- Preparing 3DS Video Fixtures & Samples ---');
console.log(`Using ffmpeg: ${FFMPEG_BIN}`);
console.log(`DCIM root: ${DCIM_ROOT}`);

// 1. Extract raw 960x240 RGBA SBS frames
const frameExtractions = [
  {
    aviSource: path.join(DCIM_ROOT, '100NIN03', 'HNI_0060.AVI'),
    outputFile: path.join(fixturesDir, 'hni_0060_frame10.rgba'),
    frameNumber: 10,
    label: 'HNI_0060 frame 10',
  },
  {
    aviSource: path.join(DCIM_ROOT, '100NIN03', 'HNI_0070.AVI'),
    outputFile: path.join(fixturesDir, 'hni_0070_frame05.rgba'),
    frameNumber: 5,
    label: 'HNI_0070 frame 5',
  },
];

for (const item of frameExtractions) {
  if (!fs.existsSync(item.aviSource)) {
    throw new Error(`Source AVI not found: ${item.aviSource}`);
  }
  console.log(`\nExtracting ${item.label}...`);
  // [0:v:0] and [0:v:1] are the dual MJPEG left/right eye streams
  const filter = `[0:v:0]select='eq(n\\,${item.frameNumber})'[v0];[0:v:1]select='eq(n\\,${item.frameNumber})'[v1];[v0][v1]hstack=inputs=2`;
  execFileSync(
    FFMPEG_BIN,
    [
      '-y',
      '-i',
      item.aviSource,
      '-filter_complex',
      filter,
      '-frames:v',
      '1',
      '-f',
      'rawvideo',
      '-pix_fmt',
      'rgba',
      item.outputFile,
    ],
    { stdio: 'inherit' }
  );

  const stat = fs.statSync(item.outputFile);
  const expectedSize = 960 * 240 * 4; // 921,600 bytes
  console.log(`Saved ${path.basename(item.outputFile)}: ${stat.size} bytes (Expected: ${expectedSize})`);
  if (stat.size !== expectedSize) {
    throw new Error(`Unexpected file size for ${item.outputFile}: got ${stat.size}, expected ${expectedSize}`);
  }
}

// 2. Copy compact real .AVI files to public/samples/
const aviCopies = [
  {
    source: path.join(DCIM_ROOT, '101NIN03', 'HNI_0065.AVI'),
    destination: path.join(samplesDir, 'HNI_0065.AVI'),
  },
  {
    source: path.join(DCIM_ROOT, '100NIN03', 'HNI_0070.AVI'),
    destination: path.join(samplesDir, 'HNI_0070.AVI'),
  },
];

for (const copy of aviCopies) {
  if (!fs.existsSync(copy.source)) {
    throw new Error(`Source AVI not found: ${copy.source}`);
  }
  fs.copyFileSync(copy.source, copy.destination);
  const stat = fs.statSync(copy.destination);
  console.log(`Copied ${path.basename(copy.destination)}: ${stat.size} bytes`);
}

// 3. Pre-convert HNI_0070.AVI to browser-ready Side-by-Side MP4 (H.264)
const sbsMp4Output = path.join(samplesDir, 'HNI_0070_sbs.mp4');
console.log(`\nTranscoding HNI_0070.AVI to browser-compatible SBS MP4...`);
execFileSync(
  FFMPEG_BIN,
  [
    '-y',
    '-i',
    path.join(DCIM_ROOT, '100NIN03', 'HNI_0070.AVI'),
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
    'fast',
    '-crf',
    '18',
    '-c:a',
    'aac',
    sbsMp4Output,
  ],
  { stdio: 'inherit' }
);
const mp4Stat = fs.statSync(sbsMp4Output);
console.log(`Generated ${path.basename(sbsMp4Output)}: ${mp4Stat.size} bytes`);

// 4. Create public/samples/manifest.json
const manifest = {
  samples: [
    {
      id: 'hni-0070-sbs',
      name: 'HNI_0070 (Pre-converted SBS MP4)',
      type: 'video/mp4',
      url: 'samples/HNI_0070_sbs.mp4',
      format: 'sbs_mp4',
      sizeBytes: mp4Stat.size,
      width: 960,
      height: 240,
      fps: 20,
      description: 'Ready-to-play Side-by-Side 3DS recording (100NIN03/HNI_0070.AVI)',
    },
    {
      id: 'hni-0070-avi',
      name: 'HNI_0070.AVI (Raw Dual MJPEG 3DS Clip)',
      type: 'video/x-msvideo',
      url: 'samples/HNI_0070.AVI',
      format: 'raw_avi',
      sizeBytes: fs.statSync(path.join(samplesDir, 'HNI_0070.AVI')).size,
      widthPerEye: 480,
      height: 240,
      fps: 20,
      description: 'Raw Nintendo 3DS dual-stream AVI recording (~372 KB)',
    },
    {
      id: 'hni-0065-avi',
      name: 'HNI_0065.AVI (Raw Dual MJPEG 3DS Clip)',
      type: 'video/x-msvideo',
      url: 'samples/HNI_0065.AVI',
      format: 'raw_avi',
      sizeBytes: fs.statSync(path.join(samplesDir, 'HNI_0065.AVI')).size,
      widthPerEye: 480,
      height: 240,
      fps: 20,
      description: 'Compact Nintendo 3DS dual-stream AVI recording (~68 KB)',
    },
  ],
};

const manifestPath = path.join(samplesDir, 'manifest.json');
fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2), 'utf-8');
console.log(`Generated manifest at ${manifestPath}`);
console.log('\n--- Fixture Preparation Complete ---');
