export interface ThreeDSMetadata {
  found: boolean;
  parallaxOffset: number;     // Raw IEEE-754 float32 at offset +40 inside "3DS1" MakerNote (e.g. -40.0 on HNI_0062.AVI)
  estimatedBaseDisp: number;  // Estimated integer stereo shift d where R[x - d] == L[x], computed as Math.round(parallaxOffset * 1.8)
  category: number;           // uint16 at offset +16 inside "3DS1" (typically 516 = 0x0204)
  activeWidth: number;        // 400 (Nintendo 3DS 5:3 top screen width)
  activeHeight: number;       // 240 (Nintendo 3DS 5:3 top screen height)
}

/**
 * Scans video container bytes (e.g. Nintendo 3DS .AVI) for the "3DS1" Exif MakerNote magic
 * and extracts camera parallax offset and stereo disparity calibration metadata.
 */
export function parse3DSMetadata(bytes: Uint8Array): ThreeDSMetadata {
  const maxScan = Math.min(bytes.length - 64, 512 * 1024);
  for (let idx = 0; idx <= maxScan; idx++) {
    if (
      bytes[idx] === 0x33 &&
      bytes[idx + 1] === 0x44 &&
      bytes[idx + 2] === 0x53 &&
      bytes[idx + 3] === 0x31
    ) {
      const view = new DataView(bytes.buffer, bytes.byteOffset + idx, 64);
      const category = view.getUint16(16, true);
      const rawFloat = view.getFloat32(40, true);
      if (Number.isFinite(rawFloat) && rawFloat >= -80.0 && rawFloat <= 80.0) {
        const parallaxOffset = Math.max(-60, Math.min(60, rawFloat));
        const estimatedBaseDisp = Math.max(-80, Math.min(32, Math.round(parallaxOffset * 1.75)));
        return {
          found: true,
          parallaxOffset,
          estimatedBaseDisp,
          category,
          activeWidth: 400,
          activeHeight: 240,
        };
      }
    }
  }

  return {
    found: false,
    parallaxOffset: 0,
    estimatedBaseDisp: -16,
    category: 0,
    activeWidth: 400,
    activeHeight: 240,
  };
}
