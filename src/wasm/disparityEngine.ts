import initWasm, { StereoDisparity } from './pkg/stereo_disparity.js';
import wasmUrl from './pkg/stereo_disparity_bg.wasm?url';

export interface DisparityTelemetry {
  computeTimeMs: number;
  eyeWidth: number;
  eyeHeight: number;
  maxDisp: number;
}

export interface DisparityResult {
  disparityMap: Uint8Array;
  telemetry: DisparityTelemetry;
}

export class WasmDisparityEngine {
  private static instance: WasmDisparityEngine | null = null;
  private static initPromise: Promise<WasmDisparityEngine> | null = null;
  private isInitialized = false;

  private halfResEngine: StereoDisparity | null = null;
  private fullResEngine: StereoDisparity | null = null;

  // Offscreen canvas for extracting video frames into RGBA buffers
  private canvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

  public static async getInstance(): Promise<WasmDisparityEngine> {
    if (WasmDisparityEngine.instance) {
      return WasmDisparityEngine.instance;
    }
    if (!WasmDisparityEngine.initPromise) {
      WasmDisparityEngine.initPromise = (async () => {
        const inst = new WasmDisparityEngine();
        await inst.init();
        WasmDisparityEngine.instance = inst;
        return inst;
      })();
    }
    return WasmDisparityEngine.initPromise;
  }

  public async init(): Promise<void> {
    if (this.isInitialized) return;

    await initWasm({ module_or_path: wasmUrl });

    // Default 3DS resolutions:
    // Full resolution: 480x240 per eye (960x240 SBS), max_disp = 24
    // Half resolution: 240x120 per eye (480x120 SBS), max_disp = 12
    this.halfResEngine = new StereoDisparity(240, 120, 12);
    this.fullResEngine = new StereoDisparity(480, 240, 24);

    if (typeof OffscreenCanvas !== 'undefined') {
      this.canvas = new OffscreenCanvas(480, 120);
      this.ctx = this.canvas.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
    } else if (typeof document !== 'undefined') {
      const htmlCanvas = document.createElement('canvas');
      htmlCanvas.width = 480;
      htmlCanvas.height = 120;
      this.canvas = htmlCanvas;
      this.ctx = htmlCanvas.getContext('2d', { willReadFrequently: true });
    }

    this.isInitialized = true;
  }

  /**
   * Computes disparity directly from an HTMLVideoElement or CanvasImageSource.
   */
  public computeFromVideo(
    source: CanvasImageSource,
    halfRes: boolean = true
  ): DisparityResult {
    if (!this.isInitialized || !this.ctx || !this.canvas) {
      throw new Error('WasmDisparityEngine must be initialized before computing disparity');
    }

    const eyeWidth = halfRes ? 240 : 480;
    const eyeHeight = halfRes ? 120 : 240;
    const sbsWidth = eyeWidth * 2;

    if (this.canvas.width !== sbsWidth || this.canvas.height !== eyeHeight) {
      this.canvas.width = sbsWidth;
      this.canvas.height = eyeHeight;
    }

    // Draw video frame to offscreen canvas
    this.ctx.drawImage(source, 0, 0, sbsWidth, eyeHeight);
    const imageData = this.ctx.getImageData(0, 0, sbsWidth, eyeHeight);

    return this.computeFromRgba(
      new Uint8Array(imageData.data.buffer, imageData.data.byteOffset, imageData.data.byteLength),
      halfRes
    );
  }

  /**
   * Computes disparity from a raw RGBA byte buffer.
   */
  public computeFromRgba(
    sbsRgba: Uint8Array,
    halfRes: boolean = true
  ): DisparityResult {
    if (!this.isInitialized) {
      throw new Error('WasmDisparityEngine must be initialized before computing disparity');
    }

    const engine = halfRes ? this.halfResEngine : this.fullResEngine;
    if (!engine) {
      throw new Error(`Disparity engine for ${halfRes ? 'half' : 'full'} resolution is not allocated`);
    }

    const eyeWidth = engine.width();
    const eyeHeight = engine.height();
    const maxDisp = engine.max_disp();

    const startTime = performance.now();
    const disparityMap = engine.compute_disparity(sbsRgba);
    const computeTimeMs = performance.now() - startTime;

    return {
      disparityMap,
      telemetry: {
        computeTimeMs,
        eyeWidth,
        eyeHeight,
        maxDisp,
      },
    };
  }

  public getHalfResEngine(): StereoDisparity | null {
    return this.halfResEngine;
  }

  public getFullResEngine(): StereoDisparity | null {
    return this.fullResEngine;
  }
}
