import {
  VERTEX_SHADER_SOURCE,
  DISPARITY_SAD_FRAG_SOURCE,
  PERSPECTIVE_WINDOW_FRAG_SOURCE,
} from './shaders.js';
import { WasmDisparityEngine } from '../wasm/disparityEngine.js';

export type DisparityBackend = 'gpu' | 'wasm';
export type UpscaleMultiplier = 1 | 2 | 3;
export type RenderMode =
  | 0 // Parallax 2D Window
  | 1 // Disparity Heatmap
  | 2 // Left Eye Only
  | 3 // Right Eye Only
  | 4 // Side-by-Side (SBS)
  | 5; // Anaglyph 3D

export interface ViewerOptions {
  upscale?: UpscaleMultiplier;
  backend?: DisparityBackend;
  renderMode?: RenderMode;
  windowCrop?: number;
  maxDisp?: number;
  baseDisp?: number;
  flipY?: boolean;
}

export interface ViewerTelemetry {
  disparityMs: number;
  renderMs: number;
  fps: number;
  backend: DisparityBackend;
  renderMode: RenderMode;
  upscale: UpscaleMultiplier;
  baseDisp: number;
}

export class WebGLParallaxViewer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;

  private vao: WebGLVertexArrayObject | null = null;
  private vbo: WebGLBuffer | null = null;

  // Pass 1: GPU SAD Disparity
  private disparityProgram: WebGLProgram | null = null;
  private dispLocs: {
    sbsTexture: WebGLUniformLocation | null;
    prevDispTexture: WebGLUniformLocation | null;
    eyeTexelSize: WebGLUniformLocation | null;
    maxDisp: WebGLUniformLocation | null;
    baseDisp: WebGLUniformLocation | null;
    temporalWeight: WebGLUniformLocation | null;
    flipY: WebGLUniformLocation | null;
  } = {
    sbsTexture: null,
    prevDispTexture: null,
    eyeTexelSize: null,
    maxDisp: null,
    baseDisp: null,
    temporalWeight: null,
    flipY: null,
  };

  // Pass 2: Perspective Window & View Modes
  private perspectiveProgram: WebGLProgram | null = null;
  private perspLocs: {
    sbsTexture: WebGLUniformLocation | null;
    dispTexture: WebGLUniformLocation | null;
    viewShift: WebGLUniformLocation | null;
    maxDisparityUV: WebGLUniformLocation | null;
    windowCrop: WebGLUniformLocation | null;
    eyeTexelSize: WebGLUniformLocation | null;
    zeroDispNorm: WebGLUniformLocation | null;
    baseDisp: WebGLUniformLocation | null;
    renderMode: WebGLUniformLocation | null;
    flipY: WebGLUniformLocation | null;
  } = {
    sbsTexture: null,
    dispTexture: null,
    viewShift: null,
    maxDisparityUV: null,
    windowCrop: null,
    eyeTexelSize: null,
    zeroDispNorm: null,
    baseDisp: null,
    renderMode: null,
    flipY: null,
  };

  private sbsTexture: WebGLTexture | null = null;
  // Ping-pong disparity textures and FBOs for temporal stabilization
  private dispTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
  private disparityFbos: [WebGLFramebuffer | null, WebGLFramebuffer | null] = [null, null];
  private pingPongIdx: number = 0;
  private frameCountTotal: number = 0;

  // Configurations
  private backend: DisparityBackend = 'gpu';
  private upscale: UpscaleMultiplier = 2;
  private renderMode: RenderMode = 0;
  private windowCrop: number = 0.06;
  private maxDisp: number = 24;
  private flipY: boolean = true; // Default to true so HTMLVideoElement is upright on screen

  // Auto-Calibrated Stereo Convergence Baseline
  private baseDisp: number = 0;
  private smoothedBaseDisp: number = 0;
  private manualBaseDisp: number | null = null;

  private eyeWidth: number = 480;
  private eyeHeight: number = 240;

  // Offscreen canvas for fast baseline estimation from video elements
  private baselineCanvas: HTMLCanvasElement | OffscreenCanvas | null = null;
  private baselineCtx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null = null;

  // Telemetry & FPS tracking
  private lastFrameTimestamp: number = performance.now();
  private frameCount: number = 0;
  private currentFps: number = 60;
  private lastFpsUpdate: number = performance.now();
  private latestTelemetry: ViewerTelemetry = {
    disparityMs: 0,
    renderMs: 0,
    fps: 60,
    backend: 'gpu',
    renderMode: 0,
    upscale: 2,
    baseDisp: 0,
  };

  constructor(canvas: HTMLCanvasElement, options?: ViewerOptions) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: true,
      powerPreference: 'high-performance',
    });

    if (!gl) {
      throw new Error('WebGL2 is not supported by this browser/device');
    }
    this.gl = gl;

    if (options) {
      if (options.upscale) this.upscale = options.upscale;
      if (options.backend) this.backend = options.backend;
      if (options.renderMode !== undefined) this.renderMode = options.renderMode;
      if (options.windowCrop !== undefined) this.windowCrop = options.windowCrop;
      if (options.maxDisp !== undefined) this.maxDisp = options.maxDisp;
      if (options.baseDisp !== undefined) {
        this.baseDisp = options.baseDisp;
        this.smoothedBaseDisp = options.baseDisp;
        this.manualBaseDisp = options.baseDisp;
      }
      if (options.flipY !== undefined) this.flipY = options.flipY;
    }

    this.initPipeline();
    this.updateCanvasResolution();
  }

  private initPipeline(): void {
    const gl = this.gl;

    // Fullscreen single-triangle geometry
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    // Triangle covering [-1, -1] to [3, -1] and [-1, 3]
    const trianglePositions = new Float32Array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]);
    gl.bufferData(gl.ARRAY_BUFFER, trianglePositions, gl.STATIC_DRAW);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    // Compile Pass 1 (SAD Disparity)
    this.disparityProgram = this.createProgram(VERTEX_SHADER_SOURCE, DISPARITY_SAD_FRAG_SOURCE);
    this.dispLocs = {
      sbsTexture: gl.getUniformLocation(this.disparityProgram, 'u_sbsTexture'),
      prevDispTexture: gl.getUniformLocation(this.disparityProgram, 'u_prevDispTexture'),
      eyeTexelSize: gl.getUniformLocation(this.disparityProgram, 'u_eyeTexelSize'),
      maxDisp: gl.getUniformLocation(this.disparityProgram, 'u_maxDisp'),
      baseDisp: gl.getUniformLocation(this.disparityProgram, 'u_baseDisp'),
      temporalWeight: gl.getUniformLocation(this.disparityProgram, 'u_temporalWeight'),
      flipY: gl.getUniformLocation(this.disparityProgram, 'u_flipY'),
    };

    // Compile Pass 2 (Perspective Window)
    this.perspectiveProgram = this.createProgram(VERTEX_SHADER_SOURCE, PERSPECTIVE_WINDOW_FRAG_SOURCE);
    this.perspLocs = {
      sbsTexture: gl.getUniformLocation(this.perspectiveProgram, 'u_sbsTexture'),
      dispTexture: gl.getUniformLocation(this.perspectiveProgram, 'u_dispTexture'),
      viewShift: gl.getUniformLocation(this.perspectiveProgram, 'u_viewShift'),
      maxDisparityUV: gl.getUniformLocation(this.perspectiveProgram, 'u_maxDisparityUV'),
      windowCrop: gl.getUniformLocation(this.perspectiveProgram, 'u_windowCrop'),
      eyeTexelSize: gl.getUniformLocation(this.perspectiveProgram, 'u_eyeTexelSize'),
      zeroDispNorm: gl.getUniformLocation(this.perspectiveProgram, 'u_zeroDispNorm'),
      baseDisp: gl.getUniformLocation(this.perspectiveProgram, 'u_baseDisp'),
      renderMode: gl.getUniformLocation(this.perspectiveProgram, 'u_renderMode'),
      flipY: gl.getUniformLocation(this.perspectiveProgram, 'u_flipY'),
    };

    // Textures setup
    this.sbsTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.allocateDisparityFbos(this.eyeWidth, this.eyeHeight);
  }

  private allocateDisparityFbos(width: number, height: number): void {
    const gl = this.gl;
    this.eyeWidth = width;
    this.eyeHeight = height;

    for (let i = 0; i < 2; i++) {
      if (!this.dispTextures[i]) {
        this.dispTextures[i] = gl.createTexture();
      }
      gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[i]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);

      if (!this.disparityFbos[i]) {
        this.disparityFbos[i] = gl.createFramebuffer();
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.disparityFbos[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTextures[i], 0);

      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        console.warn(`Disparity FBO ${i} incomplete with R8 (${status}), falling back to RGBA8`);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTextures[i], 0);
      }
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private createProgram(vsSource: string, fsSource: string): WebGLProgram {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER)!;
    gl.shaderSource(vs, vsSource);
    gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(vs);
      gl.deleteShader(vs);
      throw new Error(`Vertex shader compilation failed: ${err}`);
    }

    const fs = gl.createShader(gl.FRAGMENT_SHADER)!;
    gl.shaderSource(fs, fsSource);
    gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS)) {
      const err = gl.getShaderInfoLog(fs);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      throw new Error(`Fragment shader compilation failed: ${err}`);
    }

    const prog = gl.createProgram()!;
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const err = gl.getProgramInfoLog(prog);
      gl.deleteProgram(prog);
      throw new Error(`Program link failed: ${err}`);
    }

    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return prog;
  }

  public updateCanvasResolution(): void {
    const baseW = this.eyeWidth;
    const baseH = this.eyeHeight;
    const targetW = baseW * this.upscale;
    const targetH = baseH * this.upscale;

    if (this.canvas.width !== targetW || this.canvas.height !== targetH) {
      this.canvas.width = targetW;
      this.canvas.height = targetH;
    }
  }

  public setUpscaleMultiplier(multiplier: UpscaleMultiplier): void {
    this.upscale = multiplier;
    this.updateCanvasResolution();
  }

  public setBackend(backend: DisparityBackend): void {
    this.backend = backend;
  }

  public setRenderMode(mode: RenderMode): void {
    this.renderMode = mode;
  }

  public setWindowCrop(crop: number): void {
    this.windowCrop = Math.max(0.0, Math.min(0.2, crop));
  }

  public setMaxDisparity(maxDisp: number): void {
    this.maxDisp = maxDisp;
  }

  public setFlipY(flip: boolean): void {
    this.flipY = flip;
  }

  public getBaseDisparity(): number {
    return this.baseDisp;
  }

  public setBaseDisparity(base: number | null): void {
    this.manualBaseDisp = base;
    if (base !== null) {
      this.baseDisp = base;
      this.smoothedBaseDisp = base;
    }
  }

  public getTelemetry(): ViewerTelemetry {
    return this.latestTelemetry;
  }

  public getLastFrameTimestamp(): number {
    return this.lastFrameTimestamp;
  }

  /**
   * Fast auto-calibrated horizontal stereo baseline shift estimation.
   * Searches coarse range [-112, 32] (step 4), followed by fine range [-3, +3] (step 1).
   * Runs in < 0.2 ms on downsampled or sub-sampled frames.
   */
  public estimateGlobalBaselineFromRgba(
    rgba: Uint8Array,
    sbsWidth: number,
    sbsHeight: number
  ): number {
    const eyeW = Math.floor(sbsWidth / 2);
    const eyeH = sbsHeight;

    const yStart = Math.max(2, Math.floor(eyeH * 0.2));
    const yEnd = Math.min(eyeH - 2, Math.floor(eyeH * 0.8));
    const xStart = Math.max(4, Math.floor(eyeW * 0.2));
    const xEnd = Math.min(eyeW - 4, Math.floor(eyeW * 0.8));

    let bestCoarseD = 0;
    let minCoarseMad = 1e9;

    // Coarse pass: step 4
    for (let d = -112; d <= 32; d += 4) {
      let diffSum = 0;
      let count = 0;

      for (let y = yStart; y < yEnd; y += 4) {
        const rowOffset = y * sbsWidth * 4;
        for (let x = xStart; x < xEnd; x += 4) {
          const rx = x - d;
          if (rx >= 0 && rx < eyeW) {
            const lIdx = rowOffset + x * 4;
            const rIdx = rowOffset + (eyeW + rx) * 4;

            const lumaL = (77 * rgba[lIdx] + 150 * rgba[lIdx + 1] + 29 * rgba[lIdx + 2]) >> 8;
            const lumaR = (77 * rgba[rIdx] + 150 * rgba[rIdx + 1] + 29 * rgba[rIdx + 2]) >> 8;

            diffSum += Math.abs(lumaL - lumaR);
            count++;
          }
        }
      }

      if (count > 0) {
        const mad = diffSum / count;
        if (mad < minCoarseMad) {
          minCoarseMad = mad;
          bestCoarseD = d;
        }
      }
    }

    // Fine pass: step 1 around bestCoarseD
    let bestD = bestCoarseD;
    let minFineMad = 1e9;
    const fineMin = Math.max(-112, bestCoarseD - 3);
    const fineMax = Math.min(32, bestCoarseD + 3);

    for (let d = fineMin; d <= fineMax; d++) {
      let diffSum = 0;
      let count = 0;

      for (let y = yStart; y < yEnd; y += 2) {
        const rowOffset = y * sbsWidth * 4;
        for (let x = xStart; x < xEnd; x += 2) {
          const rx = x - d;
          if (rx >= 0 && rx < eyeW) {
            const lIdx = rowOffset + x * 4;
            const rIdx = rowOffset + (eyeW + rx) * 4;

            const lumaL = (77 * rgba[lIdx] + 150 * rgba[lIdx + 1] + 29 * rgba[lIdx + 2]) >> 8;
            const lumaR = (77 * rgba[rIdx] + 150 * rgba[rIdx + 1] + 29 * rgba[rIdx + 2]) >> 8;

            diffSum += Math.abs(lumaL - lumaR);
            count++;
          }
        }
      }

      if (count > 0) {
        const mad = diffSum / count;
        if (mad < minFineMad) {
          minFineMad = mad;
          bestD = d;
        }
      }
    }

    return bestD;
  }

  private updateBaseDisparityFromRgba(rgba: Uint8Array, sbsWidth: number, sbsHeight: number): void {
    if (this.manualBaseDisp !== null) {
      this.baseDisp = this.manualBaseDisp;
      this.smoothedBaseDisp = this.manualBaseDisp;
      return;
    }

    const detected = this.estimateGlobalBaselineFromRgba(rgba, sbsWidth, sbsHeight);
    if (this.frameCountTotal <= 1) {
      this.smoothedBaseDisp = detected;
    } else {
      // Exponential moving average for temporal stability
      this.smoothedBaseDisp = 0.75 * this.smoothedBaseDisp + 0.25 * detected;
    }
    this.baseDisp = Math.round(this.smoothedBaseDisp);
  }

  private updateBaseDisparityFromVideo(video: TexImageSource): void {
    if (this.manualBaseDisp !== null) {
      this.baseDisp = this.manualBaseDisp;
      this.smoothedBaseDisp = this.manualBaseDisp;
      return;
    }

    try {
      if (!this.baselineCanvas) {
        if (typeof OffscreenCanvas !== 'undefined') {
          this.baselineCanvas = new OffscreenCanvas(240, 60);
          this.baselineCtx = this.baselineCanvas.getContext('2d', { willReadFrequently: true }) as OffscreenCanvasRenderingContext2D;
        } else if (typeof document !== 'undefined') {
          const c = document.createElement('canvas');
          c.width = 240;
          c.height = 60;
          this.baselineCanvas = c;
          this.baselineCtx = c.getContext('2d', { willReadFrequently: true });
        }
      }

      if (this.baselineCtx && this.baselineCanvas) {
        const ctx = this.baselineCtx;
        ctx.drawImage(video as CanvasImageSource, 0, 0, 240, 60);
        const img = ctx.getImageData(0, 0, 240, 60);
        const raw = new Uint8Array(img.data.buffer, img.data.byteOffset, img.data.byteLength);

        // At 240x60, eyeW = 120, which is downsampled by factor 4 relative to 960x240 (480 eyeW)
        const detectedHalf = this.estimateGlobalBaselineFromRgba(raw, 240, 60);
        const scale = this.eyeWidth / 120;
        const detected = Math.round(detectedHalf * scale);

        if (this.frameCountTotal <= 1) {
          this.smoothedBaseDisp = detected;
        } else {
          this.smoothedBaseDisp = 0.75 * this.smoothedBaseDisp + 0.25 * detected;
        }
        this.baseDisp = Math.round(this.smoothedBaseDisp);
      }
    } catch {
      // Gracefully ignore if 2D context extraction fails in headless tests
    }
  }

  /**
   * Primary render method accepting an HTMLVideoElement or TexImageSource.
   */
  public async renderVideoFrame(
    video: TexImageSource,
    viewShift: number = 0.5
  ): Promise<ViewerTelemetry> {
    const gl = this.gl;
    const startTotal = performance.now();
    this.frameCountTotal++;

    // 1. Auto-converge global stereo baseline every 4 frames or on initial frame
    if (this.frameCountTotal % 4 === 1 || this.frameCountTotal <= 2) {
      this.updateBaseDisparityFromVideo(video);
    }

    // 2. Upload source to sbsTexture directly on GPU
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);

    let dispMs = 0;

    if (this.backend === 'gpu') {
      const startDisp = performance.now();
      this.runGpuDisparityPass();
      dispMs = performance.now() - startDisp;
    } else {
      const wasmEngine = await WasmDisparityEngine.getInstance();
      const res = wasmEngine.computeFromVideo(video as CanvasImageSource, true);
      dispMs = res.telemetry.computeTimeMs;

      // Upload disparity map to current ping-pong disparity texture
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[this.pingPongIdx]);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R8,
        res.telemetry.eyeWidth,
        res.telemetry.eyeHeight,
        0,
        gl.RED,
        gl.UNSIGNED_BYTE,
        res.disparityMap
      );
    }

    // 3. Render perspective window to canvas backbuffer
    const startRender = performance.now();
    this.runPerspectivePass(viewShift);
    const renderMs = performance.now() - startRender;

    this.updateFpsTracking(startTotal);

    this.latestTelemetry = {
      disparityMs: dispMs,
      renderMs,
      fps: this.currentFps,
      backend: this.backend,
      renderMode: this.renderMode,
      upscale: this.upscale,
      baseDisp: this.baseDisp,
    };

    return this.latestTelemetry;
  }

  /**
   * Renders a raw RGBA byte buffer (e.g. for headless tests, frame extraction, or offline export).
   */
  public renderRawRgbaFrame(
    sbsRgba: Uint8Array,
    sbsWidth: number,
    sbsHeight: number,
    viewShift: number = 0.5,
    customFlipY?: boolean
  ): ViewerTelemetry {
    const gl = this.gl;
    const startTotal = performance.now();
    this.frameCountTotal++;

    const eyeW = Math.floor(sbsWidth / 2);
    const eyeH = sbsHeight;
    if (this.eyeWidth !== eyeW || this.eyeHeight !== eyeH) {
      this.allocateDisparityFbos(eyeW, eyeH);
      this.updateCanvasResolution();
    }

    // Auto-converge global baseline
    if (this.frameCountTotal % 4 === 1 || this.frameCountTotal <= 2) {
      this.updateBaseDisparityFromRgba(sbsRgba, sbsWidth, sbsHeight);
    }

    // Upload SBS raw RGBA to GPU
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, sbsWidth, sbsHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, sbsRgba);

    let dispMs = 0;
    if (this.backend === 'gpu') {
      const startDisp = performance.now();
      this.runGpuDisparityPass();
      dispMs = performance.now() - startDisp;
    }

    const startRender = performance.now();
    this.runPerspectivePass(viewShift, customFlipY);
    const renderMs = performance.now() - startRender;

    this.updateFpsTracking(startTotal);

    this.latestTelemetry = {
      disparityMs: dispMs,
      renderMs,
      fps: this.currentFps,
      backend: this.backend,
      renderMode: this.renderMode,
      upscale: this.upscale,
      baseDisp: this.baseDisp,
    };

    return this.latestTelemetry;
  }

  private runGpuDisparityPass(): void {
    const gl = this.gl;
    const writeIdx = this.pingPongIdx;
    const readIdx = 1 - this.pingPongIdx;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.disparityFbos[writeIdx]);
    gl.viewport(0, 0, this.eyeWidth, this.eyeHeight);

    gl.useProgram(this.disparityProgram);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.dispLocs.sbsTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[readIdx]);
    gl.uniform1i(this.dispLocs.prevDispTexture, 1);

    gl.uniform2f(this.dispLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
    gl.uniform1i(this.dispLocs.maxDisp, this.maxDisp);
    gl.uniform1i(this.dispLocs.baseDisp, this.baseDisp);

    const temporalWeight = this.frameCountTotal > 1 ? 0.75 : 0.0;
    gl.uniform1f(this.dispLocs.temporalWeight, temporalWeight);
    gl.uniform1f(this.dispLocs.flipY, 0.0); // Pass 1 FBO disparity coordinates match sbsTexture 1:1

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private runPerspectivePass(viewShift: number, customFlipY?: boolean): void {
    const gl = this.gl;
    const currDispIdx = this.pingPongIdx;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    gl.useProgram(this.perspectiveProgram);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.perspLocs.sbsTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[currDispIdx]);
    gl.uniform1i(this.perspLocs.dispTexture, 1);

    gl.uniform1f(this.perspLocs.viewShift, Math.max(0.0, Math.min(1.0, viewShift)));
    gl.uniform1f(this.perspLocs.maxDisparityUV, this.maxDisp / this.eyeWidth);
    gl.uniform1f(this.perspLocs.windowCrop, this.windowCrop);
    gl.uniform2f(this.perspLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
    gl.uniform1f(this.perspLocs.zeroDispNorm, 0.5);
    gl.uniform1i(this.perspLocs.baseDisp, this.baseDisp);
    gl.uniform1i(this.perspLocs.renderMode, this.renderMode);

    const effectiveFlipY = customFlipY !== undefined ? (customFlipY ? 1.0 : 0.0) : (this.flipY ? 1.0 : 0.0);
    gl.uniform1f(this.perspLocs.flipY, effectiveFlipY);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Flip ping-pong texture index for the next frame
    this.pingPongIdx = 1 - this.pingPongIdx;
  }

  private updateFpsTracking(frameStart: number): void {
    this.frameCount++;
    const now = performance.now();
    if (now - this.lastFpsUpdate >= 500) {
      this.currentFps = Math.round((this.frameCount * 1000) / (now - this.lastFpsUpdate));
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }
    this.lastFrameTimestamp = frameStart;
  }

  public getCanvas(): HTMLCanvasElement {
    return this.canvas;
  }

  public getGLContext(): WebGL2RenderingContext {
    return this.gl;
  }

  public destroy(): void {
    const gl = this.gl;
    if (this.disparityProgram) gl.deleteProgram(this.disparityProgram);
    if (this.perspectiveProgram) gl.deleteProgram(this.perspectiveProgram);
    if (this.sbsTexture) gl.deleteTexture(this.sbsTexture);
    for (let i = 0; i < 2; i++) {
      if (this.dispTextures[i]) gl.deleteTexture(this.dispTextures[i]);
      if (this.disparityFbos[i]) gl.deleteFramebuffer(this.disparityFbos[i]);
    }
    if (this.vbo) gl.deleteBuffer(this.vbo);
    if (this.vao) gl.deleteVertexArray(this.vao);
  }
}
