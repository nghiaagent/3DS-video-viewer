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
  flipY?: boolean;
}

export interface ViewerTelemetry {
  disparityMs: number;
  renderMs: number;
  fps: number;
  backend: DisparityBackend;
  renderMode: RenderMode;
  upscale: UpscaleMultiplier;
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
    eyeTexelSize: WebGLUniformLocation | null;
    maxDisp: WebGLUniformLocation | null;
    flipY: WebGLUniformLocation | null;
  } = { sbsTexture: null, eyeTexelSize: null, maxDisp: null, flipY: null };

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
    renderMode: null,
    flipY: null,
  };

  private sbsTexture: WebGLTexture | null = null;
  private dispTexture: WebGLTexture | null = null;
  private disparityFbo: WebGLFramebuffer | null = null;

  // Configurations
  private backend: DisparityBackend = 'gpu';
  private upscale: UpscaleMultiplier = 2;
  private renderMode: RenderMode = 0;
  private windowCrop: number = 0.06;
  private maxDisp: number = 24;
  private flipY: boolean = false;

  private eyeWidth: number = 480;
  private eyeHeight: number = 240;

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
      eyeTexelSize: gl.getUniformLocation(this.disparityProgram, 'u_eyeTexelSize'),
      maxDisp: gl.getUniformLocation(this.disparityProgram, 'u_maxDisp'),
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

    this.dispTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.dispTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    this.allocateDisparityFbo(this.eyeWidth, this.eyeHeight);
  }

  private allocateDisparityFbo(width: number, height: number): void {
    const gl = this.gl;
    this.eyeWidth = width;
    this.eyeHeight = height;

    gl.bindTexture(gl.TEXTURE_2D, this.dispTexture);
    // Allocate R8 texture for disparity map
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, null);

    if (!this.disparityFbo) {
      this.disparityFbo = gl.createFramebuffer();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.disparityFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTexture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`Disparity FBO incomplete with R8 (${status}), falling back to RGBA8`);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTexture, 0);
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

  public getTelemetry(): ViewerTelemetry {
    return this.latestTelemetry;
  }

  public getLastFrameTimestamp(): number {
    return this.lastFrameTimestamp;
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

    // 1. Upload source to sbsTexture directly on GPU
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    // texImage2D directly takes HTMLVideoElement or ImageBitmap without CPU readback
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

      // Upload disparity map to dispTexture
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.dispTexture);
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

    // 2. Render perspective window to canvas backbuffer
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

    const eyeW = sbsWidth / 2;
    const eyeH = sbsHeight;
    if (this.eyeWidth !== eyeW || this.eyeHeight !== eyeH) {
      this.allocateDisparityFbo(eyeW, eyeH);
      this.updateCanvasResolution();
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
    };

    return this.latestTelemetry;
  }

  private runGpuDisparityPass(): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.disparityFbo);
    gl.viewport(0, 0, this.eyeWidth, this.eyeHeight);

    gl.useProgram(this.disparityProgram);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.dispLocs.sbsTexture, 0);

    gl.uniform2f(this.dispLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
    gl.uniform1i(this.dispLocs.maxDisp, this.maxDisp);
    gl.uniform1f(this.dispLocs.flipY, 0.0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private runPerspectivePass(viewShift: number, customFlipY?: boolean): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    gl.useProgram(this.perspectiveProgram);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.perspLocs.sbsTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dispTexture);
    gl.uniform1i(this.perspLocs.dispTexture, 1);

    gl.uniform1f(this.perspLocs.viewShift, Math.max(0.0, Math.min(1.0, viewShift)));
    gl.uniform1f(this.perspLocs.maxDisparityUV, this.maxDisp / this.eyeWidth);
    gl.uniform1f(this.perspLocs.windowCrop, this.windowCrop);
    gl.uniform2f(this.perspLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
    gl.uniform1f(this.perspLocs.zeroDispNorm, 4.0 / (this.maxDisp + 4.0));
    gl.uniform1i(this.perspLocs.renderMode, this.renderMode);

    const effectiveFlipY = customFlipY !== undefined ? (customFlipY ? 1.0 : 0.0) : (this.flipY ? 1.0 : 0.0);
    gl.uniform1f(this.perspLocs.flipY, effectiveFlipY);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
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
    if (this.dispTexture) gl.deleteTexture(this.dispTexture);
    if (this.disparityFbo) gl.deleteFramebuffer(this.disparityFbo);
    if (this.vbo) gl.deleteBuffer(this.vbo);
    if (this.vao) gl.deleteVertexArray(this.vao);
  }
}
