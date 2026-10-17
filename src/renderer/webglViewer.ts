import {
  VERTEX_SHADER_SOURCE,
  DISPARITY_SAD_FRAG_SOURCE,
  PERSPECTIVE_WINDOW_FRAG_SOURCE,
  TIEFLING_DEPTH_FILTER_FRAG_SOURCE,
  TIEFLING_MESH_VERTEX_SHADER_SOURCE,
} from './shaders.js';
import { WasmDisparityEngine } from '../wasm/disparityEngine.js';

export type DisparityBackend = 'gpu' | 'wasm';
export type UpscaleMultiplier = 1 | 2 | 3;
export type RenderMode =
  | 0 // Parallax 3D Mesh
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
  verticalTilt?: number;
  focus?: number;
  sensitivity?: number;
  meshDepth?: number;
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

export type DepthMapProvider = (timeSec: number) => { depth: Uint8Array; width: number; height: number } | null;

export class WebGLParallaxViewer {
  private canvas: HTMLCanvasElement;
  private gl: WebGL2RenderingContext;

  private vao: WebGLVertexArrayObject | null = null;
  private vbo: WebGLBuffer | null = null;

  // Subdivided 3D mesh for Tiefling Parallax Mesh mode (gridW = 200, gridH = 120)
  private meshVao: WebGLVertexArrayObject | null = null;
  private meshVbo: WebGLBuffer | null = null;
  private meshIbo: WebGLBuffer | null = null;
  private meshIndexCount: number = 0;

  // Pass 1: GPU SAD Disparity
  private disparityProgram: WebGLProgram | null = null;
  private dispLocs: {
    sbsTexture: WebGLUniformLocation | null;
    prevDispTexture: WebGLUniformLocation | null;
    daDepthTexture: WebGLUniformLocation | null;
    hasDaDepth: WebGLUniformLocation | null;
    eyeTexelSize: WebGLUniformLocation | null;
    maxDisp: WebGLUniformLocation | null;
    baseDisp: WebGLUniformLocation | null;
    temporalWeight: WebGLUniformLocation | null;
    flipY: WebGLUniformLocation | null;
  } = {
    sbsTexture: null,
    prevDispTexture: null,
    daDepthTexture: null,
    hasDaDepth: null,
    eyeTexelSize: null,
    maxDisp: null,
    baseDisp: null,
    temporalWeight: null,
    flipY: null,
  };

  // Pass 2: Tiefling Depth Filter Pass
  private depthFilterProgram: WebGLProgram | null = null;
  private depthFilterLocs: {
    rawDispTexture: WebGLUniformLocation | null;
    sbsTexture: WebGLUniformLocation | null;
    daDepthTexture: WebGLUniformLocation | null;
    hasDaDepth: WebGLUniformLocation | null;
    dispTexelSize: WebGLUniformLocation | null;
    windowCrop: WebGLUniformLocation | null;
  } = {
    rawDispTexture: null,
    sbsTexture: null,
    daDepthTexture: null,
    hasDaDepth: null,
    dispTexelSize: null,
    windowCrop: null,
  };

  // Pass 3: Perspective Window & View Modes (Fullscreen quad for modes 1..5)
  private perspectiveProgram: WebGLProgram | null = null;
  private perspLocs: {
    sbsTexture: WebGLUniformLocation | null;
    dispTexture: WebGLUniformLocation | null;
    viewShift: WebGLUniformLocation | null;
    maxDisparityUV: WebGLUniformLocation | null;
    windowCrop: WebGLUniformLocation | null;
    eyeTexelSize: WebGLUniformLocation | null;
    dispTexelSize: WebGLUniformLocation | null;
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
    dispTexelSize: null,
    zeroDispNorm: null,
    baseDisp: null,
    renderMode: null,
    flipY: null,
  };

  // Tiefling 3D Mesh Program (Mode 0)
  private tieflingMeshProgram: WebGLProgram | null = null;
  private tieflingLocs: {
    sbsTexture: WebGLUniformLocation | null;
    dispTexture: WebGLUniformLocation | null;
    viewShift: WebGLUniformLocation | null;
    verticalTilt: WebGLUniformLocation | null;
    meshDepth: WebGLUniformLocation | null;
    focus: WebGLUniformLocation | null;
    sensitivity: WebGLUniformLocation | null;
    windowCrop: WebGLUniformLocation | null;
    flipY: WebGLUniformLocation | null;
    renderMode: WebGLUniformLocation | null;
    eyeTexelSize: WebGLUniformLocation | null;
    dispTexelSize: WebGLUniformLocation | null;
    maxDisparityUV: WebGLUniformLocation | null;
    zeroDispNorm: WebGLUniformLocation | null;
    baseDisp: WebGLUniformLocation | null;
  } = {
    sbsTexture: null,
    dispTexture: null,
    viewShift: null,
    verticalTilt: null,
    meshDepth: null,
    focus: null,
    sensitivity: null,
    windowCrop: null,
    flipY: null,
    renderMode: null,
    eyeTexelSize: null,
    dispTexelSize: null,
    maxDisparityUV: null,
    zeroDispNorm: null,
    baseDisp: null,
  };

  private sbsTexture: WebGLTexture | null = null;
  // Ping-pong coarse disparity textures and FBOs for silky-smooth 2.5D relief
  private dispTextures: [WebGLTexture | null, WebGLTexture | null] = [null, null];
  private disparityFbos: [WebGLFramebuffer | null, WebGLFramebuffer | null] = [null, null];
  private pingPongIdx: number = 0;
  private frameCountTotal: number = 0;

  // Filtered Tiefling depth FBOs (Two-pass joint bilateral filter)
  private filteredDispTexture: WebGLTexture | null = null;
  private filteredDispFbo: WebGLFramebuffer | null = null;
  private tempFilterTexture: WebGLTexture | null = null;
  private tempFilterFbo: WebGLFramebuffer | null = null;

  // External DepthAnything depth map texture and state
  private daDepthTexture: WebGLTexture | null = null;
  private hasDaDepth: boolean = false;
  private daDepthWidth: number = 480;
  private daDepthHeight: number = 240;
  private daDepthFormat: number = 0;
  private lastUploadedDepthBuffer: Uint8Array | null = null;
  private depthProvider: ((timeSec: number) => { depth: Uint8Array; width: number; height: number } | null) | null = null;

  // Configurations
  private backend: DisparityBackend = 'gpu';
  private upscale: UpscaleMultiplier = 2;
  private renderMode: RenderMode = 0;
  private windowCrop: number = 0.0833; // 40px crop per side on 480x240 for 400x240 (5:3) 3DS screen
  private maxDisp: number = 20;
  private flipY: boolean = true; // Default to true so HTMLVideoElement is upright on screen

  // Tiefling 3D Mesh Perspective parameters
  private verticalTilt: number = 0.0;
  private meshDepth: number = 0.6;
  private focus: number = 0.35;
  private sensitivity: number = 0.28;

  // Auto-Calibrated Stereo Convergence Baseline (range [-80, 24])
  private baseDisp: number = 8;
  private smoothedBaseDisp: number = 8;
  private manualBaseDisp: number | null = null;
  private metadataBaseDisp: number | null = null;
  private parallaxOffset: number | null = null;

  private eyeWidth: number = 480;
  private eyeHeight: number = 240;
  private dispWidth: number = 72;
  private dispHeight: number = 44;

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
    baseDisp: 8,
  };

  constructor(canvas: HTMLCanvasElement, options?: ViewerOptions) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: true,
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
      if (options.maxDisp !== undefined) {
        this.maxDisp = options.maxDisp;
        this.meshDepth = 0.15 + (options.maxDisp / 36.0) * 0.65;
      }
      if (options.baseDisp !== undefined) {
        this.baseDisp = options.baseDisp;
        this.smoothedBaseDisp = options.baseDisp;
        this.manualBaseDisp = options.baseDisp;
      }
      if (options.flipY !== undefined) this.flipY = options.flipY;
      if (options.verticalTilt !== undefined) this.verticalTilt = options.verticalTilt;
      if (options.focus !== undefined) this.focus = options.focus;
      if (options.sensitivity !== undefined) this.sensitivity = options.sensitivity;
      if (options.meshDepth !== undefined) this.meshDepth = options.meshDepth;
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
    gl.bindVertexArray(null);

    // Initialize Subdivided 3D Plane Mesh (gridW = 120, gridH = 72)
    this.initMeshGeometry();

    // Compile Pass 1 (SAD Disparity)
    this.disparityProgram = this.createProgram(VERTEX_SHADER_SOURCE, DISPARITY_SAD_FRAG_SOURCE);
    this.dispLocs = {
      sbsTexture: gl.getUniformLocation(this.disparityProgram, 'u_sbsTexture'),
      prevDispTexture: gl.getUniformLocation(this.disparityProgram, 'u_prevDispTexture'),
      daDepthTexture: gl.getUniformLocation(this.disparityProgram, 'u_daDepthTexture'),
      hasDaDepth: gl.getUniformLocation(this.disparityProgram, 'u_hasDaDepth'),
      eyeTexelSize: gl.getUniformLocation(this.disparityProgram, 'u_eyeTexelSize'),
      maxDisp: gl.getUniformLocation(this.disparityProgram, 'u_maxDisp'),
      baseDisp: gl.getUniformLocation(this.disparityProgram, 'u_baseDisp'),
      temporalWeight: gl.getUniformLocation(this.disparityProgram, 'u_temporalWeight'),
      flipY: gl.getUniformLocation(this.disparityProgram, 'u_flipY'),
    };

    // Compile Pass 2 (Tiefling Depth Filter)
    this.depthFilterProgram = this.createProgram(VERTEX_SHADER_SOURCE, TIEFLING_DEPTH_FILTER_FRAG_SOURCE);
    this.depthFilterLocs = {
      rawDispTexture: gl.getUniformLocation(this.depthFilterProgram, 'u_rawDispTexture'),
      sbsTexture: gl.getUniformLocation(this.depthFilterProgram, 'u_sbsTexture'),
      daDepthTexture: gl.getUniformLocation(this.depthFilterProgram, 'u_daDepthTexture'),
      hasDaDepth: gl.getUniformLocation(this.depthFilterProgram, 'u_hasDaDepth'),
      dispTexelSize: gl.getUniformLocation(this.depthFilterProgram, 'u_dispTexelSize'),
      windowCrop: gl.getUniformLocation(this.depthFilterProgram, 'u_windowCrop'),
    };

    // Compile Pass 3 (Perspective Window & View Modes)
    this.perspectiveProgram = this.createProgram(VERTEX_SHADER_SOURCE, PERSPECTIVE_WINDOW_FRAG_SOURCE);
    this.perspLocs = {
      sbsTexture: gl.getUniformLocation(this.perspectiveProgram, 'u_sbsTexture'),
      dispTexture: gl.getUniformLocation(this.perspectiveProgram, 'u_dispTexture'),
      viewShift: gl.getUniformLocation(this.perspectiveProgram, 'u_viewShift'),
      maxDisparityUV: gl.getUniformLocation(this.perspectiveProgram, 'u_maxDisparityUV'),
      windowCrop: gl.getUniformLocation(this.perspectiveProgram, 'u_windowCrop'),
      eyeTexelSize: gl.getUniformLocation(this.perspectiveProgram, 'u_eyeTexelSize'),
      dispTexelSize: gl.getUniformLocation(this.perspectiveProgram, 'u_dispTexelSize'),
      zeroDispNorm: gl.getUniformLocation(this.perspectiveProgram, 'u_zeroDispNorm'),
      baseDisp: gl.getUniformLocation(this.perspectiveProgram, 'u_baseDisp'),
      renderMode: gl.getUniformLocation(this.perspectiveProgram, 'u_renderMode'),
      flipY: gl.getUniformLocation(this.perspectiveProgram, 'u_flipY'),
    };

    // Compile Tiefling 3D Mesh Program
    this.tieflingMeshProgram = this.createProgram(TIEFLING_MESH_VERTEX_SHADER_SOURCE, PERSPECTIVE_WINDOW_FRAG_SOURCE);
    this.tieflingLocs = {
      sbsTexture: gl.getUniformLocation(this.tieflingMeshProgram, 'u_sbsTexture'),
      dispTexture: gl.getUniformLocation(this.tieflingMeshProgram, 'u_dispTexture'),
      viewShift: gl.getUniformLocation(this.tieflingMeshProgram, 'u_viewShift'),
      verticalTilt: gl.getUniformLocation(this.tieflingMeshProgram, 'u_verticalTilt'),
      meshDepth: gl.getUniformLocation(this.tieflingMeshProgram, 'u_meshDepth'),
      focus: gl.getUniformLocation(this.tieflingMeshProgram, 'u_focus'),
      sensitivity: gl.getUniformLocation(this.tieflingMeshProgram, 'u_sensitivity'),
      windowCrop: gl.getUniformLocation(this.tieflingMeshProgram, 'u_windowCrop'),
      flipY: gl.getUniformLocation(this.tieflingMeshProgram, 'u_flipY'),
      renderMode: gl.getUniformLocation(this.tieflingMeshProgram, 'u_renderMode'),
      eyeTexelSize: gl.getUniformLocation(this.tieflingMeshProgram, 'u_eyeTexelSize'),
      dispTexelSize: gl.getUniformLocation(this.tieflingMeshProgram, 'u_dispTexelSize'),
      maxDisparityUV: gl.getUniformLocation(this.tieflingMeshProgram, 'u_maxDisparityUV'),
      zeroDispNorm: gl.getUniformLocation(this.tieflingMeshProgram, 'u_zeroDispNorm'),
      baseDisp: gl.getUniformLocation(this.tieflingMeshProgram, 'u_baseDisp'),
    };

    // Textures setup
    this.sbsTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    // DepthAnything external depth texture setup (gl.R8 with fallback, LINEAR filtering)
    this.daDepthTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.daDepthTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    const initialDepth = new Uint8Array([128]);
    try {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, initialDepth);
    } catch {
      try {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, 1, 1, 0, gl.LUMINANCE, gl.UNSIGNED_BYTE, initialDepth);
      } catch {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 128, 128, 255]));
      }
    }

    this.allocateDisparityFbos(this.eyeWidth, this.eyeHeight);
  }

  private initMeshGeometry(): void {
    const gl = this.gl;
    const gridW = 200;
    const gridH = 120;
    const numVerts = (gridW + 1) * (gridH + 1);
    const uvs = new Float32Array(numVerts * 2);

    let vIdx = 0;
    for (let y = 0; y <= gridH; y++) {
      const v = y / gridH;
      for (let x = 0; x <= gridW; x++) {
        const u = x / gridW;
        uvs[vIdx++] = u;
        uvs[vIdx++] = v;
      }
    }

    const numQuads = gridW * gridH;
    const indices = new Uint16Array(numQuads * 6);
    let iIdx = 0;
    for (let y = 0; y < gridH; y++) {
      for (let x = 0; x < gridW; x++) {
        const row1 = y * (gridW + 1);
        const row2 = (y + 1) * (gridW + 1);
        const i0 = row1 + x;
        const i1 = row1 + x + 1;
        const i2 = row2 + x;
        const i3 = row2 + x + 1;

        indices[iIdx++] = i0;
        indices[iIdx++] = i2;
        indices[iIdx++] = i1;

        indices[iIdx++] = i1;
        indices[iIdx++] = i2;
        indices[iIdx++] = i3;
      }
    }
    this.meshIndexCount = indices.length; // 200 * 120 * 6 = 144,000 indices

    this.meshVao = gl.createVertexArray();
    gl.bindVertexArray(this.meshVao);

    this.meshVbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.meshVbo);
    gl.bufferData(gl.ARRAY_BUFFER, uvs, gl.STATIC_DRAW);

    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.meshIbo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.meshIbo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

    gl.bindVertexArray(null);
  }

  private allocateDisparityFbos(width: number, height: number): void {
    const gl = this.gl;
    this.eyeWidth = width;
    this.eyeHeight = height;

    this.dispWidth = 72;
    this.dispHeight = 44;

    for (let i = 0; i < 2; i++) {
      if (!this.dispTextures[i]) {
        this.dispTextures[i] = gl.createTexture();
      }
      gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[i]);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, this.dispWidth, this.dispHeight, 0, gl.RED, gl.UNSIGNED_BYTE, null);

      if (!this.disparityFbos[i]) {
        this.disparityFbos[i] = gl.createFramebuffer();
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.disparityFbos[i]);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTextures[i], 0);

      const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
      if (status !== gl.FRAMEBUFFER_COMPLETE) {
        console.warn(`Disparity FBO ${i} incomplete with R8 (${status}), falling back to RGBA8`);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.dispWidth, this.dispHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.dispTextures[i], 0);
      }
    }

    // Allocate intermediate temporary filter depth texture and FBO (Pass 2A)
    if (!this.tempFilterTexture) {
      this.tempFilterTexture = gl.createTexture();
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tempFilterTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, this.dispWidth, this.dispHeight, 0, gl.RED, gl.UNSIGNED_BYTE, null);

    if (!this.tempFilterFbo) {
      this.tempFilterFbo = gl.createFramebuffer();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tempFilterFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tempFilterTexture, 0);

    const tStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (tStatus !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`Temp disparity FBO incomplete with R8 (${tStatus}), falling back to RGBA8`);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.dispWidth, this.dispHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.tempFilterTexture, 0);
    }

    // Allocate final filtered depth texture and FBO (Pass 2B)
    if (!this.filteredDispTexture) {
      this.filteredDispTexture = gl.createTexture();
    }
    gl.bindTexture(gl.TEXTURE_2D, this.filteredDispTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, this.dispWidth, this.dispHeight, 0, gl.RED, gl.UNSIGNED_BYTE, null);

    if (!this.filteredDispFbo) {
      this.filteredDispFbo = gl.createFramebuffer();
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.filteredDispFbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.filteredDispTexture, 0);

    const fStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (fStatus !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn(`Filtered disparity FBO incomplete with R8 (${fStatus}), falling back to RGBA8`);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, this.dispWidth, this.dispHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.filteredDispTexture, 0);
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
    gl.bindAttribLocation(prog, 0, 'a_position');
    gl.bindAttribLocation(prog, 0, 'a_uv');
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
    this.meshDepth = 0.15 + (maxDisp / 36.0) * 0.65;
  }

  public setFlipY(flip: boolean): void {
    this.flipY = flip;
  }

  public setVerticalTilt(tilt: number): void {
    this.verticalTilt = Math.max(-1.0, Math.min(1.0, tilt));
  }

  public getVerticalTilt(): number {
    return this.verticalTilt;
  }

  public setFocus(focus: number): void {
    this.focus = Math.max(0.0, Math.min(1.0, focus));
  }

  public getFocus(): number {
    return this.focus;
  }

  public setSensitivity(sensitivity: number): void {
    this.sensitivity = Math.max(0.0, Math.min(2.0, sensitivity));
  }

  public getSensitivity(): number {
    return this.sensitivity;
  }

  public setMeshDepth(meshDepth: number): void {
    this.meshDepth = Math.max(0.0, Math.min(2.0, meshDepth));
  }

  public getMeshDepth(): number {
    return this.meshDepth;
  }

  public setThreeDSMetadata(parallaxOffset: number, estimatedBaseDisp: number | null): void {
    this.parallaxOffset = parallaxOffset;
    this.metadataBaseDisp = estimatedBaseDisp;
    if (estimatedBaseDisp !== null && Number.isFinite(estimatedBaseDisp)) {
      this.baseDisp = estimatedBaseDisp;
      this.smoothedBaseDisp = estimatedBaseDisp;
    }
  }

  public getThreeDSMetadata(): { parallaxOffset: number | null; estimatedBaseDisp: number | null } {
    return {
      parallaxOffset: this.parallaxOffset,
      estimatedBaseDisp: this.metadataBaseDisp,
    };
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

  public setDepthProvider(
    provider: ((timeSec: number) => { depth: Uint8Array; width: number; height: number } | null) | null
  ): void {
    this.depthProvider = provider;
  }

  public getDepthProvider(): ((timeSec: number) => { depth: Uint8Array; width: number; height: number } | null) | null {
    return this.depthProvider;
  }

  /**
   * Upload an external depth map (e.g. from DepthAnythingV2/V3) to the GPU.
   * Format: Uint8Array of length (width * height), values 0..255.
   * Optimized with zero-reallocation gl.texSubImage2D (< 0.02 ms at 60 FPS).
   */
  public uploadExternalDepthMap(depthData: Uint8Array, width: number, height: number): void {
    // If buffer reference is identical and dimensions match, skip upload entirely (<0.02ms)
    if (
      this.hasDaDepth &&
      this.daDepthWidth === width &&
      this.daDepthHeight === height &&
      this.lastUploadedDepthBuffer === depthData
    ) {
      return;
    }

    const gl = this.gl;
    if (!this.daDepthTexture) {
      this.daDepthTexture = gl.createTexture();
    }
    gl.bindTexture(gl.TEXTURE_2D, this.daDepthTexture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);

    // If texture is already allocated with matching dimensions, use fast texSubImage2D without reallocation
    if (this.hasDaDepth && this.daDepthWidth === width && this.daDepthHeight === height) {
      try {
        if (this.daDepthFormat === gl.RGBA) {
          const rgba = new Uint8Array(width * height * 4);
          for (let i = 0; i < depthData.length; i++) {
            const v = depthData[i];
            const idx = i * 4;
            rgba[idx] = v;
            rgba[idx + 1] = v;
            rgba[idx + 2] = v;
            rgba[idx + 3] = 255;
          }
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, rgba);
        } else {
          gl.texSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            width,
            height,
            this.daDepthFormat || gl.RED,
            gl.UNSIGNED_BYTE,
            depthData
          );
        }
        this.lastUploadedDepthBuffer = depthData;
        return;
      } catch {
        // Fall back to full reallocation if texSubImage2D encounters an issue
      }
    }

    this.daDepthWidth = width;
    this.daDepthHeight = height;

    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    try {
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.R8,
        width,
        height,
        0,
        gl.RED,
        gl.UNSIGNED_BYTE,
        depthData
      );
      this.daDepthFormat = gl.RED;
    } catch {
      try {
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.LUMINANCE,
          width,
          height,
          0,
          gl.LUMINANCE,
          gl.UNSIGNED_BYTE,
          depthData
        );
        this.daDepthFormat = gl.LUMINANCE;
      } catch {
        const rgba = new Uint8Array(width * height * 4);
        for (let i = 0; i < depthData.length; i++) {
          const v = depthData[i];
          const idx = i * 4;
          rgba[idx] = v;
          rgba[idx + 1] = v;
          rgba[idx + 2] = v;
          rgba[idx + 3] = 255;
        }
        gl.texImage2D(
          gl.TEXTURE_2D,
          0,
          gl.RGBA,
          width,
          height,
          0,
          gl.RGBA,
          gl.UNSIGNED_BYTE,
          rgba
        );
        this.daDepthFormat = gl.RGBA;
      }
    }

    this.hasDaDepth = true;
    this.lastUploadedDepthBuffer = depthData;
  }

  public hasExternalDepthMap(): boolean {
    return this.hasDaDepth;
  }

  public clearExternalDepthMap(): void {
    this.hasDaDepth = false;
    this.lastUploadedDepthBuffer = null;
  }

  /**
   * Fast stereo baseline estimation.
   * Defaults to search range [-80, 24] (covering the true 3DS divergence window).
   * Supports custom range for automated tests.
   */
  public estimateGlobalBaselineFromRgba(
    rgba: Uint8Array,
    sbsWidth: number,
    sbsHeight: number,
    minShift: number = -80,
    maxShift: number = 24
  ): number {
    const eyeW = Math.floor(sbsWidth / 2);
    const eyeH = sbsHeight;

    const yStart = Math.max(2, Math.floor(eyeH * 0.2));
    const yEnd = Math.min(eyeH - 2, Math.floor(eyeH * 0.8));
    const xStart = Math.max(4, Math.floor(eyeW * 0.2));
    const xEnd = Math.min(eyeW - 4, Math.floor(eyeW * 0.8));

    let bestCoarseD = minShift;
    let minCoarseMad = 1e9;

    // Coarse pass: step 2
    for (let d = minShift; d <= maxShift; d += 2) {
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
    const fineMin = Math.max(minShift, bestCoarseD - 2);
    const fineMax = Math.min(maxShift, bestCoarseD + 2);

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

    const detected = this.estimateGlobalBaselineFromRgba(rgba, sbsWidth, sbsHeight, -80, 24);
    // Clamp to valid 3DS hardware stereo baseline range [-80, 24]
    const clamped = Math.max(-80, Math.min(24, detected));

    if (this.frameCountTotal <= 1 && this.metadataBaseDisp === null) {
      this.smoothedBaseDisp = clamped;
    } else {
      this.smoothedBaseDisp = 0.8 * this.smoothedBaseDisp + 0.2 * clamped;
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

        // Estimate on downsampled buffer: -20 to 8 corresponds to [-80, 32] on 480 eye
        const detectedHalf = this.estimateGlobalBaselineFromRgba(raw, 240, 60, -20, 8);
        const scale = this.eyeWidth / 120;
        const scaled = Math.round(detectedHalf * scale);
        const clamped = Math.max(-80, Math.min(24, scaled));

        if (this.frameCountTotal <= 1 && this.metadataBaseDisp === null) {
          this.smoothedBaseDisp = clamped;
        } else {
          this.smoothedBaseDisp = 0.8 * this.smoothedBaseDisp + 0.2 * clamped;
        }
        this.baseDisp = Math.round(this.smoothedBaseDisp);
      }
    } catch {
      // Gracefully ignore if 2D context extraction fails in headless environments
    }
  }

  /**
   * Primary render method accepting an HTMLVideoElement or TexImageSource.
   */
  public async renderVideoFrame(
    video: TexImageSource,
    viewShift: number = 0.5,
    verticalTilt?: number
  ): Promise<ViewerTelemetry> {
    if (verticalTilt !== undefined) {
      this.setVerticalTilt(verticalTilt);
    }
    const gl = this.gl;
    const startTotal = performance.now();
    this.frameCountTotal++;

    // Synchronous depth provider lookup for zero-desync frame-accurate playback & export
    if (this.depthProvider && typeof (video as HTMLVideoElement).currentTime === 'number') {
      const synced = this.depthProvider((video as HTMLVideoElement).currentTime);
      if (synced) {
        this.uploadExternalDepthMap(synced.depth, synced.width, synced.height);
      }
    }

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

    if (!this.hasDaDepth) {
      if (this.backend === 'gpu') {
        const startDisp = performance.now();
        this.runGpuDisparityPass();
        this.runDepthFilterPass();
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
        this.runDepthFilterPass();
      }
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
    customFlipY?: boolean,
    verticalTilt?: number
  ): ViewerTelemetry {
    if (verticalTilt !== undefined) {
      this.setVerticalTilt(verticalTilt);
    }
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
    if (!this.hasDaDepth && this.backend === 'gpu') {
      const startDisp = performance.now();
      this.runGpuDisparityPass();
      this.runDepthFilterPass();
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
    gl.viewport(0, 0, this.dispWidth, this.dispHeight);

    gl.useProgram(this.disparityProgram);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.dispLocs.sbsTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[readIdx]);
    gl.uniform1i(this.dispLocs.prevDispTexture, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.daDepthTexture);
    gl.uniform1i(this.dispLocs.daDepthTexture, 2);
    gl.uniform1f(this.dispLocs.hasDaDepth, this.hasDaDepth ? 1.0 : 0.0);

    gl.uniform2f(this.dispLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
    gl.uniform1i(this.dispLocs.maxDisp, 16);
    gl.uniform1i(this.dispLocs.baseDisp, this.baseDisp);

    const temporalWeight = this.frameCountTotal > 1 ? 0.80 : 0.0;
    gl.uniform1f(this.dispLocs.temporalWeight, temporalWeight);
    gl.uniform1f(this.dispLocs.flipY, 0.0); // Pass 1 FBO disparity coordinates match sbsTexture 1:1

    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private runDepthFilterPass(): void {
    const gl = this.gl;
    const writeIdx = this.pingPongIdx;

    if (!this.filteredDispFbo || !this.tempFilterFbo || !this.depthFilterProgram) return;

    gl.useProgram(this.depthFilterProgram);
    gl.bindVertexArray(this.vao);

    gl.uniform2f(this.depthFilterLocs.dispTexelSize, 1.0 / this.dispWidth, 1.0 / this.dispHeight);
    if (this.depthFilterLocs.windowCrop) {
      gl.uniform1f(this.depthFilterLocs.windowCrop, this.windowCrop);
    }

    // Pass 2A: dispTextures[writeIdx] -> tempFilterFbo
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.tempFilterFbo);
    gl.viewport(0, 0, this.dispWidth, this.dispHeight);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.dispTextures[writeIdx]);
    gl.uniform1i(this.depthFilterLocs.rawDispTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.depthFilterLocs.sbsTexture, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.daDepthTexture);
    gl.uniform1i(this.depthFilterLocs.daDepthTexture, 2);
    gl.uniform1f(this.depthFilterLocs.hasDaDepth, this.hasDaDepth ? 1.0 : 0.0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Pass 2B: tempFilterTexture -> filteredDispFbo
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.filteredDispFbo);
    gl.viewport(0, 0, this.dispWidth, this.dispHeight);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.tempFilterTexture);
    gl.uniform1i(this.depthFilterLocs.rawDispTexture, 0);

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
    gl.uniform1i(this.depthFilterLocs.sbsTexture, 1);

    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, this.daDepthTexture);
    gl.uniform1i(this.depthFilterLocs.daDepthTexture, 2);
    gl.uniform1f(this.depthFilterLocs.hasDaDepth, this.hasDaDepth ? 1.0 : 0.0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindVertexArray(null);
  }

  private runPerspectivePass(viewShift: number, customFlipY?: boolean): void {
    const gl = this.gl;
    const currDispIdx = this.pingPongIdx;
    const effectiveFlipY = customFlipY !== undefined ? (customFlipY ? 1.0 : 0.0) : (this.flipY ? 1.0 : 0.0);
    const clampedShift = Math.max(0.0, Math.min(1.0, viewShift));

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);

    const activeDispTexture = (this.hasDaDepth && this.daDepthTexture)
      ? this.daDepthTexture
      : (this.filteredDispTexture ?? this.dispTextures[currDispIdx]);

    const activeDispTexelX = (this.hasDaDepth && this.daDepthWidth > 0)
      ? 1.0 / this.daDepthWidth
      : 1.0 / this.dispWidth;
    const activeDispTexelY = (this.hasDaDepth && this.daDepthHeight > 0)
      ? 1.0 / this.daDepthHeight
      : 1.0 / this.dispHeight;

    if (this.renderMode === 0) {
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LEQUAL);
      gl.clearDepth(1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

      gl.useProgram(this.tieflingMeshProgram);
      gl.bindVertexArray(this.meshVao);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
      gl.uniform1i(this.tieflingLocs.sbsTexture, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, activeDispTexture);
      gl.uniform1i(this.tieflingLocs.dispTexture, 1);

      gl.uniform1f(this.tieflingLocs.viewShift, clampedShift);
      gl.uniform1f(this.tieflingLocs.verticalTilt, this.verticalTilt);
      gl.uniform1f(this.tieflingLocs.meshDepth, this.meshDepth);
      gl.uniform1f(this.tieflingLocs.focus, this.focus);
      gl.uniform1f(this.tieflingLocs.sensitivity, this.sensitivity);
      gl.uniform1f(this.tieflingLocs.windowCrop, this.windowCrop);
      gl.uniform1f(this.tieflingLocs.flipY, effectiveFlipY);
      gl.uniform1i(this.tieflingLocs.renderMode, 0);

      if (this.tieflingLocs.eyeTexelSize) {
        gl.uniform2f(this.tieflingLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
      }
      if (this.tieflingLocs.dispTexelSize) {
        gl.uniform2f(this.tieflingLocs.dispTexelSize, activeDispTexelX, activeDispTexelY);
      }
      if (this.tieflingLocs.maxDisparityUV) {
        gl.uniform1f(this.tieflingLocs.maxDisparityUV, Math.min(0.04, this.maxDisp / this.eyeWidth));
      }
      if (this.tieflingLocs.zeroDispNorm) {
        gl.uniform1f(this.tieflingLocs.zeroDispNorm, 0.5);
      }
      if (this.tieflingLocs.baseDisp) {
        gl.uniform1i(this.tieflingLocs.baseDisp, this.baseDisp);
      }

      gl.drawElements(gl.TRIANGLES, this.meshIndexCount, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
    } else {
      gl.disable(gl.DEPTH_TEST);

      gl.useProgram(this.perspectiveProgram);
      gl.bindVertexArray(this.vao);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.sbsTexture);
      gl.uniform1i(this.perspLocs.sbsTexture, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, activeDispTexture);
      gl.uniform1i(this.perspLocs.dispTexture, 1);

      gl.uniform1f(this.perspLocs.viewShift, clampedShift);
      gl.uniform1f(this.perspLocs.maxDisparityUV, Math.min(0.04, this.maxDisp / this.eyeWidth));
      gl.uniform1f(this.perspLocs.windowCrop, this.windowCrop);
      gl.uniform2f(this.perspLocs.eyeTexelSize, 1.0 / this.eyeWidth, 1.0 / this.eyeHeight);
      gl.uniform2f(this.perspLocs.dispTexelSize, activeDispTexelX, activeDispTexelY);
      gl.uniform1f(this.perspLocs.zeroDispNorm, 0.5);
      gl.uniform1i(this.perspLocs.baseDisp, this.baseDisp);
      gl.uniform1i(this.perspLocs.renderMode, this.renderMode);
      gl.uniform1f(this.perspLocs.flipY, effectiveFlipY);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
    }

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

  public getMeshIndexCount(): number {
    return this.meshIndexCount;
  }

  public getMeshGridDimensions(): { width: number; height: number } {
    return { width: 200, height: 120 };
  }

  public getDaDepthDimensions(): { width: number; height: number } {
    return { width: this.daDepthWidth, height: this.daDepthHeight };
  }

  public destroy(): void {
    const gl = this.gl;
    if (this.disparityProgram) gl.deleteProgram(this.disparityProgram);
    if (this.depthFilterProgram) gl.deleteProgram(this.depthFilterProgram);
    if (this.perspectiveProgram) gl.deleteProgram(this.perspectiveProgram);
    if (this.tieflingMeshProgram) gl.deleteProgram(this.tieflingMeshProgram);
    if (this.sbsTexture) gl.deleteTexture(this.sbsTexture);
    if (this.daDepthTexture) gl.deleteTexture(this.daDepthTexture);
    for (let i = 0; i < 2; i++) {
      if (this.dispTextures[i]) gl.deleteTexture(this.dispTextures[i]);
      if (this.disparityFbos[i]) gl.deleteFramebuffer(this.disparityFbos[i]);
    }
    if (this.filteredDispTexture) gl.deleteTexture(this.filteredDispTexture);
    if (this.filteredDispFbo) gl.deleteFramebuffer(this.filteredDispFbo);
    if (this.tempFilterTexture) gl.deleteTexture(this.tempFilterTexture);
    if (this.tempFilterFbo) gl.deleteFramebuffer(this.tempFilterFbo);
    if (this.vbo) gl.deleteBuffer(this.vbo);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.meshVbo) gl.deleteBuffer(this.meshVbo);
    if (this.meshIbo) gl.deleteBuffer(this.meshIbo);
    if (this.meshVao) gl.deleteVertexArray(this.meshVao);
    this.depthProvider = null;
    this.lastUploadedDepthBuffer = null;
  }
}

