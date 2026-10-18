/**
 * Renders the viewer's whole "stage" for export: dark background, ambient glow, the tilted glass window with its shadows,
 * rim and highlight, and the live 3D picture inside it. It reproduces the CSS in style.css / index.html (see frameGeometry.ts
 * for the shared numbers) so an exported video looks like the page, border and all.
 *
 * It owns a separate WebGL2 context: the viewer's canvas is uploaded as a texture each frame, and every output pixel is
 * mapped back onto the tilted window plane by a ray-plane intersection (the exact inverse of the CSS 3D transform).
 */
import { DEFAULT_GLOW, edgeGlowRgb, type GlowState } from '../renderer/contentGlow.js';
import { FRAME, windowBasis } from '../renderer/frameGeometry.js';
import type { WindowPose } from '../renderer/motionPath.js';

const VERTEX_SOURCE = `#version 300 es
in vec2 a_position;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAGMENT_SOURCE = `#version 300 es
precision highp float;

out vec4 outColor;

uniform sampler2D u_content;      // the viewer canvas, top row first
uniform vec2 u_stage;             // output size in px
uniform vec2 u_win;               // window size in px
uniform float u_radius;           // window corner radius in px
uniform float u_scale;            // output px per reference px

uniform vec2 u_translate;         // window centre offset in px
uniform vec3 u_ex;                // window x axis in stage space
uniform vec3 u_ey;                // window y axis in stage space
uniform float u_persp;            // perspective distance in px

uniform vec2 u_shadowOff;         // px, in window-plane coordinates
uniform vec2 u_glowOff;           // px, in stage coordinates
uniform float u_poseX;
uniform float u_poseY;
uniform float u_specAngle;        // degrees, CSS convention (0 = toward the top, clockwise)

uniform vec3 u_bg;
uniform vec3 u_haloRgb;
uniform vec3 u_glowC1;            // 0..1, from the picture's colour
uniform vec3 u_glowC2;
uniform float u_glowA1;
uniform float u_glowA2;
uniform float u_haloAlpha;
uniform float u_haloSigma;
uniform float u_shadowAlpha;
uniform float u_shadowSigma;
uniform vec3 u_edgeRgb;           // warm-white border glow
uniform float u_borderAlpha;
uniform float u_innerAlpha;
uniform float u_innerSigma;       // px
uniform float u_outerAlpha;
uniform float u_outerSigma;       // px
uniform vec2 u_glowSize;          // reference px
uniform float u_glowSigma;        // px

const float PI = 3.14159265;

float erfApprox(float x) {
    float s = sign(x);
    x = abs(x);
    float t = 1.0 / (1.0 + 0.3275911 * x);
    float y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-x * x);
    return s * y;
}

// Coverage of a Gaussian-blurred shape from its signed distance (negative inside)
float blurredCoverage(float sd, float sigma) {
    return 0.5 - 0.5 * erfApprox(sd / (max(sigma, 1e-3) * 1.41421356));
}

float sdRoundRect(vec2 p, vec2 halfSize, float r) {
    vec2 q = abs(p) - (halfSize - vec2(r));
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

void main() {
    vec2 frag = vec2(gl_FragCoord.x, u_stage.y - gl_FragCoord.y);   // px, origin top-left, y down
    vec2 q = frag - 0.5 * u_stage;                                    // px from the stage centre
    vec3 col = u_bg;

    // ---- ambient glow: blurred ellipse holding a radial gradient (stage plane, not tilted) ----
    {
        vec2 g = (q - u_glowOff) / u_scale;                          // reference px
        float rd = length(g);
        float reach = 0.5 * length(u_glowSize);                      // CSS circle gradient: farthest-corner radius
        float stop1 = 0.45 * reach;                                  // gradient stops at 45% / 70% of it
        float stop2 = 0.70 * reach;
        vec3 c1 = u_glowC1;
        vec3 c2 = u_glowC2;
        float a;
        vec3 pre;                                                    // premultiplied, as CSS interpolates
        if (rd < stop1) {
            float t = rd / stop1;
            a = mix(u_glowA1, u_glowA2, t);
            pre = mix(c1 * u_glowA1, c2 * u_glowA2, t);
        } else if (rd < stop2) {
            float t = (rd - stop1) / (stop2 - stop1);
            a = u_glowA2 * (1.0 - t);
            pre = c2 * a;
        } else {
            a = 0.0;
            pre = vec3(0.0);
        }
        vec3 rgb = a > 1e-4 ? pre / a : c2;
        float e = length(g / (0.5 * u_glowSize));
        float edge = (e - 1.0) * 0.5 * min(u_glowSize.x, u_glowSize.y) * u_scale;   // approximate signed distance in px
        float mask = blurredCoverage(edge, u_glowSigma);
        col = mix(col, rgb, a * mask);
    }

    // ---- intersect the view ray with the tilted window plane ----
    float a11 = u_ex.x + (q.x / u_persp) * u_ex.z;
    float a12 = u_ey.x + (q.x / u_persp) * u_ey.z;
    float a21 = u_ex.y + (q.y / u_persp) * u_ex.z;
    float a22 = u_ey.y + (q.y / u_persp) * u_ey.z;
    float det = a11 * a22 - a12 * a21;
    vec2 d = q - u_translate;
    vec2 local = vec2(d.x * a22 - d.y * a12, a11 * d.y - a21 * d.x) / det;   // px from the window centre, in the window plane

    vec2 hs = 0.5 * u_win;
    float sd = sdRoundRect(local, hs, u_radius);                      // negative inside the window

    // ---- halo, drop shadow and outer glow, drawn in the window plane so they tilt with it. Like a CSS box-shadow they are
    // only painted outside the window's box. Bottom to top: halo, drop shadow, warm glow. ----
    float outside = smoothstep(-0.5, 0.5, sd);
    col = mix(col, u_haloRgb, blurredCoverage(sd, u_haloSigma) * u_haloAlpha * outside);
    float sdShadow = sdRoundRect(local - u_shadowOff, hs, u_radius);
    col = mix(col, vec3(0.0), blurredCoverage(sdShadow, u_shadowSigma) * u_shadowAlpha * outside);
    col = mix(col, u_edgeRgb, blurredCoverage(sd, u_outerSigma) * u_outerAlpha * outside);

    // ---- the window: live picture, glass sheen, warm inner glow, edge line ----
    float inside = 1.0 - smoothstep(-0.5, 0.5, sd);
    if (inside > 0.0) {
        vec3 pic = texture(u_content, local / u_win + 0.5).rgb;
        col = mix(col, pic, inside);

        float th = u_specAngle * PI / 180.0;
        vec2 dir = vec2(sin(th), -cos(th));
        float len = abs(u_win.x * sin(th)) + abs(u_win.y * cos(th));
        float t = clamp(dot(local, dir) / len + 0.5, 0.0, 1.0);
        float a0 = 0.18 + u_poseX * 0.08;
        float spec = t < 0.4 ? mix(a0, 0.03, t / 0.4) : mix(0.03, 0.0, clamp((t - 0.4) / 0.3, 0.0, 1.0));
        col = mix(col, vec3(1.0), spec * inside);

        // inset glow: the blurred region outside the window, seen from inside
        col = mix(col, u_edgeRgb, blurredCoverage(-sd, u_innerSigma) * u_innerAlpha * inside);

        // inset 0 1px 0: a 1 px line along the top edge, just inside the border
        float fromTop = local.y + hs.y;
        float line = smoothstep(u_scale - 0.5, u_scale + 0.5, fromTop) * (1.0 - smoothstep(2.0 * u_scale - 0.5, 2.0 * u_scale + 0.5, fromTop));
        float flatTop = 1.0 - smoothstep(hs.x - u_radius - 1.0, hs.x - u_radius + 1.0, abs(local.x));
        col = mix(col, vec3(1.0), line * flatTop * inside * (0.35 + u_poseX * 0.15));

        float border = smoothstep(-u_scale - 0.5, -u_scale + 0.5, sd) * inside;     // 1 px line just inside the edge
        col = mix(col, u_edgeRgb, border * u_borderAlpha);
    }

    outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
`;

type Locs = Record<string, WebGLUniformLocation | null>;

export class StageRenderer {
  readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private vao: WebGLVertexArrayObject;
  private vbo: WebGLBuffer;
  private texture: WebGLTexture;
  private locs: Locs = {};

  /**
   * @param width output width in px
   * @param height output height in px
   * @param scale output px per reference px (the window is FRAME.windowWidth * scale wide)
   */
  constructor(
    readonly width: number,
    readonly height: number,
    readonly scale: number
  ) {
    this.canvas = document.createElement('canvas');
    this.canvas.width = width;
    this.canvas.height = height;
    const gl = this.canvas.getContext('webgl2', { alpha: false, antialias: false, preserveDrawingBuffer: true });
    if (!gl) throw new Error('WebGL2 is not available for the export stage renderer');
    this.gl = gl;

    this.program = this.link(VERTEX_SOURCE, FRAGMENT_SOURCE);
    for (const name of [
      'content', 'stage', 'win', 'radius', 'scale', 'translate', 'ex', 'ey', 'persp', 'shadowOff', 'glowOff', 'poseX', 'poseY',
      'specAngle', 'bg', 'haloRgb', 'haloAlpha', 'haloSigma', 'shadowAlpha', 'shadowSigma', 'edgeRgb', 'borderAlpha', 'innerAlpha', 'innerSigma', 'outerAlpha', 'outerSigma',
      'glowSize', 'glowSigma', 'glowC1', 'glowC2', 'glowA1', 'glowA2',
    ]) {
      this.locs[name] = gl.getUniformLocation(this.program, `u_${name}`);
    }

    this.vao = gl.createVertexArray()!;
    gl.bindVertexArray(this.vao);
    this.vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.program, 'a_position');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);

    this.texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  private link(vs: string, fs: string): WebGLProgram {
    const gl = this.gl;
    const compile = (type: number, src: string) => {
      const sh = gl.createShader(type)!;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`Stage shader compilation failed: ${log}`);
      }
      return sh;
    };
    const v = compile(gl.VERTEX_SHADER, vs);
    const f = compile(gl.FRAGMENT_SHADER, fs);
    const prog = gl.createProgram()!;
    gl.attachShader(prog, v);
    gl.attachShader(prog, f);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      throw new Error(`Stage program link failed: ${gl.getProgramInfoLog(prog)}`);
    }
    gl.deleteShader(v);
    gl.deleteShader(f);
    return prog;
  }

  /** Draws one frame: `content` is the viewer canvas, `pose` the window pose for the current view direction. */
  render(content: HTMLCanvasElement, pose: WindowPose, glow: GlowState = DEFAULT_GLOW): HTMLCanvasElement {
    const gl = this.gl;
    const s = this.scale;
    const L = this.locs;

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, content);
    gl.uniform1i(L.content, 0);

    const { ex, ey } = windowBasis(pose.rotateYDeg, pose.rotateXDeg);
    gl.uniform2f(L.stage, this.width, this.height);
    gl.uniform2f(L.win, FRAME.windowWidth * s, FRAME.windowHeight * s);
    gl.uniform1f(L.radius, FRAME.cornerRadius * s);
    gl.uniform1f(L.scale, s);
    gl.uniform2f(L.translate, pose.tx * s, pose.ty * s);
    gl.uniform3f(L.ex, ex[0], ex[1], ex[2]);
    gl.uniform3f(L.ey, ey[0], ey[1], ey[2]);
    gl.uniform1f(L.persp, FRAME.perspective * s);
    gl.uniform2f(L.shadowOff, pose.shadowX * s, pose.shadowY * s);
    gl.uniform2f(L.glowOff, pose.glowX * s, pose.glowY * s);
    gl.uniform1f(L.poseX, pose.x);
    gl.uniform1f(L.poseY, pose.y);
    gl.uniform1f(L.specAngle, pose.specAngleDeg);
    gl.uniform3f(L.bg, FRAME.background[0] / 255, FRAME.background[1] / 255, FRAME.background[2] / 255);
    // The halo takes the glow's inner colour (with the default glow this is the original cyan)
    gl.uniform3f(L.haloRgb, glow.c1[0] / 255, glow.c1[1] / 255, glow.c1[2] / 255);
    gl.uniform3f(L.glowC1, glow.c1[0] / 255, glow.c1[1] / 255, glow.c1[2] / 255);
    gl.uniform3f(L.glowC2, glow.c2[0] / 255, glow.c2[1] / 255, glow.c2[2] / 255);
    gl.uniform1f(L.glowA1, glow.a1);
    gl.uniform1f(L.glowA2, glow.a2);
    gl.uniform1f(L.haloAlpha, FRAME.halo.alpha);
    gl.uniform1f(L.haloSigma, (FRAME.halo.blur / 2) * s);
    gl.uniform1f(L.shadowAlpha, FRAME.shadow.alpha);
    gl.uniform1f(L.shadowSigma, (FRAME.shadow.blur / 2) * s);
    const E = FRAME.edge;
    const edgeRgb = edgeGlowRgb(glow);
    gl.uniform3f(L.edgeRgb, edgeRgb[0] / 255, edgeRgb[1] / 255, edgeRgb[2] / 255);
    gl.uniform1f(L.borderAlpha, E.borderAlpha);
    gl.uniform1f(L.innerAlpha, E.innerAlpha);
    gl.uniform1f(L.innerSigma, (E.innerBlur / 2) * s);
    gl.uniform1f(L.outerAlpha, E.outerAlpha);
    gl.uniform1f(L.outerSigma, (E.outerBlur / 2) * s);
    gl.uniform2f(L.glowSize, FRAME.glow.width, FRAME.glow.height);
    gl.uniform1f(L.glowSigma, FRAME.glow.blur * s);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    return this.canvas;
  }

  destroy(): void {
    const gl = this.gl;
    gl.deleteTexture(this.texture);
    gl.deleteBuffer(this.vbo);
    gl.deleteVertexArray(this.vao);
    gl.deleteProgram(this.program);
  }
}
