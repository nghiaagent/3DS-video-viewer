#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform sampler2D u_dispTexture;

uniform float u_viewShift;       // 0.0 (Left) .. 1.0 (Right), default 0.5
uniform float u_maxDisparityUV;  // e.g. 24.0 / 480.0 = 0.05
uniform float u_windowCrop;      // default 0.06
uniform vec2 u_eyeTexelSize;     // default vec2(1.0 / 480.0, 1.0 / 240.0)
uniform float u_zeroDispNorm;    // default 4.0 / 28.0 (point where d = 0)
uniform int u_renderMode;        // 0: Parallax, 1: Heatmap, 2: Left, 3: Right, 4: SBS, 5: Anaglyph

// Google Turbo colormap for perceptually uniform depth heatmap visualization
vec3 turboColormap(float x) {
    const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

    x = clamp(x, 0.0, 1.0);
    vec4 v4 = vec4(1.0, x, x * x, x * x * x);
    vec2 v2 = v4.zw * v4.z;
    return clamp(vec3(
        dot(v4, kRedVec4) + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4) + dot(v2, kBlueVec2)
    ), 0.0, 1.0);
}

void main() {
    // Mode 4: Raw Side-by-Side (SBS) pass-through
    if (u_renderMode == 4) {
        outColor = texture(u_sbsTexture, v_uv);
        return;
    }

    // Mode 2: Left Eye Only
    if (u_renderMode == 2) {
        outColor = texture(u_sbsTexture, vec2(v_uv.x * 0.5, v_uv.y));
        return;
    }

    // Mode 3: Right Eye Only
    if (u_renderMode == 3) {
        outColor = texture(u_sbsTexture, vec2(0.5 + v_uv.x * 0.5, v_uv.y));
        return;
    }

    // Apply floating stereo window crop
    vec2 eyeUV = v_uv * (1.0 - 2.0 * u_windowCrop) + u_windowCrop;

    // 5-tap cross blur on disparity texture for smooth warp transitions
    float dCenter = texture(u_dispTexture, eyeUV).r;
    float dL = texture(u_dispTexture, eyeUV - vec2(u_eyeTexelSize.x, 0.0)).r;
    float dR = texture(u_dispTexture, eyeUV + vec2(u_eyeTexelSize.x, 0.0)).r;
    float dU = texture(u_dispTexture, eyeUV + vec2(0.0, u_eyeTexelSize.y)).r;
    float dD = texture(u_dispTexture, eyeUV - vec2(0.0, u_eyeTexelSize.y)).r;
    float dNorm = (dCenter * 2.0 + dL + dR + dU + dD) / 6.0;

    // Mode 1: Disparity Heatmap
    if (u_renderMode == 1) {
        vec3 heatmap = turboColormap(dNorm);
        outColor = vec4(heatmap, 1.0);
        return;
    }

    // Mode 5: Anaglyph 3D (Red / Cyan)
    if (u_renderMode == 5) {
        vec3 colorL = texture(u_sbsTexture, vec2(eyeUV.x * 0.5, eyeUV.y)).rgb;
        vec3 colorR = texture(u_sbsTexture, vec2(0.5 + eyeUV.x * 0.5, eyeUV.y)).rgb;
        outColor = vec4(colorL.r, colorR.g, colorR.b, 1.0);
        return;
    }

    // Mode 0 (Default): 2D Shifting Perspective Parallax Window
    float t = clamp(u_viewShift, 0.0, 1.0);
    float dispUV = (dNorm - u_zeroDispNorm) * u_maxDisparityUV;

    // Bidirectional shift relative to camera position t
    float shiftL = t * dispUV;
    float shiftR = -(1.0 - t) * dispUV;

    vec2 uvLeft = vec2(
        clamp((eyeUV.x + shiftL) * 0.5, 0.0, 0.4999),
        clamp(eyeUV.y, 0.0, 1.0)
    );

    vec2 uvRight = vec2(
        clamp(0.5 + (eyeUV.x + shiftR) * 0.5, 0.5, 0.9999),
        clamp(eyeUV.y, 0.0, 1.0)
    );

    vec3 colorL = texture(u_sbsTexture, uvLeft).rgb;
    vec3 blended = mix(colorL, colorR, t);

    // Apple Vision style subtle edge feathering into dark glass frame
    vec2 edgeDist = min(v_uv, 1.0 - v_uv);
    float edgeFactor = clamp(min(edgeDist.x, edgeDist.y) / 0.02, 0.0, 1.0);
    edgeFactor = smoothstep(0.0, 1.0, edgeFactor);
    blended *= (0.88 + 0.12 * edgeFactor);

    outColor = vec4(blended, 1.0);
}
