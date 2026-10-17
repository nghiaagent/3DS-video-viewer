#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform sampler2D u_dispTexture;

uniform float u_viewShift;       // 0.0 (Left) .. 1.0 (Right), default 0.5
uniform float u_maxDisparityUV;  // e.g. 16.0 / 480.0 = 0.033
uniform float u_windowCrop;      // default 0.06
uniform vec2 u_eyeTexelSize;     // default vec2(1.0 / 480.0, 1.0 / 240.0)
uniform vec2 u_dispTexelSize;    // disparity FBO texel size e.g. vec2(1.0 / 80.0, 1.0 / 40.0)
uniform float u_zeroDispNorm;    // default 0.5
uniform int u_baseDisp;          // global horizontal baseline convergence offset (pixels)
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

    // Floating perspective window pan
    float t = clamp(u_viewShift, 0.0, 1.0);
    float activeWidth = 1.0 - (2.0 * u_windowCrop);
    float windowOffset = (t - 0.5) * (2.0 * u_windowCrop);
    vec2 eyeUV = vec2(
        clamp(u_windowCrop + v_uv.x * activeWidth + windowOffset, 0.01, 0.99),
        clamp(u_windowCrop + v_uv.y * (1.0 - 2.0 * u_windowCrop), 0.01, 0.99)
    );

    // 13-tap Gaussian filter on the coarse disparity texture for silky-smooth 2.5D surface
    vec2 dTexel = (u_dispTexelSize.x > 0.0) ? u_dispTexelSize : (u_eyeTexelSize * 6.0);
    float dCenter = texture(u_dispTexture, eyeUV).r;
    float dL1 = texture(u_dispTexture, eyeUV - vec2(dTexel.x, 0.0)).r;
    float dR1 = texture(u_dispTexture, eyeUV + vec2(dTexel.x, 0.0)).r;
    float dU1 = texture(u_dispTexture, eyeUV + vec2(0.0, dTexel.y)).r;
    float dD1 = texture(u_dispTexture, eyeUV - vec2(0.0, dTexel.y)).r;
    float dL2 = texture(u_dispTexture, eyeUV - vec2(2.0 * dTexel.x, 0.0)).r;
    float dR2 = texture(u_dispTexture, eyeUV + vec2(2.0 * dTexel.x, 0.0)).r;
    float dU2 = texture(u_dispTexture, eyeUV + vec2(0.0, 2.0 * dTexel.y)).r;
    float dD2 = texture(u_dispTexture, eyeUV - vec2(0.0, 2.0 * dTexel.y)).r;
    float dC1 = texture(u_dispTexture, eyeUV + vec2(dTexel.x, dTexel.y)).r;
    float dC2 = texture(u_dispTexture, eyeUV + vec2(-dTexel.x, dTexel.y)).r;
    float dC3 = texture(u_dispTexture, eyeUV + vec2(dTexel.x, -dTexel.y)).r;
    float dC4 = texture(u_dispTexture, eyeUV + vec2(-dTexel.x, -dTexel.y)).r;

    float dNorm = (dCenter * 4.0 +
        (dL1 + dR1 + dU1 + dD1) * 2.0 +
        (dL2 + dR2 + dU2 + dD2) * 1.0 +
        (dC1 + dC2 + dC3 + dC4) * 1.0) / 20.0;

    // Mode 1: Disparity Heatmap
    if (u_renderMode == 1) {
        vec3 heatmap = turboColormap(dNorm);
        outColor = vec4(heatmap, 1.0);
        return;
    }

    // Mode 5: Anaglyph 3D (Red / Cyan)
    if (u_renderMode == 5) {
        float leftX = clamp(eyeUV.x * 0.5, 0.001, 0.499);
        float rightX = clamp(0.5 + eyeUV.x * 0.5, 0.501, 0.999);
        vec3 colorL = texture(u_sbsTexture, vec2(leftX, eyeUV.y)).rgb;
        vec3 colorR = texture(u_sbsTexture, vec2(rightX, eyeUV.y)).rgb;
        outColor = vec4(colorL.r, colorR.g, colorR.b, 1.0);
        return;
    }

    // Mode 0: Crisp Apple Vision Pro 2.5D Parallax Window with Anti-Ghosting Edge Locking
    float zeroNorm = (u_zeroDispNorm > 0.0) ? u_zeroDispNorm : 0.5;
    float dispUV = (dNorm - zeroNorm) * u_maxDisparityUV;

    // Bidirectional shift relative to camera position t
    float leftX  = clamp(eyeUV.x - (t * dispUV), 0.002, 0.998) * 0.5;
    float rightX = 0.5 + clamp(eyeUV.x + ((1.0 - t) * dispUV), 0.002, 0.998) * 0.5;

    vec3 colorL = texture(u_sbsTexture, vec2(leftX, eyeUV.y)).rgb;
    vec3 colorR = texture(u_sbsTexture, vec2(rightX, eyeUV.y)).rgb;

    // Photometric agreement and anti-ghosting edge locking
    float diff = length(colorL - colorR);
    float blendFactor = smoothstep(0.0, 1.0, t);
    vec3 dominantColor = (t < 0.5) ? colorL : colorR;
    vec3 secondaryColor = (t < 0.5) ? colorR : colorL;
    float secondaryWeight = (t < 0.5) ? blendFactor : (1.0 - blendFactor);

    // Suppress secondary eye at occlusion boundaries (diff > 0.08) so edges stay 100% sharp
    float agreement = 1.0 - smoothstep(0.07, 0.20, diff);
    vec3 blended = mix(dominantColor, secondaryColor, secondaryWeight * agreement);

    // Apple Vision Spatial Window Glass Border & Portal Depth Glow
    vec2 edgeDist = min(v_uv, 1.0 - v_uv);
    float cornerRadius = 0.035;
    vec2 cornerD = max(vec2(0.0), vec2(cornerRadius) - edgeDist);
    float cornerDist = length(cornerD);
    float portalMask = 1.0 - smoothstep(cornerRadius - 0.006, cornerRadius, cornerDist);

    // Frosted depth-reactive inner rim
    float rimDist = min(edgeDist.x, edgeDist.y);
    float innerRim = smoothstep(0.0, 0.03, rimDist) * (1.0 - smoothstep(0.03, 0.055, rimDist));
    float s = (t - 0.5) * 2.0;
    float depthCueGlow = (dNorm - zeroNorm) * 0.16 * s;
    vec3 frostedRim = vec3(0.68, 0.80, 0.96) * (innerRim * (0.07 + max(0.0, depthCueGlow)));

    // Ambient border feathering
    float frameAlpha = smoothstep(0.0, 0.012, rimDist);
    blended = (blended + frostedRim) * frameAlpha * portalMask;

    outColor = vec4(blended, 1.0);
}
