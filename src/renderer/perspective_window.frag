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

float getLuma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
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
    float s = (t - 0.5) * 2.0; // -1.0 (Left) to +1.0 (Right)
    float activeWidth = 1.0 - (2.0 * u_windowCrop);
    float windowOffset = (t - 0.5) * (1.2 * u_windowCrop);
    vec2 eyeUV = vec2(
        clamp(u_windowCrop + v_uv.x * activeWidth + windowOffset, 0.005, 0.995),
        clamp(u_windowCrop + v_uv.y * (1.0 - 2.0 * u_windowCrop), 0.005, 0.995)
    );

    // 9-tap bilateral/gaussian smoothed depth dNorm to prevent surface tearing
    vec2 texel = u_eyeTexelSize;
    float d00 = texture(u_dispTexture, eyeUV + vec2(-texel.x, -texel.y)).r;
    float d10 = texture(u_dispTexture, eyeUV + vec2(0.0, -texel.y)).r;
    float d20 = texture(u_dispTexture, eyeUV + vec2(texel.x, -texel.y)).r;
    float d01 = texture(u_dispTexture, eyeUV + vec2(-texel.x, 0.0)).r;
    float dCenter = texture(u_dispTexture, eyeUV).r;
    float d21 = texture(u_dispTexture, eyeUV + vec2(texel.x, 0.0)).r;
    float d02 = texture(u_dispTexture, eyeUV + vec2(-texel.x, texel.y)).r;
    float d12 = texture(u_dispTexture, eyeUV + vec2(0.0, texel.y)).r;
    float d22 = texture(u_dispTexture, eyeUV + vec2(texel.x, texel.y)).r;

    float dNorm = (
        (d00 + d20 + d02 + d22) * 1.0 +
        (d10 + d01 + d21 + d12) * 2.0 +
        dCenter * 4.0
    ) / 16.0;

    // Mode 1: Disparity Heatmap
    if (u_renderMode == 1) {
        vec3 heatmap = turboColormap(dNorm);
        outColor = vec4(heatmap, 1.0);
        return;
    }

    // Global stereo convergence baseline UV offset
    float baseDispUV = float(u_baseDisp) * u_eyeTexelSize.x;

    // Mode 5: Anaglyph 3D (Red / Cyan) with converged baseline
    if (u_renderMode == 5) {
        float leftX = clamp(eyeUV.x * 0.5, 0.001, 0.499);
        float rightX = clamp(0.5 + (eyeUV.x - baseDispUV) * 0.5, 0.501, 0.999);
        vec3 colorL = texture(u_sbsTexture, vec2(leftX, eyeUV.y)).rgb;
        vec3 colorR = texture(u_sbsTexture, vec2(rightX, eyeUV.y)).rgb;
        outColor = vec4(colorL.r, colorR.g, colorR.b, 1.0);
        return;
    }

    // Mode 0: Apple Vision Pro 2.5D Depth-Deforming Ray-Marched Warp
    // Relative depth parallax around zero-disparity plane (dNorm = 0.5)
    float zeroNorm = (u_zeroDispNorm > 0.0) ? u_zeroDispNorm : 0.5;
    float relDispUV = (dNorm - zeroNorm) * 2.0 * u_maxDisparityUV;

    // 4-step parallax occlusion ray-march along relDispUV * s
    vec2 rayStep = vec2(relDispUV * s * 0.25, 0.0);
    vec2 marchedUV = eyeUV;
    float marchedDepth = dNorm;
    for (int step = 0; step < 4; ++step) {
        marchedUV += rayStep;
        float sampleD = texture(u_dispTexture, marchedUV).r;
        if (sampleD > marchedDepth) {
            marchedDepth = mix(marchedDepth, sampleD, 0.35);
        }
    }
    float refinedRelDisp = (marchedDepth - zeroNorm) * 2.0 * u_maxDisparityUV;

    // Sample Left eye and converged Right eye
    float leftX = clamp((eyeUV.x + s * refinedRelDisp * 0.85) * 0.5, 0.001, 0.499);
    float rightInner = eyeUV.x - baseDispUV + s * refinedRelDisp * 0.85 - refinedRelDisp * 0.5;
    float rightX = clamp(0.5 + rightInner * 0.5, 0.501, 0.999);

    vec3 colorL = texture(u_sbsTexture, vec2(leftX, eyeUV.y)).rgb;
    vec3 colorR = texture(u_sbsTexture, vec2(rightX, eyeUV.y)).rgb;

    // Photometric agreement and exposure compensation
    float photoDiff = length(colorL - colorR);
    float confidence = 1.0 - smoothstep(0.06, 0.22, photoDiff);

    float lumaL = getLuma(colorL);
    float lumaR = getLuma(colorR);
    vec3 balancedR = colorR * (lumaL / max(lumaR, 0.05));

    // Depth-warped primary eye with confidence-guided secondary fill to eliminate crossfade ghosting
    float blendWeight = smoothstep(0.15, 0.85, t);
    vec3 primaryColor = (t < 0.5) ? colorL : balancedR;
    vec3 secondaryColor = (t < 0.5) ? balancedR : colorL;
    float fillFactor = (1.0 - confidence) * 0.12 + confidence * (t < 0.5 ? blendWeight : (1.0 - blendWeight));
    vec3 blended = mix(primaryColor, secondaryColor, fillFactor);

    // Apple Vision Spatial Window Glass Border & Portal Depth Glow
    vec2 edgeDist = min(v_uv, 1.0 - v_uv);
    float cornerRadius = 0.035;
    vec2 cornerD = max(vec2(0.0), vec2(cornerRadius) - edgeDist);
    float cornerDist = length(cornerD);
    float portalMask = 1.0 - smoothstep(cornerRadius - 0.006, cornerRadius, cornerDist);

    // Frosted depth-reactive inner rim
    float rimDist = min(edgeDist.x, edgeDist.y);
    float innerRim = smoothstep(0.0, 0.03, rimDist) * (1.0 - smoothstep(0.03, 0.055, rimDist));
    float depthCueGlow = (dNorm - 0.5) * 0.16 * s;
    vec3 frostedRim = vec3(0.68, 0.80, 0.96) * (innerRim * (0.07 + max(0.0, depthCueGlow)));

    // Ambient border feathering
    float frameAlpha = smoothstep(0.0, 0.012, rimDist);
    blended = (blended + frostedRim) * frameAlpha * portalMask;

    outColor = vec4(blended, 1.0);
}
