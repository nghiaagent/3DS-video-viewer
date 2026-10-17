export const VERTEX_SHADER_SOURCE = `#version 300 es
precision highp float;

in vec2 a_position;
out vec2 v_uv;
uniform float u_flipY;

void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    vec2 uv = a_position * 0.5 + 0.5;
    if (u_flipY > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    v_uv = uv;
}
`;

export const DISPARITY_SAD_FRAG_SOURCE = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform sampler2D u_prevDispTexture;
uniform vec2 u_eyeTexelSize;     // vec2(1.0 / eyeWidth, 1.0 / eyeHeight)
uniform int u_maxDisp;           // half search radius e.g. 24 or 16
uniform int u_baseDisp;          // global convergence baseline offset e.g. -72 or 0
uniform float u_temporalWeight; // temporal stabilization e.g. 0.75, or 0.0 on first frame

float getLuma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    float maxD = float(u_baseDisp + u_maxDisp);
    float minD = float(u_baseDisp - u_maxDisp);

    // Guard bounds so left and right sampling stays safely within eye bounds
    float minX = clamp(max(6.0, maxD + 6.0) * u_eyeTexelSize.x, 0.02, 0.48);
    float maxX = clamp(1.0 - max(6.0, 6.0 - minD) * u_eyeTexelSize.x, 0.52, 0.98);
    float minY = 4.0 * u_eyeTexelSize.y;
    float maxY = 1.0 - 4.0 * u_eyeTexelSize.y;

    vec2 uv = vec2(
        clamp(v_uv.x, minX, maxX),
        clamp(v_uv.y, minY, maxY)
    );

    // Reference center samples for census transform
    vec3 centerColorL = texture(u_sbsTexture, vec2(uv.x * 0.5, uv.y)).rgb;
    float centerLumaL = getLuma(centerColorL);

    float bestSAD = 1e6;
    int bestD = u_baseDisp;

    // Search d around u_baseDisp in [u_baseDisp - u_maxDisp, u_baseDisp + u_maxDisp]
    int startD = u_baseDisp - u_maxDisp;
    int endD = u_baseDisp + u_maxDisp;

    for (int d = startD; d <= endD; ++d) {
        float cost = 0.0;
        float rightCenterLuma = getLuma(texture(u_sbsTexture, vec2(0.5 + (uv.x - float(d) * u_eyeTexelSize.x) * 0.5, uv.y)).rgb);

        // 7x5 horizontally-weighted census + luma SAD
        for (int wy = -2; wy <= 2; wy += 2) {
            float yOffset = uv.y + float(wy) * u_eyeTexelSize.y;

            for (int wx = -3; wx <= 3; wx += 1) {
                float lx = (uv.x + float(wx) * u_eyeTexelSize.x) * 0.5;
                float rx = 0.5 + (uv.x + float(wx - d) * u_eyeTexelSize.x) * 0.5;

                vec3 rgbL = texture(u_sbsTexture, vec2(lx, yOffset)).rgb;
                vec3 rgbR = texture(u_sbsTexture, vec2(rx, yOffset)).rgb;

                float lL = getLuma(rgbL);
                float lR = getLuma(rgbR);

                // Horizontal spatial kernel weighting
                float weight = (wx == 0 && wy == 0) ? 1.8 : ((abs(wx) <= 1 && wy == 0) ? 1.4 : 0.85);

                // Luma SAD + census sign correlation
                float sadTerm = abs(lL - lR);
                float censusTerm = abs(sign(lL - centerLumaL) - sign(lR - rightCenterLuma));
                cost += (sadTerm + censusTerm * 0.12) * weight;
            }
        }

        if (cost < bestSAD) {
            bestSAD = cost;
            bestD = d;
        }
    }

    // Normalized disparity: 0.5 represents exact convergence at u_baseDisp
    float normD = clamp(float(bestD - startD) / float(max(1, endD - startD)), 0.0, 1.0);

    // Temporal stabilization ping-pong blend
    if (u_temporalWeight > 0.01) {
        float prevD = texture(u_prevDispTexture, v_uv).r;
        if (prevD > 0.001 && prevD < 0.999) {
            normD = mix(normD, prevD, clamp(u_temporalWeight, 0.0, 0.82));
        }
    }

    outColor = vec4(normD, 0.0, 0.0, 1.0);
}
`;

export const PERSPECTIVE_WINDOW_FRAG_SOURCE = `#version 300 es
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
`;
