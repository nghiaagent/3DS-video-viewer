export const VERTEX_SHADER_SOURCE = `#version 300 es
precision highp float;
precision highp int;

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

export const TIEFLING_DEPTH_FILTER_FRAG_SOURCE = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_rawDispTexture;
uniform vec2 u_dispTexelSize;

void main() {
    // 9-tap Gaussian/box neighborhood samples
    float dCenter = texture(u_rawDispTexture, v_uv).r;
    float dL  = texture(u_rawDispTexture, v_uv - vec2(u_dispTexelSize.x, 0.0)).r;
    float dR  = texture(u_rawDispTexture, v_uv + vec2(u_dispTexelSize.x, 0.0)).r;
    float dU  = texture(u_rawDispTexture, v_uv + vec2(0.0, u_dispTexelSize.y)).r;
    float dD  = texture(u_rawDispTexture, v_uv - vec2(0.0, u_dispTexelSize.y)).r;
    float dUL = texture(u_rawDispTexture, v_uv + vec2(-u_dispTexelSize.x, u_dispTexelSize.y)).r;
    float dUR = texture(u_rawDispTexture, v_uv + vec2(u_dispTexelSize.x, u_dispTexelSize.y)).r;
    float dDL = texture(u_rawDispTexture, v_uv + vec2(-u_dispTexelSize.x, -u_dispTexelSize.y)).r;
    float dDR = texture(u_rawDispTexture, v_uv + vec2(u_dispTexelSize.x, -u_dispTexelSize.y)).r;

    float avg9 = (dCenter * 4.0 + (dL + dR + dU + dD) * 2.0 + (dUL + dUR + dDL + dDR) * 1.0) / 16.0;

    // Foreground max-dilation neighborhood search (radius 3)
    float maxDilated = dCenter;
    for (int dy = -3; dy <= 3; dy++) {
        for (int dx = -3; dx <= 3; dx++) {
            if (dx * dx + dy * dy <= 10) {
                float sampleD = texture(u_rawDispTexture, v_uv + vec2(float(dx), float(dy)) * u_dispTexelSize).r;
                maxDilated = max(maxDilated, sampleD);
            }
        }
    }

    // Steep-gradient detection and silhouette dilation
    float gradientMag = abs(dR - dL) + abs(dU - dD);
    float smoothedDilated = mix(maxDilated, avg9, 0.35);

    float outDepth = avg9;
    if (gradientMag > 0.08) {
        outDepth = smoothedDilated;
    }

    outColor = vec4(outDepth, 0.0, 0.0, 1.0);
}
`;

export const TIEFLING_MESH_VERTEX_SHADER_SOURCE = `#version 300 es
precision highp float;
precision highp int;

in vec2 a_uv;
out vec2 v_uv;

uniform sampler2D u_dispTexture;
uniform float u_viewShift;
uniform float u_verticalTilt;
uniform float u_meshDepth;
uniform float u_focus;
uniform float u_sensitivity;
uniform float u_windowCrop;
uniform float u_flipY;
uniform int u_renderMode;

void main() {
    vec2 uv = a_uv;
    if (u_flipY > 0.5) {
        uv.y = 1.0 - uv.y;
    }
    v_uv = uv;

    if (u_renderMode != 0) {
        gl_Position = vec4(a_uv * 2.0 - 1.0, 0.0, 1.0);
        return;
    }

    // Tiefling 3D Parallax Mesh mode
    vec2 sampleUV = clamp(vec2(u_windowCrop + uv.x * (1.0 - 2.0 * u_windowCrop), uv.y), 0.0, 1.0);
    float depth = texture(u_dispTexture, sampleUV).r;
    float z = depth * u_meshDepth;

    // Tiefling exact perspective compensation (cameraZ = 4.0)
    float cameraZ = 4.0;
    float scaleFactor = (cameraZ - z) / cameraZ;

    vec2 basePos = a_uv * 2.0 - 1.0;
    vec3 pos = vec3(basePos * scaleFactor, z);

    // Edge factor smoothly tapers displacement near outer border
    vec2 edgeDist = min(a_uv, 1.0 - a_uv);
    float edgeFactor = smoothstep(0.0, 0.06, min(edgeDist.x, edgeDist.y));

    // Focal-plane rotational parallax displacement
    vec2 mouseDelta = vec2((u_viewShift - 0.5) * 2.0, u_verticalTilt);
    float actualDepth = depth;
    vec2 rotate = mouseDelta * u_sensitivity * (1.0 - u_focus) * (actualDepth - u_focus) * vec2(-1.0, 1.0);
    pos.xy += rotate * edgeFactor;

    // Perspective projection from cameraZ = 4.0
    float perspDiv = (cameraZ - pos.z) / cameraZ;
    gl_Position = vec4(pos.xy / max(0.2, perspDiv), -pos.z * 0.5, 1.0);
}
`;

export const DISPARITY_SAD_FRAG_SOURCE = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform sampler2D u_prevDispTexture;
uniform vec2 u_eyeTexelSize;     // Full-res eye texel size e.g. vec2(1.0 / 480.0, 1.0 / 240.0)
uniform int u_maxDisp;           // Search radius e.g. 16 or 20
uniform int u_baseDisp;          // Convergence baseline offset e.g. 8 (range [-80, 24])
uniform float u_temporalWeight; // Temporal stabilization weight e.g. 0.70

float getLuma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    // Keep interior region safely away from extreme borders
    const float minX = 0.03;
    const float maxX = 0.97;
    const float minY = 0.03;
    const float maxY = 0.97;

    vec2 uv = vec2(
        clamp(v_uv.x, minX, maxX),
        clamp(v_uv.y, minY, maxY)
    );

    // Reference center samples for census transform
    vec3 centerColorL = texture(u_sbsTexture, vec2(uv.x * 0.5, uv.y)).rgb;
    float centerLumaL = getLuma(centerColorL);

    // Compute Left 3x3 center mean luminance for ZSAD
    float sumMeanL = 0.0;
    for (int cy = -1; cy <= 1; cy++) {
        float yOff = clamp(uv.y + float(cy) * u_eyeTexelSize.y, 0.01, 0.99);
        for (int cx = -1; cx <= 1; cx++) {
            float lx = clamp((uv.x + float(cx) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498);
            sumMeanL += getLuma(texture(u_sbsTexture, vec2(lx, yOff)).rgb);
        }
    }
    float meanL = sumMeanL / 9.0;

    float bestSAD = 1e6;
    int bestD = u_baseDisp;

    // Search d around u_baseDisp in [u_baseDisp - u_maxDisp, u_baseDisp + u_maxDisp]
    int startD = u_baseDisp - u_maxDisp;
    int endD = u_baseDisp + u_maxDisp;

    for (int d = startD; d <= endD; ++d) {
        float cost = 0.0;
        float rightCenterLuma = getLuma(texture(u_sbsTexture, vec2(0.5 + clamp((uv.x - float(d) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498), uv.y)).rgb);

        // Compute Right 3x3 center mean luminance at offset d for ZSAD
        float sumMeanR = 0.0;
        for (int cy = -1; cy <= 1; cy++) {
            float yOff = clamp(uv.y + float(cy) * u_eyeTexelSize.y, 0.01, 0.99);
            for (int cx = -1; cx <= 1; cx++) {
                float rx = 0.5 + clamp((uv.x + float(cx - d) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498);
                sumMeanR += getLuma(texture(u_sbsTexture, vec2(rx, yOff)).rgb);
            }
        }
        float meanR = sumMeanR / 9.0;
        float lumaDiff = meanL - meanR;

        // 9x7 patch matching sampling full-res u_sbsTexture
        for (int wy = -3; wy <= 3; wy += 2) {
            float yOffset = clamp(uv.y + float(wy) * u_eyeTexelSize.y, 0.01, 0.99);

            for (int wx = -4; wx <= 4; wx += 2) {
                float lx = clamp((uv.x + float(wx) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498);
                float rx = 0.5 + clamp((uv.x + float(wx - d) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498);

                vec3 rgbL = texture(u_sbsTexture, vec2(lx, yOffset)).rgb;
                vec3 rgbR = texture(u_sbsTexture, vec2(rx, yOffset)).rgb;

                float lL = getLuma(rgbL);
                float lR = getLuma(rgbR);

                // Center-weighted kernel
                float weight = (wx == 0 && wy == 0) ? 2.0 : ((abs(wx) <= 2 && abs(wy) <= 1) ? 1.4 : 0.85);

                // Zero-Mean SAD (ZSAD) + illumination-invariant census term
                float sadTerm = abs((lL - lR) - lumaDiff);
                float censusTerm = abs(sign(lL - centerLumaL) - sign(lR - rightCenterLuma));
                cost += (sadTerm + censusTerm * 0.10) * weight;
            }
        }

        if (cost < bestSAD) {
            bestSAD = cost;
            bestD = d;
        }
    }

    // Normalized disparity: 0.5 represents convergence at u_baseDisp
    float normD = clamp(float(bestD - startD) / float(max(1, endD - startD)), 0.0, 1.0);

    // Ping-pong temporal stabilization
    if (u_temporalWeight > 0.01) {
        float prevD = texture(u_prevDispTexture, v_uv).r;
        if (prevD > 0.001 && prevD < 0.999) {
            normD = mix(normD, prevD, clamp(u_temporalWeight, 0.0, 0.75));
        }
    }

    outColor = vec4(normD, 0.0, 0.0, 1.0);
}
`;

export const PERSPECTIVE_WINDOW_FRAG_SOURCE = `#version 300 es
precision highp float;
precision highp int;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform sampler2D u_dispTexture;

uniform float u_viewShift;       // 0.0 (Left) .. 1.0 (Right), default 0.5
uniform float u_maxDisparityUV;  // e.g. 16.0 / 480.0 = 0.033
uniform float u_windowCrop;      // default 0.0833
uniform vec2 u_eyeTexelSize;     // default vec2(1.0 / 480.0, 1.0 / 240.0)
uniform vec2 u_dispTexelSize;    // disparity FBO texel size e.g. vec2(1.0 / 160.0, 1.0 / 96.0)
uniform float u_zeroDispNorm;    // default 0.5
uniform int u_baseDisp;          // global horizontal baseline convergence offset (pixels)
uniform int u_renderMode;        // 0: Parallax 3D Mesh, 1: Heatmap, 2: Left, 3: Right, 4: SBS, 5: Anaglyph

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

    // Active 400x240 (5:3) 3DS window inside 480x240 frame
    float activeWidth = 1.0 - (2.0 * u_windowCrop);
    vec2 eyeUV = vec2(
        clamp(u_windowCrop + v_uv.x * activeWidth, 0.002, 0.998),
        clamp(v_uv.y, 0.002, 0.998)
    );

    // Mode 0: Tiefling 3D Parallax Mesh mode
    // Samples ONLY the single converged Left Eye color stream (clean, ghost-free, jump-free)
    if (u_renderMode == 0) {
        vec3 colorL = texture(u_sbsTexture, vec2(eyeUV.x * 0.5, eyeUV.y)).rgb;

        // Subtle rounded corner portal mask (cornerRadius = 0.025)
        vec2 edgeDist = min(v_uv, 1.0 - v_uv);
        float cornerRadius = 0.025;
        vec2 cornerD = max(vec2(0.0), vec2(cornerRadius) - edgeDist);
        float cornerDist = length(cornerD);
        float portalMask = 1.0 - smoothstep(cornerRadius - 0.005, cornerRadius, cornerDist);

        // Ambient border feathering
        float rimDist = min(edgeDist.x, edgeDist.y);
        float frameAlpha = smoothstep(0.0, 0.008, rimDist);

        outColor = vec4(colorL * portalMask * frameAlpha, 1.0);
        return;
    }

    // Mode 1: Disparity Heatmap
    if (u_renderMode == 1) {
        float dVal = texture(u_dispTexture, eyeUV).r;
        vec3 heatmap = turboColormap(dVal);
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

    outColor = vec4(0.0, 0.0, 0.0, 1.0);
}
`;
