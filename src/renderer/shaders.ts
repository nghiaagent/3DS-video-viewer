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
uniform sampler2D u_sbsTexture;
uniform sampler2D u_daDepthTexture;
uniform float u_hasDaDepth;
uniform vec2 u_dispTexelSize;
uniform float u_windowCrop;

void main() {
    // Center reference samples
    float dCenter = texture(u_rawDispTexture, v_uv).r;
    vec3 centerColor = texture(u_sbsTexture, vec2(v_uv.x * 0.5, v_uv.y)).rgb;
    float daCenter = (u_hasDaDepth > 0.5) ? texture(u_daDepthTexture, v_uv).r : dCenter;

    float sumWeights = 0.0;
    float sumFilteredD = 0.0;

    // Continuous branchless Gaussian spatial weight: tightened to twoSigmaSpatialSq = 6.0 for sharp edges
    const float twoSigmaSpatialSq = 6.0;
    // Joint guided filter / color-prior smoothness: tightened to twoSigmaColorSq = 0.015 for sharp object boundaries
    const float twoSigmaColorSq = 0.015;

    for (int dy = -4; dy <= 4; dy++) {
        for (int dx = -4; dx <= 4; dx++) {
            vec2 offsetUV = clamp(v_uv + vec2(float(dx), float(dy)) * u_dispTexelSize, 0.002, 0.998);
            float sampleD = texture(u_rawDispTexture, offsetUV).r;
            vec3 sampleColor = texture(u_sbsTexture, vec2(offsetUV.x * 0.5, offsetUV.y)).rgb;

            float spatialDistSq = float(dx * dx + dy * dy);
            vec3 colorDiff = sampleColor - centerColor;
            float colorDistSq = dot(colorDiff, colorDiff);

            // Smooth isotropic Gaussian spatial weight (sigmaSpatial = 2.8)
            float wSpatial = exp(-spatialDistSq / twoSigmaSpatialSq);
            // Joint Guided Filter / Color-Prior Smoothness term
            float wColor = exp(-colorDistSq / twoSigmaColorSq);
            // Continuous branchless soft foreground dilation weight without diamond steps: exp((sampleD - dCenter) * 1.8)
            float wForeground = exp((sampleD - dCenter) * 1.8);

            float weight = wSpatial * wColor * wForeground;

            if (u_hasDaDepth > 0.5) {
                float sampleDa = texture(u_daDepthTexture, offsetUV).r;
                float daDiff = abs(sampleDa - daCenter);
                float wDa = exp(-daDiff * 10.0);
                weight *= wDa;
            }

            sumFilteredD += sampleD * weight;
            sumWeights += weight;
        }
    }

    float outDepth = sumFilteredD / max(1e-4, sumWeights);
    if (u_hasDaDepth > 0.5) {
        outDepth = mix(outDepth, daCenter, 0.25);
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
uniform vec2 u_dispTexelSize;
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
    float depth;
    if (u_dispTexelSize.x < 0.005) {
        // High-resolution DepthAnything V3: edge-aware foreground-preserving 5-tap filter at ±0.75 texel offsets
        depth = texture(u_dispTexture, sampleUV).r;
        float dCenter = depth;
        vec2 off = u_dispTexelSize * 0.75;
        float dL = texture(u_dispTexture, clamp(sampleUV - vec2(off.x, 0.0), 0.0, 1.0)).r;
        float dR = texture(u_dispTexture, clamp(sampleUV + vec2(off.x, 0.0), 0.0, 1.0)).r;
        float dU = texture(u_dispTexture, clamp(sampleUV + vec2(0.0, off.y), 0.0, 1.0)).r;
        float dD = texture(u_dispTexture, clamp(sampleUV - vec2(0.0, off.y), 0.0, 1.0)).r;
        depth = mix(dCenter, max(dCenter, max(max(dL, dR), max(dU, dD))), 0.35);
    } else {
        // Fallback coarse disparity 5-tap smooth sample
        float dC = texture(u_dispTexture, sampleUV).r;
        float dL = texture(u_dispTexture, clamp(sampleUV - vec2(u_dispTexelSize.x, 0.0), 0.0, 1.0)).r;
        float dR = texture(u_dispTexture, clamp(sampleUV + vec2(u_dispTexelSize.x, 0.0), 0.0, 1.0)).r;
        float dU = texture(u_dispTexture, clamp(sampleUV + vec2(0.0, u_dispTexelSize.y), 0.0, 1.0)).r;
        float dD = texture(u_dispTexture, clamp(sampleUV - vec2(0.0, u_dispTexelSize.y), 0.0, 1.0)).r;
        depth = dC * 0.4 + (dL + dR + dU + dD) * 0.15;
    }
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
uniform sampler2D u_daDepthTexture;
uniform float u_hasDaDepth;
uniform vec2 u_eyeTexelSize;     // Full-res eye texel size e.g. vec2(1.0 / 480.0, 1.0 / 240.0)
uniform int u_maxDisp;           // Search radius e.g. 16
uniform int u_baseDisp;          // Convergence baseline offset e.g. 8 (range [-80, 24])
uniform float u_temporalWeight; // Temporal stabilization weight e.g. 0.80

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

    // Sample 5x5 macro-block window in Left eye (wy: -6..+6 step 3, wx: -8..+8 step 4)
    // aggregates a 17x13 pixel region of the 3DS eye
    float lumaL[25];
    float sumMeanL = 0.0;
    int sIdx = 0;
    for (int wy = -6; wy <= 6; wy += 3) {
        float yOff = clamp(uv.y + float(wy) * u_eyeTexelSize.y, 0.01, 0.99);
        for (int wx = -8; wx <= 8; wx += 4) {
            float lx = clamp((uv.x + float(wx) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498);
            float l = getLuma(texture(u_sbsTexture, vec2(lx, yOff)).rgb);
            lumaL[sIdx] = l;
            sumMeanL += l;
            sIdx++;
        }
    }
    float meanL = sumMeanL / 25.0;

    // Local texture energy / variance across the macro-block
    float gradEnergy = 0.0;
    for (int i = 0; i < 25; i++) {
        gradEnergy += abs(lumaL[i] - meanL);
    }

    // Search d around u_baseDisp in [u_baseDisp - u_maxDisp, u_baseDisp + u_maxDisp]
    int startD = u_baseDisp - u_maxDisp;
    int endD = u_baseDisp + u_maxDisp;
    int numCandidates = endD - startD + 1;

    const int MAX_CANDIDATES = 65;
    float costs[MAX_CANDIDATES];
    for (int i = 0; i < MAX_CANDIDATES; ++i) {
        costs[i] = 1e6;
    }

    float daD = 0.5;
    int predD = u_baseDisp;
    int searchMin = startD;
    int searchMax = endD;

    if (u_hasDaDepth > 0.5) {
        daD = texture(u_daDepthTexture, v_uv).r;
        predD = startD + int(round(daD * float(endD - startD)));
        searchMin = max(startD, predD - 4);
        searchMax = min(endD, predD + 4);
    }

    float bestSAD = 1e6;
    int bestD = (u_hasDaDepth > 0.5) ? predD : u_baseDisp;

    for (int d = searchMin; d <= searchMax; ++d) {
        int candIdx = d - startD;
        if (candIdx < 0 || candIdx >= MAX_CANDIDATES) continue;

        float rightCenterLuma = getLuma(texture(u_sbsTexture, vec2(0.5 + clamp((uv.x - float(d) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498), uv.y)).rgb);

        // Compute Right eye patch samples & mean for ZSAD
        float sumMeanR = 0.0;
        float lumaR[25];
        int rIdx = 0;
        for (int wy = -6; wy <= 6; wy += 3) {
            float yOff = clamp(uv.y + float(wy) * u_eyeTexelSize.y, 0.01, 0.99);
            for (int wx = -8; wx <= 8; wx += 4) {
                float rx = 0.5 + clamp((uv.x + float(wx - d) * u_eyeTexelSize.x) * 0.5, 0.002, 0.498);
                float l = getLuma(texture(u_sbsTexture, vec2(rx, yOff)).rgb);
                lumaR[rIdx] = l;
                sumMeanR += l;
                rIdx++;
            }
        }
        float meanR = sumMeanR / 25.0;
        float lumaDiff = meanL - meanR;

        // Aggregate ZSAD + census across 5x5 window
        float cost = 0.0;
        int pIdx = 0;
        for (int wy = -6; wy <= 6; wy += 3) {
            for (int wx = -8; wx <= 8; wx += 4) {
                float lL = lumaL[pIdx];
                float lR = lumaR[pIdx];
                pIdx++;

                // Center-weighted Gaussian spatial kernel
                float weight = (wx == 0 && wy == 0) ? 2.0 : ((abs(wx) <= 4 && abs(wy) <= 3) ? 1.4 : 0.85);

                // Zero-Mean SAD (ZSAD) + illumination-invariant census term
                float sadTerm = abs((lL - lR) - lumaDiff);
                float censusTerm = abs(sign(lL - centerLumaL) - sign(lR - rightCenterLuma));
                cost += (sadTerm + censusTerm * 0.10) * weight;
            }
        }

        // Prior regularizer:
        // When DepthAnything is active, regularize toward predD with strong penalty
        // When DepthAnything is inactive, use subtle baseline quadratic penalty toward u_baseDisp
        float priorPenalty = (u_hasDaDepth > 0.5)
            ? (0.012 * float((d - predD) * (d - predD)))
            : (0.0015 * float((d - u_baseDisp) * (d - u_baseDisp)));
        cost += priorPenalty;

        costs[candIdx] = cost;

        if (cost < bestSAD) {
            bestSAD = cost;
            bestD = d;
        }
    }

    // Find second best SAD outside the immediate 1-pixel neighborhood of bestD
    int bestIdx = bestD - startD;
    float secondBestSAD = 1e6;
    for (int i = 0; i < MAX_CANDIDATES; ++i) {
        if (i >= numCandidates) break;
        if (abs(i - bestIdx) > 1 && costs[i] < 1e5) {
            if (costs[i] < secondBestSAD) {
                secondBestSAD = costs[i];
            }
        }
    }
    if (secondBestSAD > 1e5) {
        secondBestSAD = bestSAD * 1.5;
    }

    // Subpixel parabolic interpolation using costMinus (bestD - 1) and costPlus (bestD + 1)
    float costMinus = (bestIdx > 0 && costs[bestIdx - 1] < 1e5) ? costs[bestIdx - 1] : bestSAD;
    float costPlus = (bestIdx < numCandidates - 1 && bestIdx + 1 < MAX_CANDIDATES && costs[bestIdx + 1] < 1e5) ? costs[bestIdx + 1] : bestSAD;

    float denom = 2.0 * (costMinus + costPlus - 2.0 * bestSAD);
    float subpixelD = float(bestD) + ((denom > 1e-4) ? clamp((costMinus - costPlus) / denom, -0.5, 0.5) : 0.0);

    // Normalized disparity: 0.5 represents convergence at u_baseDisp
    float normD = clamp((subpixelD - float(startD)) / float(max(1, endD - startD)), 0.0, 1.0);

    // Uniqueness & Texture confidence weighting
    float uniqueness = clamp((secondBestSAD - bestSAD) / max(bestSAD, 0.05), 0.0, 1.0);
    float texConf = smoothstep(0.08, 0.35, gradEnergy);
    float conf = clamp(uniqueness * texConf, 0.0, 1.0);

    if (u_hasDaDepth > 0.5) {
        // SBS photometric warp verification pass
        float daShiftPx = float(startD) + daD * float(endD - startD);
        vec3 colL = texture(u_sbsTexture, vec2(uv.x * 0.5, uv.y)).rgb;
        vec3 colR_da = texture(u_sbsTexture, vec2(0.5 + clamp((uv.x - daShiftPx * u_eyeTexelSize.x) * 0.5, 0.002, 0.498), uv.y)).rgb;
        float errDA = length(colL - colR_da);

        vec3 colR_best = texture(u_sbsTexture, vec2(0.5 + clamp((uv.x - subpixelD * u_eyeTexelSize.x) * 0.5, 0.002, 0.498), uv.y)).rgb;
        float errBest = length(colL - colR_best);

        // Only blend in stereo deviation where errBest < errDA - 0.03 (warp verification)
        float sbsImprovement = smoothstep(0.02, 0.10, errDA - errBest);
        normD = mix(daD, normD, 0.35 * sbsImprovement);
    } else {
        // Ambiguous or flat regions smoothly revert to 0.5 convergence plane
        normD = mix(0.5, normD, conf);
    }

    // Ping-pong temporal stabilization
    if (u_temporalWeight > 0.01) {
        float prevD = texture(u_prevDispTexture, v_uv).r;
        if (prevD > 0.001 && prevD < 0.999) {
            normD = mix(normD, prevD, clamp(u_temporalWeight, 0.0, 0.85));
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
    // Blends Left Eye and Right Eye using photometric agreement and local stereo disparity
    if (u_renderMode == 0) {
        vec3 colorL = texture(u_sbsTexture, vec2(eyeUV.x * 0.5, eyeUV.y)).rgb;

        float dVal = texture(u_dispTexture, eyeUV).r;
        float baseDispPx = float(u_baseDisp);
        float rightEyeX = clamp(eyeUV.x - (baseDispPx + (dVal - 0.35) * 28.0) * u_eyeTexelSize.x, 0.002, 0.998);
        vec3 colorR = texture(u_sbsTexture, vec2(0.5 + rightEyeX * 0.5, eyeUV.y)).rgb;

        float colorDiff = length(colorL - colorR);
        float stereoAgreement = exp(-colorDiff * colorDiff * 18.0);
        float rightBlend = smoothstep(0.35, 0.85, u_viewShift) * stereoAgreement * 0.65;
        vec3 stereoColor = mix(colorL, colorR, rightBlend);

        // Subtle rounded corner portal mask (cornerRadius = 0.025)
        vec2 edgeDist = min(v_uv, 1.0 - v_uv);
        float cornerRadius = 0.025;
        vec2 cornerD = max(vec2(0.0), vec2(cornerRadius) - edgeDist);
        float cornerDist = length(cornerD);
        float portalMask = 1.0 - smoothstep(cornerRadius - 0.005, cornerRadius, cornerDist);

        // Ambient border feathering
        float rimDist = min(edgeDist.x, edgeDist.y);
        float frameAlpha = smoothstep(0.0, 0.008, rimDist);

        outColor = vec4(stereoColor * portalMask * frameAlpha, 1.0);
        return;
    }

    // Mode 1: Disparity Heatmap
    if (u_renderMode == 1) {
        float dVal;
        if (u_dispTexelSize.x < 0.005) {
            // Full-resolution DepthAnything V3: preserve razor-sharp edges without blur
            dVal = texture(u_dispTexture, eyeUV).r;
        } else {
            // Fallback coarse disparity: 5-tap smooth sample
            float dC = texture(u_dispTexture, eyeUV).r;
            float dL = texture(u_dispTexture, clamp(eyeUV - vec2(u_dispTexelSize.x, 0.0), 0.0, 1.0)).r;
            float dR = texture(u_dispTexture, clamp(eyeUV + vec2(u_dispTexelSize.x, 0.0), 0.0, 1.0)).r;
            float dU = texture(u_dispTexture, clamp(eyeUV + vec2(0.0, u_dispTexelSize.y), 0.0, 1.0)).r;
            float dD = texture(u_dispTexture, clamp(eyeUV - vec2(0.0, u_dispTexelSize.y), 0.0, 1.0)).r;
            dVal = dC * 0.4 + (dL + dR + dU + dD) * 0.15;
        }
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
