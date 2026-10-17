#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform sampler2D u_prevDispTexture;
uniform vec2 u_eyeTexelSize;     // Full-res eye texel size e.g. vec2(1.0 / 480.0, 1.0 / 240.0)
uniform int u_maxDisp;           // Search radius e.g. 16 or 20
uniform int u_baseDisp;          // Convergence baseline offset e.g. 8 (range [-4, 16])
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

    float bestSAD = 1e6;
    int bestD = u_baseDisp;

    // Search d around u_baseDisp in [u_baseDisp - u_maxDisp, u_baseDisp + u_maxDisp]
    int startD = u_baseDisp - u_maxDisp;
    int endD = u_baseDisp + u_maxDisp;

    for (int d = startD; d <= endD; ++d) {
        float cost = 0.0;
        float rightCenterLuma = getLuma(texture(u_sbsTexture, vec2(0.5 + (uv.x - float(d) * u_eyeTexelSize.x) * 0.5, uv.y)).rgb);

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

                // Luma SAD + illumination-invariant census term
                float sadTerm = abs(lL - lR);
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
