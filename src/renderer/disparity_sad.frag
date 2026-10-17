#version 300 es
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
