#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_sbsTexture;
uniform vec2 u_eyeTexelSize; // vec2(1.0 / eyeWidth, 1.0 / eyeHeight)
uniform int u_maxDisp;       // e.g. 24 or 12

float getLuma(vec3 rgb) {
    return dot(rgb, vec3(0.299, 0.587, 0.114));
}

void main() {
    // Clamp to valid interior region to avoid out-of-bounds and replicate edges seamlessly
    float minX = float(u_maxDisp + 4) * u_eyeTexelSize.x;
    float maxX = 1.0 - 8.0 * u_eyeTexelSize.x;
    float minY = 4.0 * u_eyeTexelSize.y;
    float maxY = 1.0 - 4.0 * u_eyeTexelSize.y;

    vec2 uv = vec2(
        clamp(v_uv.x, minX, maxX),
        clamp(v_uv.y, minY, maxY)
    );

    float bestSAD = 1e6;
    int bestD = -4;

    // 5x5 stepped SAD block matching across d in -4..u_maxDisp
    for (int d = -4; d <= u_maxDisp; ++d) {
        float sad = 0.0;

        for (int wy = -4; wy <= 4; wy += 2) {
            float yOffset = float(wy) * u_eyeTexelSize.y;
            float leftY = uv.y + yOffset;
            float rightY = uv.y + yOffset;

            for (int wx = -4; wx <= 4; wx += 2) {
                float lx = (uv.x + float(wx) * u_eyeTexelSize.x) * 0.5;
                float rx = 0.5 + (uv.x + float(wx - d) * u_eyeTexelSize.x) * 0.5;

                float lumaL = getLuma(texture(u_sbsTexture, vec2(lx, leftY)).rgb);
                float lumaR = getLuma(texture(u_sbsTexture, vec2(rx, rightY)).rgb);

                sad += abs(lumaL - lumaR);
            }
        }

        if (sad < bestSAD) {
            bestSAD = sad;
            bestD = d;
        }
    }

    // Normalize bestD from [-4, u_maxDisp] into [0.0, 1.0]
    float normD = clamp(float(bestD + 4) / float(u_maxDisp + 4), 0.0, 1.0);
    outColor = vec4(normD, 0.0, 0.0, 1.0);
}
