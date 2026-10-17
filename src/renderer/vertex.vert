#version 300 es
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
