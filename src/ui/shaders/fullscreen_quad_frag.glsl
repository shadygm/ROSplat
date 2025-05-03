#version 430
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D screen_tex;
void main() {
    FragColor = texture(screen_tex, vUV);
    // Debug: Uncomment to show UV gradient instead
    // FragColor = vec4(vUV, 0.0, 1.0);
}
