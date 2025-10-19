// Neuron Glow Shader
uniform vec2 resolution;
uniform float time;

void main() {
    vec2 uv = gl_FragCoord.xy / resolution;
    float glow = sin(time * 2.0) * 0.5 + 0.5;
    gl_FragColor = vec4(0.48, 0.74, 0.54, glow * 0.8);
}