#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec4 pos;
//layout(location = 1) in float pad1;
//layout(location = 2) in float pad2;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = fragColor;
}