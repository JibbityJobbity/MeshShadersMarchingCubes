#version 450

#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

layout(location = 0) in Interpolant {
	flat vec4 fragColor;
};
//layout(location = 1) in vec4 pos;
//layout(location = 1) in float pad1;
//layout(location = 2) in float pad2;

layout(location = 0) out vec4 outColor;
// Uncomment for access to sunlightDir
// Or just make it a constant which is probably better
/*
layout (binding = 0) readonly buffer MeshInfo {
	vec3 sunlightDir;
	float scale;
	uint subdivisions;
	uint8_t p[512];
	uint8_t p12[512];
	int8_t cellConfigs[256][15];
    vec4 edgePositions_[12];
};*/

void main() {
	/*const float ambient = 0.5;
	const float diffusion = 0.5;
	const float specular = 0.0;
	const vec3 lightColour = vec3(114, 237, 255) / 255;
	float luminance = 
		ambient +
		max(0.0, (diffusion * dot(fragColor.xyz, sunlightDir)));
    outColor = vec4(lightColour * luminance, 1.0);
	*/
	outColor = fragColor;
}