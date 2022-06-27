#version 450

#extension GL_NV_mesh_shader : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage: require

#define MAX_VERTICES 64
#define MAX_TRIS 126

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

out taskNV Task {
	uint   subIDs[32];
} OUT;

struct Vertex {
	float posX, posY,
		  colorR, colorG, colorB;
};

layout (binding = 0) readonly buffer MeshInfo {
	float scale;
	uint subdivisions;
	uint8_t p[256];
};

void main() {
	uint ti = gl_LocalInvocationID.x;
	uint mgi = gl_WorkGroupID.x;

	gl_TaskCountNV = (subdivisions/15)*(subdivisions/15);
	//gl_TaskCountNV = 1;
}