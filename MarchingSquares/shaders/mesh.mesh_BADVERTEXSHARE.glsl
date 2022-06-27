#version 450

#extension GL_NV_mesh_shader : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#define MAX_VERTICES 256
#define MAX_PRIMITIVES 450
//#define MAX_VERTICES 128
//#define CELL_COUNT 

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;

struct CellConfig {
	uint8_t indices[9];
};
/*
const uint8_t squareConfigurations[16][9] = uint8_t[][](
uint8_t[](0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF),
uint8_t[](0, 1, 3, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF), // 0001
uint8_t[](1, 2, 4, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF), // 0010
uint8_t[](0, 2, 4, 4, 3, 0, 0xFF, 0xFF, 0xFF), // 0011
uint8_t[](4, 7, 6, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF), // 0100
uint8_t[](0, 1, 3, 4, 7, 6, 0xFF, 0xFF, 0xFF), // 0101
uint8_t[](1, 2, 7, 1, 7, 6, 0xFF, 0xFF, 0xFF), // 0110
uint8_t[](0, 2, 7, 0, 7, 6, 0, 6, 3), // 0111
uint8_t[](3, 6, 5, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF), // 1000
uint8_t[](0, 1, 6, 0, 6, 5, 0xFF, 0xFF, 0xFF), // 1001
uint8_t[](1, 2, 4, 3, 6, 5, 0xFF, 0xFF, 0xFF), // 1010
uint8_t[](0, 2, 5, 2, 4, 6, 5, 2, 6), // 1011
uint8_t[](0, 2, 4, 3, 0, 4, 0xFF, 0xFF, 0xFF), // 1100
uint8_t[](0, 7, 5, 0, 1, 4, 7, 0, 4), // 1101
uint8_t[](2, 7, 5, 2, 5, 3, 1, 2, 3), // 1110
uint8_t[](0, 2, 5, 7, 5, 2, 0xFF, 0xFF, 0xFF)
);
*/
/*
const uint8_t squareConfigurationsGlobal[16][9] = uint8_t[][](
uint8_t[](uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)),
uint8_t[](uint8_t(0), uint8_t(1), uint8_t(3), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 0001
uint8_t[](uint8_t(1), uint8_t(2), uint8_t(4), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 0010
uint8_t[](uint8_t(0), uint8_t(2), uint8_t(4), uint8_t(4), uint8_t(3), uint8_t(0), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 0011
uint8_t[](uint8_t(4), uint8_t(7), uint8_t(6), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 0100
uint8_t[](uint8_t(0), uint8_t(1), uint8_t(3), uint8_t(4), uint8_t(7), uint8_t(6), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 0101
uint8_t[](uint8_t(1), uint8_t(2), uint8_t(7), uint8_t(1), uint8_t(7), uint8_t(6), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 0110
uint8_t[](uint8_t(0), uint8_t(2), uint8_t(7), uint8_t(0), uint8_t(7), uint8_t(6), uint8_t(0), uint8_t(6), uint8_t(3)), // 0111
uint8_t[](uint8_t(3), uint8_t(6), uint8_t(5), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 1000
uint8_t[](uint8_t(0), uint8_t(1), uint8_t(6), uint8_t(0), uint8_t(6), uint8_t(5), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 1001
uint8_t[](uint8_t(1), uint8_t(2), uint8_t(4), uint8_t(3), uint8_t(6), uint8_t(5), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 1010
uint8_t[](uint8_t(0), uint8_t(2), uint8_t(5), uint8_t(2), uint8_t(4), uint8_t(6), uint8_t(5), uint8_t(2), uint8_t(6)), // 1011
uint8_t[](uint8_t(0), uint8_t(2), uint8_t(4), uint8_t(3), uint8_t(0), uint8_t(4), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF)), // 1100
uint8_t[](uint8_t(0), uint8_t(7), uint8_t(5), uint8_t(0), uint8_t(1), uint8_t(4), uint8_t(7), uint8_t(0), uint8_t(4)), // 1101
uint8_t[](uint8_t(2), uint8_t(7), uint8_t(5), uint8_t(2), uint8_t(5), uint8_t(3), uint8_t(1), uint8_t(2), uint8_t(3)), // 1110
uint8_t[](uint8_t(0), uint8_t(2), uint8_t(5), uint8_t(7), uint8_t(5), uint8_t(2), uint8_t(0xFF), uint8_t(0xFF), uint8_t(0xFF))
);*/

const vec2 cellPositions[8] = vec2[](
	vec2(0.f, 0.f),
	vec2(0.5f, 0.f),
	vec2(1.f, 0.f),
	vec2(0.f, 0.5f),
	vec2(1.f, 0.5f),
	vec2(0.f, 1.f),
	vec2(0.5f, 1.f),
	vec2(1.f, 1.f)
);

struct Vertex {
	float posX, posY,
		  colorR, colorG, colorB;
};

layout (location = 0) out Interpolant
{
	vec4 color;
} v_out[];

layout (binding = 0) readonly buffer MeshInfo {
	float scale;
	uint subdivisions;
	uint8_t p[512];
	uint8_t squareConfigurations[16][9];
};

layout(push_constant) uniform constants {
	float time;
};

in taskNV block {
	uint something;
};

vec3 fade(vec3 t) {
	return t * t * t * (t * (t * 6 - 15) + 10);
}

float grad(int hash, float x, float y, float z) {
	int h = hash & 15;
	// Convert lower 4 bits of hash into 12 gradient directions
	float u = h < 8 ? x : y,
		v = h < 4 ? y : h == 12 || h == 14 ? x : z;
	return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

float noiseSample(vec3 pos) {
	// Find the unit cube that contains the point
	// NOTE can I vectorize the operation or do I do it manually
	ivec3 unitCell = ivec3(floor(pos)) & 255;

	// Find relative x, y,z of point in cube
	pos -= floor(pos);

	// Compute fade curves for each of x, y, z
	vec3 faded = fade(pos);

	// Hash coordinates of the 8 cube corners
	int A = int(p[unitCell.x]) + unitCell.y;
	int AA = int(p[A]) + unitCell.z;
	int AB = int(p[A + 1]) + unitCell.z;
	int B = int(p[unitCell.x + 1]) + unitCell.y;
	int BA = int(p[B]) + unitCell.z;
	int BB = int(p[B + 1]) + unitCell.z;

	// Add blended results from 8 corners of cube
	float res = mix(
		mix(
			mix(
				grad(int(p[AA]), pos.x, pos.y, pos.z),
				grad(int(p[BA]), pos.x - 1, pos.y, pos.z),
				faded.x
			),
			mix(
				grad(int(p[AB]), pos.x, pos.y - 1, pos.z),
				grad(int(p[BB]), pos.x - 1, pos.y - 1, pos.z),
				faded.x
			),
			faded.y
		),
		mix(
			mix(
				grad(int(p[AA + 1]), pos.x, pos.y, pos.z - 1),
				grad(int(p[BA + 1]), pos.x - 1, pos.y, pos.z - 1),
				faded.x
			),
			mix(
				grad(int(p[AB + 1]), pos.x, pos.y - 1, pos.z - 1),
				grad(int(p[BB + 1]), pos.x - 1, pos.y - 1, pos.z - 1),
				faded.x
			),
			faded.y
		),
		faded.z
	);
	return (res + 1.0) / 2.0;
}

shared float noiseMap[16*16];

void main() {
	uint gtid = gl_GlobalInvocationID.x;
	uint gid = gl_WorkGroupID.x;
	uint tid = gl_LocalInvocationIndex.x;
	uint max_per_meshlet_row = 16;

	uint noiseMapLoopCount = uint(ceil(256.f / gl_WorkGroupSize.x));
	for (uint i = 0; i < noiseMapLoopCount; i++) {
		uint noiseIndex = i * gl_WorkGroupSize.x + tid;
		float noiseValue = noiseSample(vec3(noiseIndex % 16 + time, noiseIndex / 16, 0.f));
		noiseValue = noiseIndex % 2 == 0 && noiseIndex/16 % 2 == 0 ? 0.f : 1.f;
		if (noiseIndex < 16 * 16) {
			noiseMap[noiseIndex] = noiseValue;
		}
	}

	// Populate verts
	uint gridWidth = sqrt(MAX_VERTICES / 3);
	uint cellCount = pow(gridWidth, 2);
	uint vertexLoopCount = cellCount / gl_WorkGroupSize.x;
	for (uint i = 0; i < vertexLoopCount; i++) {
		uint cellIndex = vertexLoopCount * gl_WorkGroupSize.x + tid;
		vec4 cellPos = vec4(
			cellIndex % gridWidth,
			cellIndex / gridWidth,
			0.f,
			0.f
		);

		gl_MeshVerticesNV[cellIndex * 3 + 0].gl_Position = cellPos
			+ vec4(0.f, 0.f, 0.f, 1.f);
		gl_MeshVerticesNV[cellIndex * 3 + 1].gl_Position = cellPos
			+ vec4(0.f, 0.5f, 0.f, 1.f);
		gl_MeshVerticesNV[cellIndex * 3 + 2].gl_Position = cellPos
			+ vec4(0.f, 0.f, 0.5f, 1.f);
	}

	// Populate tris
	cellCount = 15 * 15;
	uint cellLoopCount = cellCount / gl_WorkGroupSize.x;
	uint primCount = 0;
	for (uint i = 0; i < 1; i++) {
		uint cellIndex = (i * gl_WorkGroupSize.x) + tid;
		vec4 cellPos = vec4(
			cellIndex % 15,
			cellIndex / 15,
			0.f,
			0.f
		);

		uint config = 0;
		config |= (noiseMap[int(cellPos.x) + (int(cellPos.y) * 16)] > 0.5f ? 1 : 0) << 0;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y) * 16)] > 0.5f ? 1 : 0) << 1;
		config |= (noiseMap[int(cellPos.x) + 1 + ((int(cellPos.y) + 1) * 16)] > 0.5f ? 1 : 0) << 2;
		config |= (noiseMap[int(cellPos.x) + ((int(cellPos.y) + 1) * 16)] > 0.5f ? 1 : 0) << 3;
		//config = 0xF;
		uint8_t configIndices[9] = squareConfigurations[config];
		for (uint j = 0; j < 3; j++) { // per triangle (max 3 tris per cell)
			bool write = 
				configIndices[j*3 + 2] != 0xFF &&
				j*3 + 2 < MAX_VERTICES &&
				j*3 + 2 < MAX_PRIMITIVES/3;
			uvec4 votePrims = subgroupBallot(write);
			
			for (uint k = 0; k < 3; k++) { // per vertex
				uint posIndex = uint(configIndices[j*3 + k]);
				uint writeIndex = k + subgroupBallotExclusiveBitCount(votePrims) * 3 + primCount*3;
				vec4 outPos = vec4(cellPositions[posIndex], 0.f, 1.f) + cellPos;
				outPos /= vec4(15, 15, 1, 1);
				outPos *= vec4(2.0, 2.0, 1.0, 1.0);
				outPos -= vec4(1.0, 1.0, 0.0, 0.0);

				if (write) {
					gl_MeshVerticesNV[writeIndex].gl_Position = outPos;
					v_out[writeIndex].color = vec4(
						1.f,
						1.f,
						1.f,
						1.f
					);
					gl_PrimitiveIndicesNV[writeIndex] = writeIndex;
				}
			}
			primCount += subgroupBallotBitCount(votePrims);
		}
	}
	gl_PrimitiveCountNV = primCount;
	/*

	for (uint i = tid; i < MAX_VERTICES; i += gl_WorkGroupSize.x) {
		vec4 pos = vec4(
			(float(i % max_per_meshlet_row) / subdivisions) + (float(gid % (subdivisions / max_per_meshlet_row)) * 15 / subdivisions),
			(float(i / max_per_meshlet_row) / subdivisions) + (float(gid / (subdivisions / max_per_meshlet_row)) * 15 / subdivisions),
			0.0, 1.0);
		pos *= vec4(2.0, 2.0, 1.0, 1.0);
		pos -= vec4(1.0, 1.0, 0.0, 0.0);

		vec4 offset = vec4(
			noiseSample(scale * vec3(scale * pos.x + time/4, scale * pos.y + time/4, time/4 + 0.0f)) * 2 - 1,
			noiseSample(scale * vec3(scale * pos.x + time/4, scale * pos.y + time/4, time/4 + 1.0f)) * 2 - 1,
			0.f,
			0.f
		);
		pos += (1.f/16) * offset;
		gl_MeshVerticesNV[i].gl_Position = pos;
		v_out[i].color = vec4(
			1.f,
			1.f,
			1.f,
			1.f
		);
	}

	// TODO get a constant instead of magic number
	for (uint i = tid; i < 225; i += gl_WorkGroupSize.x) {
		gl_PrimitiveIndicesNV[0 + i * 6] = 0 + i + (i / 15);
		gl_PrimitiveIndicesNV[1 + i * 6] = 1 + i + (i / 15);
		gl_PrimitiveIndicesNV[2 + i * 6] = 16 + i + (i / 15);
		gl_PrimitiveIndicesNV[3 + i * 6] = 1 + i + (i / 15);
		gl_PrimitiveIndicesNV[4 + i * 6] = 17 + i + (i / 15);
		gl_PrimitiveIndicesNV[5 + i * 6] = 16 + i + (i / 15);
	}

	gl_PrimitiveCountNV = 450;  // pow(ceil(sqrt(max_vertices)) - 1, 2)	
	*/
}
