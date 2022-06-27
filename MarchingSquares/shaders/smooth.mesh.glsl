#version 450

#extension GL_NV_mesh_shader : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#define MAX_VERTICES 256
#define MAX_PRIMITIVES 256
#define PI 3.1415926535897932384626433832795

#error Shading method not implemented

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;

struct CellConfig {
	uint8_t indices[15];
};

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
	vec3 sunlightDir;
	float scale;
	uint subdivisions;
	uint8_t p[512];
	int8_t cellConfigs[256][15];
    vec3 edgePositions[12];
};

layout(binding = 1) buffer Output {
	uint8_t localSpaceTriCounts[LOCAL_SPACE_COUNT];
	uint localSpaceShouldDraw[LOCAL_SPACE_COUNT / 32];
};

layout(push_constant) uniform constants{
	mat4 proj;
	mat4 camera;
	float time;
};

in taskNV Task{
	uint subspaceIndex;
	uint localSpaceIndex[64];
} IN;

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
	ivec3 unitCell = ivec3(pos) & 255;

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

shared float noiseMap[4*4*4];

void main() {
	const uint gtid = gl_GlobalInvocationID.x;
	const uint gid = gl_WorkGroupID.x;
	const uint tid = gl_LocalInvocationIndex.x;
	const float noiseSize = 16.f;
	const uint gridWidth = uint(pow(floor(MAX_VERTICES / 3.f), 1.f / 3));
	const uint gridCount = uint(pow(float(gridWidth), 3));
	vec3 subspacePos = vec3(
		IN.subspaceIndex % (SUBDIVISIONS / 12),
		(IN.subspaceIndex / (SUBDIVISIONS / 12)) % (SUBDIVISIONS / 12),
		(IN.subspaceIndex / uint(pow(SUBDIVISIONS / 12, 2.f)))
	) * 12;
	const uint cellCount = uint(pow(uint(gridWidth - 1), 3.f));
	uint localSpaceId = IN.localSpaceIndex[gid];
	vec3 localSpacePos = subspacePos + vec3(
		localSpaceId % gridWidth,
		(localSpaceId / gridWidth) % gridWidth,
		localSpaceId / uint(pow(gridWidth, 2.f))
	) * (gridWidth - 1);
	float threshold = 0.5;// + 0.2 * sin(time * PI / 4);
	const uint max_per_meshlet_row = 16;

	// Populate noise
	uint noiseMapLoopCount = uint(ceil((4.f * 4 * 4) / gl_WorkGroupSize.x));
	for (uint i = 0; i < noiseMapLoopCount; i++) {
		uint noiseIndex = i * gl_WorkGroupSize.x + tid;
		vec3 noisePos = vec3(
			(noiseIndex % (gridWidth)) + SUBDIVISIONS * time,
			(noiseIndex / (gridWidth)) % uint(gridWidth),
			floor(noiseIndex / pow(gridWidth, 2.f))
		) + localSpacePos;

		// layering and morphing
		float noiseValue = noiseSample(SCALE * noisePos / SUBDIVISIONS);
		//noiseValue = (abs(noisePos.x - 0.f) < 0.1f)
		//	&& (abs(noisePos.y - 0.f) < 0.1f)
		//	&& (abs(noisePos.z - 0.f) < 0.1f) ? 1.f : 0.f;
		if (noiseIndex < 4.f * 4 * 4) {
			noiseMap[noiseIndex] = noiseValue;
		}
	}
	//noiseMap[21] = noiseMap[21] > threshold ? 0.f : 1.f; // local space indicator

	// Populate verts
	/*const uint gridWidth = min(
		uint(floor(pow(floor(MAX_VERTICES / 3.f), 1.f/3))),
		uint(floor(pow(floor(MAX_PRIMITIVES / 3.f), 1.f/3)))
	);*/
	const uint vertexLoopCount = uint(ceil(float(gridCount) / gl_WorkGroupSize.x));
	mat4 projCamera = proj * camera;
	for (uint i = 0; i < vertexLoopCount; i++) {
		uint cellIndex = (i * gl_WorkGroupSize.x) + tid; 

		vec3 cellPos = vec3(
			cellIndex % gridWidth,
			(cellIndex / gridWidth) % (gridWidth),
			cellIndex / uint(pow(gridWidth, 2.f))
		) + localSpacePos;

		vec4 white = (1.f).xxxx;
		/*vec4 outPos = vec4(cellPositions[posIndex], 0.f, 1.f) + cellPos;
		outPos /= vec4(15, 15, 1, 1);
		outPos *= vec4(2.0, 2.0, 1.0, 1.0);
		outPos -= vec4(1.0, 1.0, 0.0, 0.0);*/
		vec3 outPos = (1.f).xxx;

		outPos = noiseSize * (cellPos + vec3(0.f, 0.f, 0.5f)) / SUBDIVISIONS;
		gl_MeshVerticesNV[(cellIndex * 3) + 0].gl_Position = projCamera * vec4(outPos, 1.f);
		v_out[(cellIndex * 3) + 0].color = vec4(1.f, 0.f, 0.f, 1.f);
		
		outPos = noiseSize * (cellPos + vec3(0.5f, 0.f, 0.f)) / SUBDIVISIONS;
		gl_MeshVerticesNV[(cellIndex * 3) + 1].gl_Position = projCamera * vec4(outPos, 1.f);
		v_out[(cellIndex * 3) + 1].color = vec4(0.f, 1.f, 0.f, 1.f);
		
		outPos = noiseSize * (cellPos + vec3(0.f, 0.5f, 0.f)) / SUBDIVISIONS;
		gl_MeshVerticesNV[(cellIndex * 3) + 2].gl_Position = projCamera * vec4(outPos, 1.f);
		v_out[(cellIndex * 3) + 2].color = vec4(0.f, 0.f, 1.f, 1.f);
	}

	// Populate tris
	uint cellLoopCount = uint(ceil(float(cellCount) / gl_WorkGroupSize.x));
	uint primCount = 0;
	for (uint i = 0; i < cellLoopCount; i++) {
		uint cellIndex = (i * gl_WorkGroupSize.x) + tid;
		vec4 cellPos = vec4(
			cellIndex % (gridWidth-1),
			(cellIndex / (gridWidth-1)) % (gridWidth-1),
			cellIndex / uint(pow(gridWidth-1, 2.f)),
			0.f
		);

		uint config = 0;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 0;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 1;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 2;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 3;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 4;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 5;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 6;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 7;
		//config = 1;
		//localSpaceTriCounts[64 * IN.subspaceIndex + gid] = uint8_t(gid);

		int8_t configIndices[15] = cellConfigs[config];
		for (uint j = 0; j < 5; j++) { // per triangle (max 5 tris per cell)
			bool write =
				configIndices[j*3] != -1 &&
					//j*3 + 2 < MAX_VERTICES &&
					//j*3 + 2 < MAX_PRIMITIVES/3 &&
					cellPos.z < (gridWidth-1);
			uvec4 votePrims = subgroupBallot(write);
			uint primCountOffset = subgroupBallotExclusiveBitCount(votePrims);
			
			for (uint k = 0; k < 3; k++) { // per vertex
				uint8_t posIndex = configIndices[j*3 + k];
				uint writeIndex = k + (primCountOffset + primCount) * 3;
				uint writtenIndex = uint(
					((3 * (cellPos.x)) + 
						(3 * (cellPos.y * gridWidth)) +
						(3 * (cellPos.z * pow((gridWidth), 2.f)))) + 
					(posIndex & 0xF) +
					((posIndex & 0x20) != 0 ? 3 : 0) +
					((posIndex & 0x40) != 0 ? (3 * (gridWidth)) : 0) +
					((posIndex & 0x80) != 0 ? (3 * pow((gridWidth), 2.f)) : 0)
					);

				if (write) {
					gl_PrimitiveIndicesNV[writeIndex] = writtenIndex;
				}
			}

			primCount += subgroupBallotBitCount(votePrims);
		}
	}
	//gl_PrimitiveIndicesNV[0] = 0;
	//gl_PrimitiveIndicesNV[1] = 1;
	//gl_PrimitiveIndicesNV[2] = 2;


	gl_PrimitiveCountNV = primCount;
#if DRAW_DATA_MEASURE
	localSpaceTriCounts[64 * IN.subspaceIndex + localSpaceId] = uint8_t(primCount);
#endif
}
