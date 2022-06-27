#version 450

#extension GL_NV_mesh_shader : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require

#define MAX_PRIMITIVES 41//41//41
#define MAX_VERTICES 120//MAX_PRIMITIVES*3//196
#define PI 3.1415926535897932384626433832795
#define CACHE_NOISE 1

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
layout(triangles, max_vertices = MAX_VERTICES, max_primitives = MAX_PRIMITIVES) out;

struct CellConfig {
	uint8_t indices[15];
};

const vec4 edgePositions[] = vec4[](
    // Bottom
    vec4( 0.f , -0.5f, -0.5f, 1.f),
    vec4( 0.5f, -0.5f,  0.f , 1.f),
    vec4( 0.f , -0.5f,  0.5f, 1.f),
    vec4(-0.5f, -0.5f,  0.f , 1.f),
    // Top
    vec4( 0.f ,  0.5f, -0.5f, 1.f),
    vec4( 0.5f,  0.5f,  0.f , 1.f),
    vec4( 0.f ,  0.5f,  0.5f, 1.f),
    vec4(-0.5f,  0.5f,  0.f , 1.f),
    // Middle
    vec4(-0.5f,  0.f , -0.5f, 1.f),
    vec4( 0.5f,  0.f , -0.5f, 1.f),
    vec4( 0.5f,  0.f ,  0.5f, 1.f),
    vec4(-0.5f,  0.f ,  0.5f, 1.f)
);


struct Vertex {
	float posX, posY,
		  colorR, colorG, colorB;
};

layout (location = 0) out Interpolant
{
	flat vec4 color;
	//mat4 somethingElse;
} v_out[];

layout (binding = 0) readonly buffer MeshInfo {
	vec3 sunlightDir;
	float scale;
	uint subdivisions;
	uint8_t p[512];
	uint8_t p12[512];
	int8_t cellConfigs[256][15];
    vec4 edgePositions_[12];
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
	int A =  int(p[unitCell.x]) + unitCell.y;
	int AA = int(p[A]) + unitCell.z;
	int AB = int(p[A + 1]) + unitCell.z;
	int B =  int(p[unitCell.x + 1]) + unitCell.y;
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

//https://weber.itn.liu.se/~stegu/simplexnoise/simplexnoise.pdf
// 3D simplex noise

ivec3 grad3[] = ivec3[](ivec3(1,1,0),ivec3(-1,1,0),ivec3(1,-1,0),ivec3(-1,-1,0),
ivec3(1,0,1),ivec3(-1,0,1),ivec3(1,0,-1),ivec3(-1,0,-1),
ivec3(0,1,1),ivec3(0,-1,1),ivec3(0,1,-1),ivec3(0,-1,-1));

float simplexNoise(vec3 pos) {
	float n0, n1, n2, n3; // Noise contributions from the four corners
    // Skew the input space to determine which simplex cell we're in
	const float F3 = 1.0/3.0;
    float s = (pos.x+pos.y+pos.z)*F3; // Very nice and simple skew factor for 3D
    uint i = uint(floor(pos.x+s));
    uint j = uint(floor(pos.y+s));
    uint k = uint(floor(pos.z+s));
	const float G3 = 1.0/6.0;
    float t = (i+j+k)*G3;
    float X0 = i-t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = j-t;
    float Z0 = k-t;
    float x0 = pos.x-X0; // The x,y,z distances from the cell origin
    float y0 = pos.y-Y0;
    float z0 = pos.z-Z0;
    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
    if(x0>=y0) {
      if(y0>=z0)
        { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; } // X Y Z order
        else if(x0>=z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; } // X Z Y order
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; } // Z X Y order
      }
    else { // x0<y0
      if(y0<z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; } // Z Y X order
      else if(x0<z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; } // Y Z X order
      else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; } // Y X Z order
    }
    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.
    float x1 = x0 - i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - j1 + G3;
    float z1 = z0 - k1 + G3;
    float x2 = x0 - i2 + 2.0*G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - j2 + 2.0*G3;
    float z2 = z0 - k2 + 2.0*G3;
    float x3 = x0 - 1.0 + 3.0*G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0 + 3.0*G3;
    float z3 = z0 - 1.0 + 3.0*G3;
    // Work out the hashed gradient indices of the four simplex corners
    uint ii = i & 255;
    uint jj = j & 255;
    uint kk = k & 255;
    int gi0 = p12[ii+p[jj+p[kk]]];
    int gi1 = p12[ii+i1+p[jj+j1+p[kk+k1]]];
    int gi2 = p12[ii+i2+p[jj+j2+p[kk+k2]]];
    int gi3 = p12[ii+1+p[jj+1+p[kk+1]]];
    // Calculate the contribution from the four corners
    float t0 = 0.5 - x0*x0 - y0*y0 - z0*z0;
    if(t0<0) n0 = 0.0;
    else {
      t0 *= t0;
      n0 = t0 * t0 * dot(grad3[gi0], vec3(x0, y0, z0));
    }
    float t1 = 0.5 - x1*x1 - y1*y1 - z1*z1;
    if(t1<0) n1 = 0.0;
    else {
      t1 *= t1;
      n1 = t1 * t1 * dot(grad3[gi1], vec3(x1, y1, z1));
    }
    float t2 = 0.5 - x2*x2 - y2*y2 - z2*z2;
    if(t2<0) n2 = 0.0;
    else {
      t2 *= t2;
      n2 = t2 * t2 * dot(grad3[gi2], vec3(x2, y2, z2));
    }
    float t3 = 0.5 - x3*x3 - y3*y3 - z3*z3;
    if(t3<0) n3 = 0.0;
    else {
      t3 *= t3;
      n3 = t3 * t3 * dot(grad3[gi3], vec3(x3, y3, z3));
    }
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return (32.0*(n0 + n1 + n2 + n3)) / 2 + 0.5;
}


shared float noiseMap[3*3*3];

void main() {
	const uint gtid = gl_GlobalInvocationID.x;
	const uint gid = gl_WorkGroupID.x;
	const uint tid = gl_LocalInvocationIndex.x;
	const float noiseSize = 16.f;
	const uint gridWidth = 3;//uint(pow(floor(MAX_VERTICES / 3.f), 1.f / 3));
	const uint mcGridWidth = gridWidth - 1;
	const uint gridCount = uint(pow(float(gridWidth), 3));
	vec3 subspacePos = vec3(
		IN.subspaceIndex % (SUBDIVISIONS / (mcGridWidth * 4)),
		(IN.subspaceIndex / (SUBDIVISIONS / (mcGridWidth * 4))) % (SUBDIVISIONS / (mcGridWidth * 4)),
		(IN.subspaceIndex / uint(pow(SUBDIVISIONS / float(mcGridWidth * 4), 2.f)))
	) * mcGridWidth * 4;
	const uint cellCount = uint(pow(mcGridWidth, 3.f));
	uint localSpaceId = IN.localSpaceIndex[gid];
	vec3 localSpacePos = subspacePos + vec3(
		localSpaceId % 4,
		(localSpaceId / 4) % 4,
		localSpaceId / uint(pow(4, 2.f))
	) * mcGridWidth;
	float threshold = 0.5;// + 0.2 * sin(time * PI / 4);
	const uint max_per_meshlet_row = 16;
	
	// Populate noise
	uint noiseMapLoopCount = uint(ceil(float(noiseMap.length()) / gl_WorkGroupSize.x));
	for (uint i = 0; i < noiseMapLoopCount; i++) {
		uint noiseIndex = i * gl_WorkGroupSize.x + tid;
		vec3 noisePos = vec3(
			(noiseIndex % (gridWidth)) + SUBDIVISIONS * time,
			(noiseIndex / (gridWidth)) % uint(gridWidth),
			floor(noiseIndex / pow(gridWidth, 2.f))
		) + localSpacePos;

		// layering and morphing
		float noiseValue = simplexNoise(SCALE * noisePos / SUBDIVISIONS);
		//noiseValue = length(noisePos) / SUBDIVISIONS;
		//noiseValue = (abs(noisePos.x - 0.f) < 0.1f)
		//	&& (abs(noisePos.y - 0.f) < 0.1f)
		//	&& (abs(noisePos.z - 0.f) < 0.1f) ? 1.f : 0.f;
		if (noiseIndex < noiseMap.length()) {
			noiseMap[noiseIndex] = noiseValue;
		}
	}
	//noiseMap[21] = noiseMap[21] > threshold ? 0.f : 1.f; // local space indicator

	mat4 projCamera = proj * camera;

	// Populate tris
	uint cellLoopCount = uint(ceil(float(cellCount) / gl_WorkGroupSize.x));
	uint primCount = 0;
	for (uint i = 0; i < cellLoopCount; i++) {
		uint cellIndex = (i * gl_WorkGroupSize.x) + tid;
		vec3 cellPos = vec3(
			cellIndex % mcGridWidth,
			(cellIndex / mcGridWidth) % mcGridWidth,
			cellIndex / uint(pow(mcGridWidth, 2.f))
		);

		uint config = 0;
#if CACHE_NOISE
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 0;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 1;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 2;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y    ) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 3;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 4;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z    ) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 5;
		config |= (noiseMap[int(cellPos.x) + 1 + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 6;
		config |= (noiseMap[int(cellPos.x)     + (int(cellPos.y + 1) * gridWidth) + (int(cellPos.z + 1) * gridWidth * gridWidth)] > threshold ? 1 : 0) << 7;
#else
		config |= noiseSample(cellPos + vec3(0, 0, 0)) > threshold ? 1 : 0 << 0;
		config |= noiseSample(cellPos + vec3(1, 0, 0)) > threshold ? 1 : 0 << 1;
		config |= noiseSample(cellPos + vec3(1, 0, 1)) > threshold ? 1 : 0 << 2;
		config |= noiseSample(cellPos + vec3(0, 0, 1)) > threshold ? 1 : 0 << 3;
		config |= noiseSample(cellPos + vec3(0, 1, 0)) > threshold ? 1 : 0 << 4;
		config |= noiseSample(cellPos + vec3(1, 1, 0)) > threshold ? 1 : 0 << 5;
		config |= noiseSample(cellPos + vec3(1, 1, 1)) > threshold ? 1 : 0 << 6;
		config |= noiseSample(cellPos + vec3(0, 1, 1)) > threshold ? 1 : 0 << 7;
#endif
		//config = 0x33 << 1;

		int8_t configIndices[15] = cellConfigs[config];
		for (uint j = 0; j < 5; j++) { // per triangle (max 5 tris per cell)
			bool write =
				configIndices[j*3 + 2] != -1 &&
					//(j*3 + 2 + primCount) < MAX_VERTICES &&
					//(j*3 + 2 + primCount) < MAX_PRIMITIVES/3 &&
					cellPos.z < mcGridWidth;
			uvec4 votePrims = subgroupBallot(write);
			uint primCountOffset = subgroupBallotExclusiveBitCount(votePrims);
			vec3 positions[3];
			
			for (uint k = 0; k < 3; k++) { // per vertex
				vec4 colors[] = vec4[](
					vec4(1.f, 0.f, 0.f, 1.f),
					vec4(0.f, 1.f, 0.f, 1.f),
					vec4(0.f, 0.f, 1.f, 1.f)
				);
				uint8_t posIndex = configIndices[j*3 + k];
				uint writeIndex = k + (primCountOffset + primCount) * 3;
				positions[k] = edgePositions[posIndex].xyz;
				vec4 pos = projCamera * vec4(noiseSize * (edgePositions[posIndex].xyz  + cellPos + localSpacePos) / SUBDIVISIONS, 1.f);

				//int junkIndex = 256;
				//junkIndex--;
				if (write) {
					gl_MeshVerticesNV[writeIndex].gl_Position = pos;
					gl_PrimitiveIndicesNV[writeIndex] = writeIndex;

					//v_out[writeIndex].color = colors[writeIndex % 3];
					//primCount += uint(positions[k]); // JUNK try and force memory request
				}
				//junkIndex++;
			}
			vec3 v = positions[0] - positions[1];
			vec3 w = positions[0] - positions[2]; 
			vec3 normal = normalize(vec3(
				(v.y * w.z) - (v.z * w.y),
				(v.z * w.x) - (v.x * w.z),
				(v.x * w.y) - (v.y * w.x)
			));
			//normal = abs(normal);
			normal += 1;
			normal /= 2;
			normal = normalize(normal);
			if (write) {
				v_out[(primCountOffset + primCount) * 3].color =
					v_out[(primCountOffset + primCount) * 3 + 1].color =
					v_out[(primCountOffset + primCount) * 3 + 2].color = 
						vec4(normal, 1.f);
				//v_out[(primCountOffset + primCount) * 3].somethingElse = projCamera;
			}

			primCount += subgroupBallotBitCount(votePrims);
			//primCount += configIndices[0]; // JUNK try and force memory request
		}
	}
	
	gl_PrimitiveCountNV = primCount;
	//gl_PrimitiveCountNV = 0;

	/*
	if (IN.subspaceIndex == 0 && IN.localSpaceIndex[0] == 0 && tid == 0) {
		const int bufferEater = MAX_VERTICES - MAX_VERTICES;
		gl_MeshVerticesNV[0 + bufferEater].gl_Position = projCamera * vec4(0, 1, 0, 1);
		gl_PrimitiveIndicesNV[0] = 0 + bufferEater;
		v_out[0 + bufferEater].color = (1.f).xxxx;

		gl_MeshVerticesNV[1 + bufferEater].gl_Position = projCamera * vec4(0, 0, 1, 1);
		gl_PrimitiveIndicesNV[1] = 1 + bufferEater;
		v_out[1 + bufferEater].color = (1.f).xxxx;
	
		gl_MeshVerticesNV[2 + bufferEater].gl_Position = projCamera * vec4(1, 1, 0, 1);
		gl_PrimitiveIndicesNV[2] = 2 + bufferEater;
		v_out[2 + bufferEater].color = (1.f).xxxx;
	
		//gl_PrimitiveCountNV = 1;
	}
	*/
	

#if DRAW_DATA_MEASURE
	localSpaceTriCounts[64 * IN.subspaceIndex + localSpaceId] = uint8_t(primCount);
#endif
}
