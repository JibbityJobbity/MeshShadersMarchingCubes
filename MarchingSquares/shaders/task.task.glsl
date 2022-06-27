#version 450

#extension GL_NV_mesh_shader : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_KHR_shader_subgroup_ballot : require

#define PI 3.1415926535897932384626433832795

#define ELECT_METHOD_CENTER 1
#define ELECT_METHOD_CORNERS 0

layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer MeshInfo {
	vec3 sunlightDir;
	float scale;
	uint subdivisions;
	uint8_t p[512]; // Couldn't put these arrays in constant memory for some reason
	uint8_t p12[512];
	int8_t cellConfigs[256][15];
    vec4 edgePositions[12];
};

// For measuring culling method accuracy
layout(binding = 1) buffer Output {
	uint8_t localSpaceTriCounts[LOCAL_SPACE_COUNT];
	uint localSpaceShouldDraw[LOCAL_SPACE_COUNT / 32];
};

layout(push_constant) uniform constants{
	mat4 proj;
	mat4 camera; // Should really be combined on the CPU
	float time;
};

// New Perlin noise, from Ken Perlin's reference Java implementation
// Original used doubles, only need float. F16 might be better on tensor cores?
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

out taskNV Task {
	uint subspaceIndex;
	uint localSpaceIndex[64];
} OUT;

#if ELECT_METHOD_CORNERS
shared float noiseMap[5*5*5];
#endif

void main() {
	const uint tid = gl_LocalInvocationID.x;
	const uint gid = gl_WorkGroupID.x;

	const uint mcGridSize = 2; // Local space size
	// Could tie the threshold to a function! But inhibits compiler optimizations.
	float noiseThreshold = 0.5;// + 0.2 * sin(time * PI / 4);

	// 4 local spaces for a subspace
	vec3 subspacePos = vec3(
		gid % (SUBDIVISIONS / (mcGridSize*4)),
		(gid / (SUBDIVISIONS / (mcGridSize*4))) % (SUBDIVISIONS / (mcGridSize*4)),
		(gid / uint(pow(SUBDIVISIONS / float(mcGridSize*4), 2.f)))
	) * mcGridSize * 4;

	uint localSpaceCount = 0;
#if ELECT_METHOD_CENTER
	const float electThreshold = 0.005;// + (0.02 * sin(time * PI));
	const uint noiseLoopCount = 64 / gl_WorkGroupSize.x;
	for (uint i = 0; i < noiseLoopCount; i++) {
		uint noiseIndex = i * gl_WorkGroupSize.x + tid;

		vec3 noisePos = vec3(
			(noiseIndex % 4) + (SUBDIVISIONS * (time / mcGridSize)),
			(noiseIndex / 4) % 4,
			noiseIndex / 16
		);
		const vec3 middleOfLocalSpace = (0.5).xxx;
		
		// Sample noise
		float noiseHere = simplexNoise(SCALE * (mcGridSize*(noisePos + middleOfLocalSpace) + subspacePos) / SUBDIVISIONS);
		bool elect = abs(noiseHere - noiseThreshold) < electThreshold;
		//elect = true;
		uvec4 ballot = subgroupBallot(elect);
		uint localOffset = subgroupBallotExclusiveBitCount(ballot);
#if DRAW_DATA_MEASURE
		OUT.localSpaceIndex[noiseIndex] = noiseIndex;
		localSpaceShouldDraw[2 * gid + i] = ballot.x;
		localSpaceCount += gl_WorkGroupSize.x;
#else
		if (elect)
			OUT.localSpaceIndex[localSpaceCount + localOffset] = noiseIndex;
		localSpaceCount += subgroupBallotBitCount(ballot);
#endif
	}

#elif ELECT_METHOD_CORNERS
	// Noise Loop
	const uint noiseLoopCount = uint(ceil(float(noiseMap.length()) / gl_WorkGroupSize.x));
	const float bias = 0.000225;
	for (uint i = 0; i < noiseLoopCount; i++) {
		uint noiseIndex = i * gl_WorkGroupSize.x + tid;

		vec3 noisePos = vec3(
			(noiseIndex % 5) + (SUBDIVISIONS * (time / mcGridSize)),
			(noiseIndex / 5) % 5,
			noiseIndex / 25 
		);
		const vec3 middleOfLocalSpace = (0.5).xxx;
		float noiseHere = simplexNoise(SCALE * (mcGridSize * noisePos + subspacePos) / SUBDIVISIONS);
		if (noiseIndex < noiseMap.length())
			noiseMap[noiseIndex] = noiseHere;
	}
	
	const uint localSpaceLoopCount = 64 / gl_WorkGroupSize.x;
	for (uint i = 0; i < localSpaceLoopCount; i++) {
		uint noiseIndex = i * gl_WorkGroupSize.x + tid;
		// For loop with positions would be better...
		// ... and a function to convert indices.
		float maximum = bias +
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16)],
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 1],
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 5],
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 5 + 1],
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25],
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25 + 1],
			max(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25 + 5],
			    noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25 + 5 + 1])))))));
		float minimum = (-bias) + 
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16)],
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 1],
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 5],
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 5 + 1],
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25],
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25 + 1],
			min(noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25 + 5],
			    noiseMap[noiseIndex + (noiseIndex/4) + 5*(noiseIndex/16) + 25 + 5 + 1])))))));
		// Stupid thing that doesn't work, probably needs brackets or something...
		// If corners reside on the same side of the threshold
		/*bool elect = noiseMap[noiseIndex] > noiseThreshold ==
			noiseMap[noiseIndex + 1] > noiseThreshold ==
			noiseMap[noiseIndex + 4] > noiseThreshold ==
			noiseMap[noiseIndex + 4 + 1] > noiseThreshold ==
			noiseMap[noiseIndex + 16] > noiseThreshold ==
			noiseMap[noiseIndex + 16 + 1] > noiseThreshold ==
			noiseMap[noiseIndex + 16 + 4] > noiseThreshold ==
			noiseMap[noiseIndex + 16 + 4 + 1] > noiseThreshold;*/
		bool elect = minimum < noiseThreshold && maximum > noiseThreshold;
		//elect = true;

		uvec4 ballot = subgroupBallot(elect);
		uint localOffset = subgroupBallotExclusiveBitCount(ballot);
#if DRAW_DATA_MEASURE
		OUT.localSpaceIndex[noiseIndex] = noiseIndex;
		localSpaceShouldDraw[2 * gid + i] = ballot.x;
		localSpaceCount += gl_WorkGroupSize.x;
#else
		if (elect)
			OUT.localSpaceIndex[localSpaceCount + localOffset] = noiseIndex;
		localSpaceCount += subgroupBallotBitCount(ballot);
#endif
	}
#endif

	if (tid == 0) {
		OUT.subspaceIndex = gid;
#if DRAW_DATA_MEASURE
		gl_TaskCountNV = localSpaceCount;
#else
		gl_TaskCountNV = localSpaceCount;
#endif
	}
	//gl_TaskCountNV = 1; // JUNK
	//OUT.localSpaceIndex[0] = 0;
}