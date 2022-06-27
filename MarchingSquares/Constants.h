#ifndef CONSTANTS_H
#define CONSTANTS_H

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

const int MAX_FRAMES_IN_FLIGHT = 2;
const int SUBDIVISIONS =  8 * 32; // Marching cubes resolution (was 12 * 32)
const float SCALE = 2; // Noise scale (left at 2)
const int FRAME_SAMPLE_COUNT = 200; // Number of samples when measuring accuracy
#define DEFAULT_NOISE_PBOX 1 // Standardize noise
#define DRAW_DATA_MEASURE 0 // Measure accuracy rather than performance
#define TIME_SCALE 0
#define TIMER_ENABLE 0 // Stop after reaching timer or enough samples
#define CELL_INDICES_ENCODE 0 // Read blanket.mesh.glsl
#define BLOAT 0	// Enable sparse memory layout (cause bad cache performance)
#if BLOAT
#error Not implemented!
#define BLOAT_FACTOR 64
#endif
const glm::vec3 STARTING_POS = glm::vec3(-4, 8, -4);
const glm::vec3 STARTING_ROT = glm::vec3(0.796997547, -0.165999562, 0);
const glm::vec3 SUNLIGHT_DIR = glm::normalize(glm::vec3(1.f));

const double PI = 3.141592653589793238463;

const bool performanceMarkers = true;
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

#endif
