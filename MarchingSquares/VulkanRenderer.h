#pragma once

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#elif __linux__
#define VK_USE_PLATFORM_XLIB_KHR
#endif
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

#include <GLFW/glfw3.h>
#ifdef _WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#elif __linux__
#define GLFW_EXPOSE_NATIVE_X11
#endif
#include <GLFW/glfw3native.h>

#include <glm/glm.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <optional>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <random>

#include "wrappers.h"
#include "FileObserver.h"
#include "Constants.h"

#ifdef __linux__
#define ARRAYSIZE(a) \
  ((sizeof(a) / sizeof(*(a))) / \
  static_cast<size_t>(!(sizeof(a) % sizeof(*(a)))))
#endif

#define MESH_SHADERS 1
#define LOCAL_SPACE_COUNT ((SUBDIVISIONS/2) * (SUBDIVISIONS/2) * (SUBDIVISIONS/2))

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
#if MESH_SHADERS == 1
    VK_NV_MESH_SHADER_EXTENSION_NAME,
#endif
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct MeshInfo {
    glm::vec3 sunlightDir;
    float scale;
    uint32_t subdivisions;
#if BLOAT
    uint8_t pBox[512 * 512];
#else
    uint8_t pBox[512];
#endif
    uint8_t pBox12[512];
    int8_t cellConfigs[256][15];
    alignas(16) glm::vec4 edgePositions[12];
};

struct DrawOutputInfo {
    uint8_t localSpaceTriCounts[LOCAL_SPACE_COUNT];
    uint32_t localSpaceShouldDraw[LOCAL_SPACE_COUNT / 32];
};

struct alignas(16) PushConstantInfo {
    glm::mat4 proj;
    glm::mat4 camera;
    float time;
};

struct GameState {
    glm::vec3 pos;
    glm::vec3 rotate;
    float time, mouseX, mouseY;
};

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = vk::VertexInputRate::eVertex;

        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = vk::Format::eR32G32Sfloat;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = vk::Format::eR32G32B32Sfloat;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices = {
    0, 1, 2, 2, 3, 0
};

class VulkanRenderer {
public:
    void run();

private:
    bool running;
    bool firstFrame;
    GameState gameState;
    PushConstantInfo pushConstantInfo;
    std::ofstream drawDataOutput;

    vk::DynamicLoader dl;
    GLFWwindow* window;

    vk::Instance instance;
    vk::DebugUtilsMessengerEXT debugMessenger;
    vk::SurfaceKHR surface;

    vk::PhysicalDevice physicalDevice;
    vk::Device device;

    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::SwapchainKHR swapChain;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::Image> depthImages;
    std::vector<vk::ImageView> depthImageViews;
    vk::DeviceMemory depthImageMemory;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;

    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::DescriptorSetLayout descriptorSetLayout;
    vk::Pipeline graphicsPipeline;

    vk::CommandPool commandPool;

    vk::Buffer meshInfoBuffer;
    vk::DeviceMemory meshInfoMemory;
    std::vector<vk::Buffer> drawOutputBuffers;
    std::vector<vk::DeviceMemory> drawOutputMemorys;
    DrawOutputInfo* downloadedDrawInfo;
    vk::Buffer downloadBuffer;
    vk::DeviceMemory downloadMemory;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::DescriptorPool descriptorPool;

    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;
    size_t lastFrame = -1;
    size_t frameCount = 0;
    std::chrono::high_resolution_clock::time_point startTime;
    std::vector<FileObserver*> shaderObservers;
    std::vector<ShaderInfo> shaderInfos {
#if MESH_SHADERS
        { "mesh_shader", "shaders/flat.mesh.glsl", vk::ShaderStageFlagBits::eMeshNV }, // blanket, flat, smooth
        { "task_shader", "shaders/task.task.glsl", vk::ShaderStageFlagBits::eTaskNV },
#else
        { "vertex_shader", "shaders/vertex.vert.glsl", vk::ShaderStageFlagBits::eVertex },
#endif
        { "fragment_shader", "shaders/plain.frag.glsl", vk::ShaderStageFlagBits::eFragment },
    };
    std::vector<DescriptorSetLayoutBinding> layoutBindings{
        DescriptorSetLayoutBinding { vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eTaskNV | vk::ShaderStageFlagBits::eMeshNV | vk::ShaderStageFlagBits::eFragment },
        DescriptorSetLayoutBinding { vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eTaskNV | vk::ShaderStageFlagBits::eMeshNV },
    };


    bool framebufferResized = false;
    void initWindow();
    static void framebufferResizeCallback(GLFWwindow* window, int width, int height);
    void initVulkan();
    void mainLoop();
    void checkShaders();
    void cleanupSwapChain();
    void cleanup();
    void recreateSwapChain();
    void createInstance();
    void populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo);
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createDepthResources();
    void createRenderPass();
    void createGraphicsPipeline();
    void createDescriptorSetLayout(const std::vector<DescriptorSetLayoutBinding>& layouts);
    void createDescriptorPool(size_t descriptorCount);
    void createDescriptorSets();
    void createFramebuffers();
    void createCommandPool();
    void createMeshInfo();
    void createOutputBuffers();
    void createDownloadResources();
    void createDrawInfoFile();
    void createShaderObservers();
    void setupWindowControls();
    void createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory);
    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    void allocateCommandBuffers();
    void recordCommandBuffer(uint32_t index);
    void createSyncObjects();
    void drawFrame();
    void handleEvents();
    void downloadDrawData(uint32_t imageIndex);
    vk::Format findDepthFormat();
    vk::ShaderModule createShaderModule(const std::vector<char>& code);
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
    SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice device);
    bool isDeviceSuitable(vk::PhysicalDevice device);
    bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
    std::vector<const char*> getRequiredExtensions();
    bool checkValidationLayerSupport();
    static std::string readFile(const std::string& filename);
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData);
    vk::ShaderModule compileGlslFile(const std::string& source_name, const std::vector<ShadercMacro>& macros, vk::ShaderStageFlags stage, const std::string& source, bool optimize = false, bool debugInfo = false);
};

const static int8_t cellConfigurations[256][15] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,9,8,1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,2,10,0,2,9,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,8,3,2,10,8,10,9,8,-1,-1,-1,-1,-1,-1},
    {3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,8,11,0,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,11,2,1,9,11,9,8,11,-1,-1,-1,-1,-1,-1},
    {3,10,1,11,10,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,10,1,0,8,10,8,11,10,-1,-1,-1,-1,-1,-1},
    {3,9,0,3,11,9,11,10,9,-1,-1,-1,-1,-1,-1},
    {9,8,10,10,8,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,7,3,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,1,9,4,7,1,7,3,1,-1,-1,-1,-1,-1,-1},
    {1,2,10,8,4,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,4,7,3,0,4,1,2,10,-1,-1,-1,-1,-1,-1},
    {9,2,10,9,0,2,8,4,7,-1,-1,-1,-1,-1,-1},
    {2,10,9,2,9,7,2,7,3,7,9,4,-1,-1,-1},
    {8,4,7,3,11,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,4,7,11,2,4,2,0,4,-1,-1,-1,-1,-1,-1},
    {9,0,1,8,4,7,2,3,11,-1,-1,-1,-1,-1,-1},
    {4,7,11,9,4,11,9,11,2,9,2,1,-1,-1,-1},
    {3,10,1,3,11,10,7,8,4,-1,-1,-1,-1,-1,-1},
    {1,11,10,1,4,11,1,0,4,7,11,4,-1,-1,-1},
    {4,7,8,9,0,11,9,11,10,11,0,3,-1,-1,-1},
    {4,7,11,4,11,9,9,11,10,-1,-1,-1,-1,-1,-1},
    {9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,5,4,1,5,0,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,5,4,8,3,5,3,1,5,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,10,4,9,5,-1,-1,-1,-1,-1,-1},
    {5,2,10,5,4,2,4,0,2,-1,-1,-1,-1,-1,-1},
    {2,10,5,3,2,5,3,5,4,3,4,8,-1,-1,-1},
    {9,5,4,2,3,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,11,2,0,8,11,4,9,5,-1,-1,-1,-1,-1,-1},
    {0,5,4,0,1,5,2,3,11,-1,-1,-1,-1,-1,-1},
    {2,1,5,2,5,8,2,8,11,4,8,5,-1,-1,-1},
    {10,3,11,10,1,3,9,5,4,-1,-1,-1,-1,-1,-1},
    {4,9,5,0,8,1,8,10,1,8,11,10,-1,-1,-1},
    {5,4,0,5,0,11,5,11,10,11,0,3,-1,-1,-1},
    {5,4,8,5,8,10,10,8,11,-1,-1,-1,-1,-1,-1},
    {9,7,8,5,7,9,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,3,0,9,5,3,5,7,3,-1,-1,-1,-1,-1,-1},
    {0,7,8,0,1,7,1,5,7,-1,-1,-1,-1,-1,-1},
    {1,5,3,3,5,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,7,8,9,5,7,10,1,2,-1,-1,-1,-1,-1,-1},
    {10,1,2,9,5,0,5,3,0,5,7,3,-1,-1,-1},
    {8,0,2,8,2,5,8,5,7,10,5,2,-1,-1,-1},
    {2,10,5,2,5,3,3,5,7,-1,-1,-1,-1,-1,-1},
    {7,9,5,7,8,9,3,11,2,-1,-1,-1,-1,-1,-1},
    {9,5,7,9,7,2,9,2,0,2,7,11,-1,-1,-1},
    {2,3,11,0,1,8,1,7,8,1,5,7,-1,-1,-1},
    {11,2,1,11,1,7,7,1,5,-1,-1,-1,-1,-1,-1},
    {9,5,8,8,5,7,10,1,3,10,3,11,-1,-1,-1},
    {5,7,0,5,0,9,7,11,0,1,0,10,11,10,0},
    {11,10,0,11,0,3,10,5,0,8,0,7,5,7,0},
    {11,10,5,7,11,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,0,1,5,10,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,8,3,1,9,8,5,10,6,-1,-1,-1,-1,-1,-1},
    {1,6,5,2,6,1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,6,5,1,2,6,3,0,8,-1,-1,-1,-1,-1,-1},
    {9,6,5,9,0,6,0,2,6,-1,-1,-1,-1,-1,-1},
    {5,9,8,5,8,2,5,2,6,3,2,8,-1,-1,-1},
    {2,3,11,10,6,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,0,8,11,2,0,10,6,5,-1,-1,-1,-1,-1,-1},
    {0,1,9,2,3,11,5,10,6,-1,-1,-1,-1,-1,-1},
    {5,10,6,1,9,2,9,11,2,9,8,11,-1,-1,-1},
    {6,3,11,6,5,3,5,1,3,-1,-1,-1,-1,-1,-1},
    {0,8,11,0,11,5,0,5,1,5,11,6,-1,-1,-1},
    {3,11,6,0,3,6,0,6,5,0,5,9,-1,-1,-1},
    {6,5,9,6,9,11,11,9,8,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,3,0,4,7,3,6,5,10,-1,-1,-1,-1,-1,-1},
    {1,9,0,5,10,6,8,4,7,-1,-1,-1,-1,-1,-1},
    {10,6,5,1,9,7,1,7,3,7,9,4,-1,-1,-1},
    {6,1,2,6,5,1,4,7,8,-1,-1,-1,-1,-1,-1},
    {1,2,5,5,2,6,3,0,4,3,4,7,-1,-1,-1},
    {8,4,7,9,0,5,0,6,5,0,2,6,-1,-1,-1},
    {7,3,9,7,9,4,3,2,9,5,9,6,2,6,9},
    {3,11,2,7,8,4,10,6,5,-1,-1,-1,-1,-1,-1},
    {5,10,6,4,7,2,4,2,0,2,7,11,-1,-1,-1},
    {0,1,9,4,7,8,2,3,11,5,10,6,-1,-1,-1},
    {9,2,1,9,11,2,9,4,11,7,11,4,5,10,6},
    {8,4,7,3,11,5,3,5,1,5,11,6,-1,-1,-1},
    {5,1,11,5,11,6,1,0,11,7,11,4,0,4,11},
    {0,5,9,0,6,5,0,3,6,11,6,3,8,4,7},
    {6,5,9,6,9,11,4,7,9,7,11,9,-1,-1,-1},
    {10,4,9,6,4,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,10,6,4,9,10,0,8,3,-1,-1,-1,-1,-1,-1},
    {10,0,1,10,6,0,6,4,0,-1,-1,-1,-1,-1,-1},
    {8,3,1,8,1,6,8,6,4,6,1,10,-1,-1,-1},
    {1,4,9,1,2,4,2,6,4,-1,-1,-1,-1,-1,-1},
    {3,0,8,1,2,9,2,4,9,2,6,4,-1,-1,-1},
    {0,2,4,4,2,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,3,2,8,2,4,4,2,6,-1,-1,-1,-1,-1,-1},
    {10,4,9,10,6,4,11,2,3,-1,-1,-1,-1,-1,-1},
    {0,8,2,2,8,11,4,9,10,4,10,6,-1,-1,-1},
    {3,11,2,0,1,6,0,6,4,6,1,10,-1,-1,-1},
    {6,4,1,6,1,10,4,8,1,2,1,11,8,11,1},
    {9,6,4,9,3,6,9,1,3,11,6,3,-1,-1,-1},
    {8,11,1,8,1,0,11,6,1,9,1,4,6,4,1},
    {3,11,6,3,6,0,0,6,4,-1,-1,-1,-1,-1,-1},
    {6,4,8,11,6,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,10,6,7,8,10,8,9,10,-1,-1,-1,-1,-1,-1},
    {0,7,3,0,10,7,0,9,10,6,7,10,-1,-1,-1},
    {10,6,7,1,10,7,1,7,8,1,8,0,-1,-1,-1},
    {10,6,7,10,7,1,1,7,3,-1,-1,-1,-1,-1,-1},
    {1,2,6,1,6,8,1,8,9,8,6,7,-1,-1,-1},
    {2,6,9,2,9,1,6,7,9,0,9,3,7,3,9},
    {7,8,0,7,0,6,6,0,2,-1,-1,-1,-1,-1,-1},
    {7,3,2,6,7,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,11,10,6,8,10,8,9,8,6,7,-1,-1,-1},
    {2,0,7,2,7,11,0,9,7,6,7,10,9,10,7},
    {1,8,0,1,7,8,1,10,7,6,7,10,2,3,11},
    {11,2,1,11,1,7,10,6,1,6,7,1,-1,-1,-1},
    {8,9,6,8,6,7,9,1,6,11,6,3,1,3,6},
    {0,9,1,11,6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,8,0,7,0,6,3,11,0,11,6,0,-1,-1,-1},
    {7,11,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,8,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,1,9,11,7,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,1,9,8,3,1,11,7,6,-1,-1,-1,-1,-1,-1},
    {10,1,2,6,11,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,8,6,11,7,-1,-1,-1,-1,-1,-1},
    {2,9,0,2,10,9,6,11,7,-1,-1,-1,-1,-1,-1},
    {6,11,7,2,10,3,10,8,3,10,9,8,-1,-1,-1},
    {7,2,3,6,2,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {7,0,8,7,6,0,6,2,0,-1,-1,-1,-1,-1,-1},
    {2,7,6,2,3,7,0,1,9,-1,-1,-1,-1,-1,-1},
    {1,6,2,1,8,6,1,9,8,8,7,6,-1,-1,-1},
    {10,7,6,10,1,7,1,3,7,-1,-1,-1,-1,-1,-1},
    {10,7,6,1,7,10,1,8,7,1,0,8,-1,-1,-1},
    {0,3,7,0,7,10,0,10,9,6,10,7,-1,-1,-1},
    {7,6,10,7,10,8,8,10,9,-1,-1,-1,-1,-1,-1},
    {6,8,4,11,8,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,6,11,3,0,6,0,4,6,-1,-1,-1,-1,-1,-1},
    {8,6,11,8,4,6,9,0,1,-1,-1,-1,-1,-1,-1},
    {9,4,6,9,6,3,9,3,1,11,3,6,-1,-1,-1},
    {6,8,4,6,11,8,2,10,1,-1,-1,-1,-1,-1,-1},
    {1,2,10,3,0,11,0,6,11,0,4,6,-1,-1,-1},
    {4,11,8,4,6,11,0,2,9,2,10,9,-1,-1,-1},
    {10,9,3,10,3,2,9,4,3,11,3,6,4,6,3},
    {8,2,3,8,4,2,4,6,2,-1,-1,-1,-1,-1,-1},
    {0,4,2,4,6,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,9,0,2,3,4,2,4,6,4,3,8,-1,-1,-1},
    {1,9,4,1,4,2,2,4,6,-1,-1,-1,-1,-1,-1},
    {8,1,3,8,6,1,8,4,6,6,10,1,-1,-1,-1},
    {10,1,0,10,0,6,6,0,4,-1,-1,-1,-1,-1,-1},
    {4,6,3,4,3,8,6,10,3,0,3,9,10,9,3},
    {10,9,4,6,10,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,5,7,6,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,5,11,7,6,-1,-1,-1,-1,-1,-1},
    {5,0,1,5,4,0,7,6,11,-1,-1,-1,-1,-1,-1},
    {11,7,6,8,3,4,3,5,4,3,1,5,-1,-1,-1},
    {9,5,4,10,1,2,7,6,11,-1,-1,-1,-1,-1,-1},
    {6,11,7,1,2,10,0,8,3,4,9,5,-1,-1,-1},
    {7,6,11,5,4,10,4,2,10,4,0,2,-1,-1,-1},
    {3,4,8,3,5,4,3,2,5,10,5,2,11,7,6},
    {7,2,3,7,6,2,5,4,9,-1,-1,-1,-1,-1,-1},
    {9,5,4,0,8,6,0,6,2,6,8,7,-1,-1,-1},
    {3,6,2,3,7,6,1,5,0,5,4,0,-1,-1,-1},
    {6,2,8,6,8,7,2,1,8,4,8,5,1,5,8},
    {9,5,4,10,1,6,1,7,6,1,3,7,-1,-1,-1},
    {1,6,10,1,7,6,1,0,7,8,7,0,9,5,4},
    {4,0,10,4,10,5,0,3,10,6,10,7,3,7,10},
    {7,6,10,7,10,8,5,4,10,4,8,10,-1,-1,-1},
    {6,9,5,6,11,9,11,8,9,-1,-1,-1,-1,-1,-1},
    {3,6,11,0,6,3,0,5,6,0,9,5,-1,-1,-1},
    {0,11,8,0,5,11,0,1,5,5,6,11,-1,-1,-1},
    {6,11,3,6,3,5,5,3,1,-1,-1,-1,-1,-1,-1},
    {1,2,10,9,5,11,9,11,8,11,5,6,-1,-1,-1},
    {0,11,3,0,6,11,0,9,6,5,6,9,1,2,10},
    {11,8,5,11,5,6,8,0,5,10,5,2,0,2,5},
    {6,11,3,6,3,5,2,10,3,10,5,3,-1,-1,-1},
    {5,8,9,5,2,8,5,6,2,3,8,2,-1,-1,-1},
    {9,5,6,9,6,0,0,6,2,-1,-1,-1,-1,-1,-1},
    {1,5,8,1,8,0,5,6,8,3,8,2,6,2,8},
    {1,5,6,2,1,6,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,6,1,6,10,3,8,6,5,6,9,8,9,6},
    {10,1,0,10,0,6,9,5,0,5,6,0,-1,-1,-1},
    {0,3,8,5,6,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,5,6,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,7,5,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,5,10,11,7,5,8,3,0,-1,-1,-1,-1,-1,-1},
    {5,11,7,5,10,11,1,9,0,-1,-1,-1,-1,-1,-1},
    {10,7,5,10,11,7,9,8,1,8,3,1,-1,-1,-1},
    {11,1,2,11,7,1,7,5,1,-1,-1,-1,-1,-1,-1},
    {0,8,3,1,2,7,1,7,5,7,2,11,-1,-1,-1},
    {9,7,5,9,2,7,9,0,2,2,11,7,-1,-1,-1},
    {7,5,2,7,2,11,5,9,2,3,2,8,9,8,2},
    {2,5,10,2,3,5,3,7,5,-1,-1,-1,-1,-1,-1},
    {8,2,0,8,5,2,8,7,5,10,2,5,-1,-1,-1},
    {9,0,1,5,10,3,5,3,7,3,10,2,-1,-1,-1},
    {9,8,2,9,2,1,8,7,2,10,2,5,7,5,2},
    {1,3,5,3,7,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,8,7,0,7,1,1,7,5,-1,-1,-1,-1,-1,-1},
    {9,0,3,9,3,5,5,3,7,-1,-1,-1,-1,-1,-1},
    {9,8,7,5,9,7,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {5,8,4,5,10,8,10,11,8,-1,-1,-1,-1,-1,-1},
    {5,0,4,5,11,0,5,10,11,11,3,0,-1,-1,-1},
    {0,1,9,8,4,10,8,10,11,10,4,5,-1,-1,-1},
    {10,11,4,10,4,5,11,3,4,9,4,1,3,1,4},
    {2,5,1,2,8,5,2,11,8,4,5,8,-1,-1,-1},
    {0,4,11,0,11,3,4,5,11,2,11,1,5,1,11},
    {0,2,5,0,5,9,2,11,5,4,5,8,11,8,5},
    {9,4,5,2,11,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,5,10,3,5,2,3,4,5,3,8,4,-1,-1,-1},
    {5,10,2,5,2,4,4,2,0,-1,-1,-1,-1,-1,-1},
    {3,10,2,3,5,10,3,8,5,4,5,8,0,1,9},
    {5,10,2,5,2,4,1,9,2,9,4,2,-1,-1,-1},
    {8,4,5,8,5,3,3,5,1,-1,-1,-1,-1,-1,-1},
    {0,4,5,1,0,5,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {8,4,5,8,5,3,9,0,5,0,3,5,-1,-1,-1},
    {9,4,5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,11,7,4,9,11,9,10,11,-1,-1,-1,-1,-1,-1},
    {0,8,3,4,9,7,9,11,7,9,10,11,-1,-1,-1},
    {1,10,11,1,11,4,1,4,0,7,4,11,-1,-1,-1},
    {3,1,4,3,4,8,1,10,4,7,4,11,10,11,4},
    {4,11,7,9,11,4,9,2,11,9,1,2,-1,-1,-1},
    {9,7,4,9,11,7,9,1,11,2,11,1,0,8,3},
    {11,7,4,11,4,2,2,4,0,-1,-1,-1,-1,-1,-1},
    {11,7,4,11,4,2,8,3,4,3,2,4,-1,-1,-1},
    {2,9,10,2,7,9,2,3,7,7,4,9,-1,-1,-1},
    {9,10,7,9,7,4,10,2,7,8,7,0,2,0,7},
    {3,7,10,3,10,2,7,4,10,1,10,0,4,0,10},
    {1,10,2,8,7,4,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,7,1,3,-1,-1,-1,-1,-1,-1},
    {4,9,1,4,1,7,0,8,1,8,7,1,-1,-1,-1},
    {4,0,3,7,4,3,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {4,8,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,8,10,11,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,11,9,10,-1,-1,-1,-1,-1,-1},
    {0,1,10,0,10,8,8,10,11,-1,-1,-1,-1,-1,-1},
    {3,1,10,11,3,10,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,2,11,1,11,9,9,11,8,-1,-1,-1,-1,-1,-1},
    {3,0,9,3,9,11,1,2,9,2,11,9,-1,-1,-1},
    {0,2,11,8,0,11,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {3,2,11,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,10,8,9,-1,-1,-1,-1,-1,-1},
    {9,10,2,0,9,2,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {2,3,8,2,8,10,0,1,8,1,10,8,-1,-1,-1},
    {1,10,2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {1,3,8,9,1,8,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,9,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {0,3,8,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
};
