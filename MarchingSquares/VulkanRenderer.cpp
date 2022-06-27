#include "stdafx.h"
#include "VulkanRenderer.h"
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr) {
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    }
    else {
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr) {
        func(instance, debugMessenger, pAllocator);
    }
}

// Compiles a GLSL shader source to a shader module.
vk::ShaderModule VulkanRenderer::compileGlslFile(const std::string& source_name, const std::vector<ShadercMacro>& macros, vk::ShaderStageFlags stage, const std::string& source, bool optimize, bool debugInfo) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    for (const auto& m : macros) {
        options.AddMacroDefinition(m.key, m.value);
    }
    if (optimize)
        options.SetOptimizationLevel(shaderc_optimization_level_performance);
    if (debugInfo)
        options.SetGenerateDebugInfo();
    options.SetTargetSpirv(shaderc_spirv_version_1_5);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    // vk::ShaderStageFlags isn't enum or class, can't use switch and can't cast to vk::ShaderStageFlagBits...  :/
    shaderc_shader_kind shaderKind;
    if (stage == vk::ShaderStageFlagBits::eVertex)
        shaderKind = shaderc_vertex_shader;
    else if (stage == vk::ShaderStageFlagBits::eFragment)
        shaderKind = shaderc_fragment_shader;
    else if (stage == vk::ShaderStageFlagBits::eCompute)
        shaderKind = shaderc_compute_shader;
    else if (stage == vk::ShaderStageFlagBits::eGeometry)
        shaderKind = shaderc_geometry_shader;
    else if (stage == vk::ShaderStageFlagBits::eTessellationControl)
        shaderKind = shaderc_tess_control_shader;
    else if (stage == vk::ShaderStageFlagBits::eTessellationEvaluation)
        shaderKind = shaderc_tess_evaluation_shader;
    else if (stage == vk::ShaderStageFlagBits::eTaskNV)
        shaderKind = shaderc_task_shader;
    else if (stage == vk::ShaderStageFlagBits::eMeshNV)
        shaderKind = shaderc_mesh_shader;
    else
        shaderKind = shaderc_callable_shader;

    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source, shaderKind, source_name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << module.GetErrorMessage();
        return { VK_NULL_HANDLE };
    }
    std::vector<uint32_t> shaderCode = { module.cbegin(), module.cend() };

    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
    createInfo.pCode = shaderCode.data();

    vk::ShaderModule shaderModule = device.createShaderModule(createInfo);
    if (!shaderModule) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

void VulkanRenderer::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

void VulkanRenderer::initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
}

void VulkanRenderer::framebufferResizeCallback(GLFWwindow* window, int width, int height) {
    auto app = reinterpret_cast<VulkanRenderer*>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

void VulkanRenderer::initVulkan() {
    gameState = GameState{
        STARTING_POS,
        STARTING_ROT,
        0.f
    };

    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createDepthResources();
    createRenderPass();
    createDescriptorSetLayout(layoutBindings);
    createGraphicsPipeline();
    createFramebuffers();
    createCommandPool();
    createDescriptorPool(layoutBindings.size());
    createMeshInfo();
    createOutputBuffers();
    createDownloadResources();
#if DRAW_DATA_MEASURE
    createDrawInfoFile();
#endif
    createDescriptorSets();
    allocateCommandBuffers();
    createSyncObjects();
    if (!TIMER_ENABLE)
		createShaderObservers();

    setupWindowControls();
}

void VulkanRenderer::setupWindowControls() {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
}

void VulkanRenderer::mainLoop() {
    running = true;
    startTime = std::chrono::high_resolution_clock::now();
    while (running) {
        checkShaders();
        glfwPollEvents();
        handleEvents();
        drawFrame();
    }

    using namespace std::chrono;
    std::cout << "Average framerate: " << frameCount / duration_cast<duration<double>>(high_resolution_clock::now() - startTime).count() << std::endl;

    auto result = device.waitForFences(imagesInFlight, true, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Error waiting for fences");
    }
    //device.waitIdle();
}

void VulkanRenderer::checkShaders() {
    bool recompile = false;
    for (auto observer : shaderObservers) {
        if (observer->Updated) {
            recompile = true;
            observer->Reset();
        }
    }
    if (recompile) {
        auto result = device.waitForFences(inFlightFences, true, UINT64_MAX);
        if (result != vk::Result::eSuccess) {
            throw std::runtime_error("Error waiting for fences");
        }
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        createGraphicsPipeline();
        allocateCommandBuffers();
    }
}

void VulkanRenderer::cleanupSwapChain() {
    for (auto framebuffer : swapChainFramebuffers) {
        device.destroyFramebuffer(framebuffer);
    }

    device.freeCommandBuffers(commandPool, static_cast<uint32_t>(commandBuffers.size()), commandBuffers.data());

    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);

    for (auto imageView : depthImageViews) {
        device.destroyImageView(imageView);
    }
    for (auto image : depthImages) {
        device.destroyImage(image);
    }
    device.freeMemory(depthImageMemory);

    for (auto imageView : swapChainImageViews) {
        device.destroyImageView(imageView);
    }

    device.destroySwapchainKHR(swapChain);
}

void VulkanRenderer::cleanup() {
    for (auto observer : shaderObservers) {
        observer->Stop();
    }
    for (auto observer : shaderObservers) {
        observer->Join();
        delete observer;
    }
    shaderObservers.clear();
    cleanupSwapChain();

    device.destroyDescriptorPool(descriptorPool);
    device.destroyDescriptorSetLayout(descriptorSetLayout);

    device.unmapMemory(downloadMemory);
    downloadedDrawInfo = nullptr;
    for (size_t i = 0; i < drawOutputBuffers.size(); i++) {
        device.destroyBuffer(drawOutputBuffers[i]);
        device.freeMemory(drawOutputMemorys[i]);
    }
    device.destroyBuffer(downloadBuffer);
    device.freeMemory(downloadMemory);
    device.destroyBuffer(meshInfoBuffer);
    device.freeMemory(meshInfoMemory);
    drawDataOutput << std::flush;
    drawDataOutput.close();

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        device.destroySemaphore(renderFinishedSemaphores[i]);
        device.destroySemaphore(imageAvailableSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }

    device.destroyCommandPool(commandPool);

    device.destroy();

    if (enableValidationLayers) {
        instance.destroyDebugUtilsMessengerEXT(debugMessenger);
    }

    instance.destroySurfaceKHR(surface, nullptr);
    instance.destroy();

    glfwDestroyWindow(window);

    glfwTerminate();
}

void VulkanRenderer::recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    device.waitForFences(imagesInFlight, true, UINT64_MAX);

    cleanupSwapChain();

    createSwapChain();
    createImageViews();
    createDepthResources();
    createRenderPass();
    createGraphicsPipeline();
    createFramebuffers();
    allocateCommandBuffers();

    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
}

void VulkanRenderer::createInstance() {
    PFN_vkGetInstanceProcAddr func = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    vk::DebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (vk::DebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount = 0;

        createInfo.pNext = nullptr;
    }

    instance = vk::createInstance(createInfo);
    if (!instance) {
        throw std::runtime_error("failed to create instance!");
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
}

void VulkanRenderer::populateDebugMessengerCreateInfo(vk::DebugUtilsMessengerCreateInfoEXT& createInfo) {
    createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError; // vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose
    createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;
    createInfo.pfnUserCallback = debugCallback;
}

void VulkanRenderer::setupDebugMessenger() {
    if (!enableValidationLayers) return;

    vk::DebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    debugMessenger = instance.createDebugUtilsMessengerEXT(createInfo);
    if (!debugMessenger) {
        throw std::runtime_error("failed to set up debug messenger!");
    }
}

void VulkanRenderer::createSurface() {
    // TODO surface stuff sucks rn
    if (glfwCreateWindowSurface(instance, window, nullptr, (VkSurfaceKHR*)&surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

void VulkanRenderer::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    bool found = false;
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            found = true;
            break;
        }
    }

    if (!found) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void VulkanRenderer::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::DeviceCreateInfo createInfo{};

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceFeatures.fillModeNonSolid = true;
    vk::PhysicalDeviceFeatures2 deviceFeatures2;
    deviceFeatures2.features = deviceFeatures;
    vk::PhysicalDeviceVulkan11Features device11Features;
    device11Features.storageBuffer16BitAccess = true;
    vk::PhysicalDeviceMeshShaderFeaturesNV meshShaderFeatures;
    deviceFeatures2.pNext = &meshShaderFeatures;
    meshShaderFeatures.meshShader = true;
    meshShaderFeatures.taskShader = true;
    meshShaderFeatures.pNext = &device11Features;
    vk::PhysicalDeviceVulkan12Features device12Features;
    device12Features.storageBuffer8BitAccess = true;
    device12Features.shaderInt8 = true;
    device12Features.uniformAndStorageBuffer8BitAccess = true;
    device11Features.pNext = &device12Features;
    
    createInfo.pEnabledFeatures = nullptr;
    createInfo.pNext = &deviceFeatures2;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidationLayers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    device = physicalDevice.createDevice(createInfo);
    if (!device) {
        throw std::runtime_error("failed to create logical device!");
    }

    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    device.getQueue(indices.graphicsFamily.value(), 0, &graphicsQueue);
    device.getQueue(indices.presentFamily.value(), 0, &presentQueue);
}

void VulkanRenderer::createSwapChain() {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo{};
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    createInfo.presentMode = presentMode;
    createInfo.clipped = true;

    swapChain = device.createSwapchainKHR(createInfo);
    if (!swapChain) {
        throw std::runtime_error("failed to create swap chain!");
    }

    swapChainImages = device.getSwapchainImagesKHR(swapChain);

    swapChainImageFormat = surfaceFormat.format;
    swapChainExtent = extent;
}

void VulkanRenderer::createImageViews() {
    swapChainImageViews.resize(swapChainImages.size());

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        vk::ImageViewCreateInfo createInfo{};
        createInfo.image = swapChainImages[i];
        createInfo.viewType = vk::ImageViewType::e2D;
        createInfo.format = swapChainImageFormat;
        createInfo.components = {
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity
        };
        createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        createInfo.subresourceRange.baseMipLevel = 0;
        createInfo.subresourceRange.levelCount = 1;
        createInfo.subresourceRange.baseArrayLayer = 0;
        createInfo.subresourceRange.layerCount = 1;

        swapChainImageViews[i] = device.createImageView(createInfo);
        if (!swapChainImageViews[i]) {
            throw std::runtime_error("failed to create image views!");
        }
    }
}

vk::Format VulkanRenderer::findDepthFormat() {
    const std::vector<vk::Format> depthFormats {
        vk::Format::eD32Sfloat,
        vk::Format::eD32SfloatS8Uint,
        vk::Format::eD24UnormS8Uint
    };
    const auto tiling = vk::ImageTiling::eOptimal;
    const auto features = vk::FormatFeatureFlagBits::eDepthStencilAttachment;

    for (vk::Format format : depthFormats) {
        vk::FormatProperties props = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("ERROR: no proper depth formats were found");
}

void VulkanRenderer::createDepthResources() {
    depthImages.clear();
    vk::Format depthFormat = findDepthFormat();
    
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        vk::ImageCreateInfo imageCreateInfo;
        imageCreateInfo.imageType = vk::ImageType::e2D;
        imageCreateInfo.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;
        imageCreateInfo.tiling = vk::ImageTiling::eOptimal;
        imageCreateInfo.extent = vk::Extent3D(swapChainExtent);
        imageCreateInfo.extent.depth = 1;
        imageCreateInfo.mipLevels = 1;
        imageCreateInfo.arrayLayers = 1;
        imageCreateInfo.format = depthFormat;
        imageCreateInfo.initialLayout = vk::ImageLayout::eUndefined;
        imageCreateInfo.samples = vk::SampleCountFlagBits::e1;
        imageCreateInfo.sharingMode = vk::SharingMode::eExclusive;

        depthImages.push_back(device.createImage(imageCreateInfo));
    }

    auto memoryRequirements = device.getImageMemoryRequirements(depthImages.back());
    size_t imageMemorySize;
    vk::MemoryAllocateInfo allocInfo;
    allocInfo.allocationSize = memoryRequirements.size * 3;
    allocInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
    depthImageMemory = device.allocateMemory(allocInfo);
    for (size_t i = 0; i < swapChainImages.size(); i++) {
        device.bindImageMemory(depthImages[i], depthImageMemory, memoryRequirements.size * i);
    }

    depthImageViews.clear();

    vk::ImageViewCreateInfo viewInfo;
    viewInfo.viewType = vk::ImageViewType::e2D;
    viewInfo.components = {
        vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity
    };
    viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;
    viewInfo.format = depthFormat;
    for (const auto& i : depthImages) {
        viewInfo.image = i;
        depthImageViews.push_back(device.createImageView(viewInfo));
    }
}

void VulkanRenderer::createRenderPass() {
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainImageFormat;
    colorAttachment.samples = vk::SampleCountFlagBits::e1;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentDescription depthAttachment = colorAttachment;
    depthAttachment.format = findDepthFormat();
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentDescription attachments[]{
        colorAttachment,
        depthAttachment
    };

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.srcAccessMask = vk::AccessFlagBits::eNoneKHR;
    dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = ARRAYSIZE(attachments);
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    renderPass = device.createRenderPass(renderPassInfo);
    if (!renderPass) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void VulkanRenderer::createGraphicsPipeline() {
    std::vector<vk::PipelineShaderStageCreateInfo> stageCreateInfos;
    for (const auto& shaderInfo : shaderInfos) {
        auto shaderCode = readFile(shaderInfo.path);
        auto shaderModule = compileGlslFile(shaderInfo.name, {
            { "SUBDIVISIONS", std::to_string(SUBDIVISIONS) },
            { "LOCAL_SPACE_COUNT", std::to_string(LOCAL_SPACE_COUNT) },
            { "SCALE", std::to_string(SCALE) },
            { "DRAW_DATA_MEASURE", std::to_string(DRAW_DATA_MEASURE) }
        }, shaderInfo.stage, shaderCode, true, true);

        vk::PipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.stage = shaderInfo.stage;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";

        stageCreateInfos.push_back(shaderStageInfo);
    }

#if !MESH_SHADERS
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    inputAssembly.primitiveRestartEnable = false;
#endif

    vk::Viewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float)swapChainExtent.width;
    viewport.height = (float)swapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    vk::Rect2D scissor{};
    scissor.offset = vk::Offset2D { 0, 0 };
    scissor.extent = swapChainExtent;

    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = false;
    rasterizer.rasterizerDiscardEnable = false;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = vk::CullModeFlagBits::eNone;
    rasterizer.frontFace = vk::FrontFace::eClockwise;
    rasterizer.depthBiasEnable = false;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sampleShadingEnable = false;
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    colorBlendAttachment.blendEnable = false;

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = false;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;

    vk::PushConstantRange pushConstantRanges[]{
        {
            vk::ShaderStageFlagBits::eMeshNV | vk::ShaderStageFlagBits::eTaskNV,
            0,
            sizeof(PushConstantInfo)
        }
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = ARRAYSIZE(pushConstantRanges);
    pipelineLayoutInfo.pPushConstantRanges = pushConstantRanges;

    pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
    if (!pipelineLayout) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    vk::PipelineDepthStencilStateCreateInfo depthStencil;
    depthStencil.depthTestEnable = true;
    depthStencil.depthWriteEnable = true;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = false;
    depthStencil.stencilTestEnable = false;

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.stageCount = stageCreateInfos.size();
    pipelineInfo.pStages = stageCreateInfos.data();
#if !MESH_SHADERS
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
#endif
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.layout = pipelineLayout;
    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    auto pipelineCreateResult = device.createGraphicsPipeline(nullptr, pipelineInfo);
    graphicsPipeline = pipelineCreateResult.value;
    if (pipelineCreateResult.result != vk::Result::eSuccess) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    for (const auto& thing : stageCreateInfos) {
        device.destroyShaderModule(thing.module);
    }
}

void VulkanRenderer::createDescriptorSetLayout(const std::vector<DescriptorSetLayoutBinding>& layouts) {
    std::vector <vk::DescriptorSetLayoutBinding> layoutBindings;

    int i = 0;
    for (const auto& layout : layouts) {
        layoutBindings.emplace_back(i++, layout.type, 1, layout.stages);
    }

    vk::DescriptorSetLayoutCreateInfo layoutInfo;
    layoutInfo.bindingCount = layoutBindings.size();
    layoutInfo.pBindings = layoutBindings.data();

    descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
}

void VulkanRenderer::createDescriptorPool(size_t descriptorCount) {
    vk::DescriptorPoolSize poolSize;
    poolSize.descriptorCount = descriptorCount;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = swapChainImages.size();

    descriptorPool = device.createDescriptorPool(poolInfo);
}

void VulkanRenderer::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = swapChainImages.size();
    allocInfo.pSetLayouts = layouts.data();

    auto sets = device.allocateDescriptorSets(allocInfo);
    if (sets.empty()) {
        throw std::runtime_error("error allocating descriptor sets");
    }

    descriptorSets = sets;

    vk::DescriptorBufferInfo miBufferInfo;
    miBufferInfo = meshInfoBuffer;
    miBufferInfo.offset = 0;
    miBufferInfo.range = sizeof(MeshInfo);

    for (size_t i = 0; i < descriptorSets.size(); i++) {
        vk::DescriptorBufferInfo outBufferInfo;
        outBufferInfo = drawOutputBuffers[i];
        outBufferInfo.offset = 0;
        outBufferInfo.range = sizeof(DrawOutputInfo);

        std::vector<vk::WriteDescriptorSet> descriptorWrites(2);
        descriptorWrites[0].dstSet = descriptorSets[i];
        descriptorWrites[0].pBufferInfo = &miBufferInfo;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = vk::DescriptorType::eStorageBuffer;

        descriptorWrites[1].dstSet = descriptorSets[i];
        descriptorWrites[1].pBufferInfo = &outBufferInfo;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    
        device.updateDescriptorSets(descriptorWrites, {});
    }
}

void VulkanRenderer::createFramebuffers() {
    swapChainFramebuffers.clear();

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vk::ImageView attachments[] = {
            swapChainImageViews[i],
            depthImageViews[i]
        };

        vk::FramebufferCreateInfo framebufferInfo{};
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = ARRAYSIZE(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;

        swapChainFramebuffers.push_back(device.createFramebuffer(framebufferInfo));
        if (!swapChainFramebuffers.back()) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanRenderer::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

    commandPool = device.createCommandPool(poolInfo);
    if (!commandPool) {
        throw std::runtime_error("failed to create graphics command pool!");
    }
}

void VulkanRenderer::createMeshInfo() {
    // Vertex buffer
    vk::DeviceSize meshInfoBufferSize = sizeof(MeshInfo);
    vk::DeviceSize stagingBufferSize = meshInfoBufferSize;

    vk::Buffer stagingBuffer;
    vk::DeviceMemory stagingBufferMemory;
    createBuffer(stagingBufferSize, vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent, stagingBuffer, stagingBufferMemory);
    void* data = device.mapMemory(stagingBufferMemory, 0, stagingBufferSize, vk::MemoryMapFlags{ 0 }); // MAP

    // Mesh info
    auto meshInfo = std::make_unique<MeshInfo>();
    meshInfo->sunlightDir = SUNLIGHT_DIR;
    meshInfo->scale = SCALE;
    meshInfo->subdivisions = SUBDIVISIONS;
    std::vector<uint8_t> p;
    p.resize(256);
    std::iota(p.begin(), p.end(), 0);
#if DEFAULT_NOISE_PBOX
    std::default_random_engine engine(0);
#else
    std::default_random_engine engine(time(NULL));
#endif
    std::shuffle(p.begin(), p.end(), engine);
    p.insert(p.end(), p.begin(), p.end());
    for (size_t i = 0; i < 512; i++) {
        meshInfo->pBox12[i] = p[i] % 12;
    }
    memcpy(meshInfo->pBox, p.data(), sizeof(meshInfo->pBox));
#if BLOAT
    for (size_t i = 0; i < BLOAT_FACTOR; i++) {
        memcpy(meshInfo->pBox + p.size() * i, p.data(), p.size() * sizeof(p[0]));
    }
#endif

#if CELL_INDICES_ENCODE
    // 0x20 = on right cell
    // 0x40 = on bottom cell
    // 0x80 = on front cell
    uint8_t configPermute[12] = {
        1,
        0 | 0x20,
        1 | 0x80,
        0,
        1 | 0x40,
        0 | 0x20 | 0x40,
        1 | 0x80 | 0x40,
        0 | 0x40,
        2,
        2 | 0x20,
        2 | 0x80 | 0x20,
        2 | 0x80
    };
    for (size_t i = 0; i < ARRAYSIZE(cellConfigurations); i++) {
        for (size_t j = 0; j < ARRAYSIZE(cellConfigurations[i]); j++) {
            int8_t v = cellConfigurations[i][j];
            if (v != -1)
                meshInfo.cellConfigs[i][j] = configPermute[v];
            else
                meshInfo.cellConfigs[i][j] = -1;
        }
    }
#else
    memcpy(meshInfo->cellConfigs, cellConfigurations, sizeof(cellConfigurations));
#endif
    // Bottom
    meshInfo->edgePositions[0]  = glm::vec4( 0.f , -0.5f, -0.5f, 1.f);
    meshInfo->edgePositions[1]  = glm::vec4( 0.5f, -0.5f,  0.f , 1.f); // here
    meshInfo->edgePositions[2]  = glm::vec4( 0.f , -0.5f,  0.5f, 1.f);
    meshInfo->edgePositions[3]  = glm::vec4(-0.5f, -0.5f,  0.f , 1.f); // here
    // Top
    meshInfo->edgePositions[4]  = glm::vec4( 0.f ,  0.5f, -0.5f, 1.f);
    meshInfo->edgePositions[5]  = glm::vec4( 0.5f,  0.5f,  0.f , 1.f);
    meshInfo->edgePositions[6]  = glm::vec4( 0.f ,  0.5f,  0.5f, 1.f);
    meshInfo->edgePositions[7]  = glm::vec4(-0.5f,  0.5f,  0.f , 1.f);
    // Middle
    meshInfo->edgePositions[8]  = glm::vec4(-0.5f,  0.f , -0.5f, 1.f);
    meshInfo->edgePositions[9]  = glm::vec4( 0.5f,  0.f , -0.5f, 1.f);
    meshInfo->edgePositions[10] = glm::vec4( 0.5f,  0.f ,  0.5f, 1.f);
    meshInfo->edgePositions[11] = glm::vec4(-0.5f,  0.f ,  0.5f, 1.f);

    memcpy(data, meshInfo.get(), meshInfoBufferSize);
    createBuffer(meshInfoBufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eDeviceLocal, meshInfoBuffer, meshInfoMemory);
    copyBuffer(stagingBuffer, meshInfoBuffer, meshInfoBufferSize);

    device.unmapMemory(stagingBufferMemory); // UNMAP
    device.destroyBuffer(stagingBuffer);
    device.freeMemory(stagingBufferMemory);
}

void VulkanRenderer::createOutputBuffers() {
    vk::DeviceSize bufferSize = sizeof(DrawOutputInfo);

    drawOutputBuffers.resize(swapChainImages.size());
    drawOutputMemorys.resize(swapChainImages.size());

    for (size_t i = 0; i < drawOutputBuffers.size(); i++) {
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, vk::MemoryPropertyFlagBits::eDeviceLocal, drawOutputBuffers[i], drawOutputMemorys[i]);
    }
}

void VulkanRenderer::createDownloadResources() {
    vk::DeviceSize bufferSize = sizeof(DrawOutputInfo);
    createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer, vk::MemoryPropertyFlagBits::eHostCoherent, downloadBuffer, downloadMemory);
    downloadedDrawInfo = (DrawOutputInfo*)device.mapMemory(downloadMemory, 0, sizeof(DrawOutputInfo));
}

void VulkanRenderer::createDrawInfoFile() {
    drawDataOutput = std::ofstream("out.csv");
    drawDataOutput << "frameno,tp-tri-count,fn-tri-count,true-positive,false-positive,false-negative,true-negative" << std::endl;
}

void VulkanRenderer::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& bufferMemory) {
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;

    buffer = device.createBuffer(bufferInfo);
    if (!buffer) {
        throw std::runtime_error("failed to create buffer!");
    }

    vk::MemoryRequirements memRequirements;
    memRequirements = device.getBufferMemoryRequirements(buffer);

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    bufferMemory = device.allocateMemory(allocInfo);
    if (!bufferMemory) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    device.bindBufferMemory(buffer, bufferMemory, 0);
}

void VulkanRenderer::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::FenceCreateInfo fenceCreateInfo;
    vk::Fence fence = device.createFence(fenceCreateInfo);

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    vk::CommandBuffer commandBuffer;
    commandBuffer = device.allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion{};
    copyRegion.size = size;
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);

    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    graphicsQueue.submit(1, &submitInfo, fence);
    device.waitForFences(1, &fence, true, UINT64_MAX);

    device.destroyFence(fence);

    device.freeCommandBuffers(commandPool, 1, &commandBuffer);
}

uint32_t VulkanRenderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties;
    memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void VulkanRenderer::allocateCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = (uint32_t)swapChainFramebuffers.size();

    commandBuffers = device.allocateCommandBuffers(allocInfo);
    if (commandBuffers.empty()) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

void VulkanRenderer::recordCommandBuffer(uint32_t index) {
    auto& commandBuffer = commandBuffers[index];
    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

    commandBuffer.begin(beginInfo);
    if (0) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    vk::RenderPassBeginInfo renderPassInfo{};
    renderPassInfo.renderPass = renderPass;
    renderPassInfo.framebuffer = swapChainFramebuffers[index];
    renderPassInfo.renderArea.offset = vk::Offset2D{ 0, 0 };
    renderPassInfo.renderArea.extent = swapChainExtent;

    vk::ClearValue clearValues[2];
    clearValues[0].color.setFloat32({ 0.0f, 0.0f, 0.0f, 0.0f });
    clearValues[1].depthStencil.depth = 1.0f;
    clearValues[1].depthStencil.stencil = 0;
    renderPassInfo.clearValueCount = ARRAYSIZE(clearValues);
    renderPassInfo.pClearValues = clearValues;

    commandBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, { descriptorSets[index] }, {});
    commandBuffer.pushConstants(
        pipelineLayout,
        vk::ShaderStageFlagBits::eMeshNV | vk::ShaderStageFlagBits::eTaskNV,
        0,
        sizeof(PushConstantInfo),
        &pushConstantInfo
    );
    if (performanceMarkers) {
        vk::DebugUtilsLabelEXT labelInfo;
        labelInfo
            .setPLabelName("Drawing total area")
            .setColor({ 1.0f, 0.5f, 0.25f, 1.0f }); // array wrappers to set colour wants the builder pattern :/
        commandBuffer.beginDebugUtilsLabelEXT(labelInfo);
    }
#if MESH_SHADERS
    // Staggered Dispatch
#if 0
    for (size_t i = 0; i < std::ceil(std::pow(SUBDIVISIONS / 12, 3) / 64); i++) {
        size_t tasksToLaunch = std::min((int)(std::pow(SUBDIVISIONS / 12, 3) - i * 64), 64);
        commandBuffers[index].drawMeshTasksNV(tasksToLaunch, i * 64);
    }
#else
    commandBuffer.drawMeshTasksNV(std::pow(SUBDIVISIONS / 8, 3), 0);
#endif
#else
    vk::Buffer vertexBuffers[] = { vertexBuffer };
    vk::DeviceSize offsets[] = { 0 };
    commandBuffers[index].bindVertexBuffers(0, 1, vertexBuffers, offsets);

    commandBuffers[index].bindIndexBuffer(indexBuffer, 0, VK_INDEX_TYPE_UINT16);

    commandBuffers[index].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
#endif
    if (performanceMarkers)
        commandBuffer.endDebugUtilsLabelEXT();

    commandBuffer.endRenderPass();

    commandBuffer.end();
    if (0) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void VulkanRenderer::createSyncObjects() {
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

    vk::SemaphoreCreateInfo semaphoreInfo{};

    vk::FenceCreateInfo fenceInfo{};
    fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
        renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
        inFlightFences[i] = device.createFence(fenceInfo);
    }
}

void VulkanRenderer::createShaderObservers() {
    shaderObservers.reserve(shaderInfos.size());
    for (const auto& shader : shaderInfos) {
        shaderObservers.push_back(new FileObserver(shader.path, std::chrono::milliseconds(1000)));
    }
}

void VulkanRenderer::handleEvents() {
    using namespace std::chrono;
    running = !glfwWindowShouldClose(window) && GLFW_PRESS != glfwGetKey(window, GLFW_KEY_ESCAPE);
#if TIMER_ENABLE
#if !DRAW_DATA_MEASURE
    running = running && duration_cast<duration<double>>(high_resolution_clock::now() - startTime).count() < 60;
#else
    running = running && frameCount <= FRAME_SAMPLE_COUNT;
#endif
#endif

    double mouseX, mouseY;
    const double sensitivity = 0.001f;
    glfwGetCursorPos(window, &mouseX, &mouseY);
    if (frameCount == 0) {
        gameState.mouseX = mouseX;
        gameState.mouseY = mouseY;
    }
    gameState.rotate.x += (mouseX - gameState.mouseX) * sensitivity;
    gameState.rotate.y -= (mouseY - gameState.mouseY) * sensitivity;
    gameState.mouseX = mouseX;
    gameState.mouseY = mouseY;
    gameState.rotate.y = std::clamp(gameState.rotate.y, -1.3f, 1.3f);

    const auto up = glm::vec3(0.f, -1.f, 0.f);
    glm::vec3 target = glm::vec3(
        sin(gameState.rotate.x) * cos(gameState.rotate.y),
        sin(gameState.rotate.y),
        cos(gameState.rotate.x) * cos(gameState.rotate.y)
    );

    double now = glfwGetTime();
    double deltaTime = now - gameState.time;
    gameState.time = now;

    double speed = 5.f;
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)) {
        speed *= 4;
    }

    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_W)) {
        gameState.pos += glm::vec3(deltaTime) * target * glm::vec3(speed);
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_A)) {
        gameState.pos += glm::vec3(deltaTime) * -glm::normalize(glm::cross(target, up)) * glm::vec3(speed);
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_S)) {
        gameState.pos += glm::vec3(deltaTime) * -target * glm::vec3(speed);
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_D)) {
        gameState.pos += glm::vec3(deltaTime) * glm::normalize(glm::cross(target, up)) * glm::vec3(speed);
    }
    if (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_R)) {
        gameState.pos = STARTING_POS;
        gameState.rotate = STARTING_ROT;
    }

    pushConstantInfo.proj = glm::perspective(glm::radians(75.f), (float)WIDTH/HEIGHT, 0.01f, 1000.f);
    pushConstantInfo.camera = glm::lookAtRH(gameState.pos, gameState.pos + target, up);
#if !DRAW_DATA_MEASURE
    pushConstantInfo.time = gameState.time * TIME_SCALE;
#else
    pushConstantInfo.time = 60.f * frameCount / FRAME_SAMPLE_COUNT;
#endif

    /*
    std::cout << "Time: " << deltaTime << std::endl;
    std::cout << "Rotate: " << gameState.rotate.x << ", " << gameState.rotate.y << ", " << gameState.rotate.z << std::endl;
    std::cout << "Pos: " << gameState.pos.x << ", " << gameState.pos.y << ", " << gameState.pos.z << std::endl;
    std::cout << std::endl;
    */
}

void VulkanRenderer::downloadDrawData(uint32_t imageIndex) {
    copyBuffer(drawOutputBuffers[imageIndex], downloadBuffer, sizeof(DrawOutputInfo));
}

void VulkanRenderer::drawFrame() {
    device.waitForFences(1, &inFlightFences[currentFrame], true, UINT64_MAX);
    if (DRAW_DATA_MEASURE && lastFrame != -1) {
        downloadDrawData(lastFrame);
        int falseNegative = 0,
            falsePositive = 0,
            trueNegative = 0,
            truePositive = 0,
            tpTriCount = 0,
            fnTriCount = 0,
            maxTriCount = 0;
        for (size_t i = 0; i < LOCAL_SPACE_COUNT; i++) {
            // IF positive (should've drawn)
            if (downloadedDrawInfo->localSpaceShouldDraw[i / 32] & 1 << (i % 32)) {
                tpTriCount += downloadedDrawInfo->localSpaceTriCounts[i];
                // IF actually did draw
                if (downloadedDrawInfo->localSpaceTriCounts[i])
                    truePositive++;
                else
                    falsePositive++;
            }
            else {
                fnTriCount += downloadedDrawInfo->localSpaceTriCounts[i];
                if (!downloadedDrawInfo->localSpaceTriCounts[i])
                    trueNegative++;
                else
                    falseNegative++;
            }
            maxTriCount = std::max(maxTriCount, (int)downloadedDrawInfo->localSpaceTriCounts[i]);
        }

        drawDataOutput << frameCount << ',' << tpTriCount << ',' << fnTriCount << ',' << truePositive << ',' << falsePositive << ',' << falseNegative << ',' << trueNegative << ',' << maxTriCount << std::endl;
    }

    auto acquireResult = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], nullptr);
    uint32_t imageIndex = acquireResult.value;
    lastFrame = imageIndex;
    frameCount++;

    if (acquireResult.result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain();
        return;
    }
    else if (acquireResult.result != vk::Result::eSuccess && acquireResult.result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    if ((VkFence)imagesInFlight[imageIndex] != 0) {
        auto res = device.waitForFences(1, &imagesInFlight[imageIndex], true, UINT64_MAX);
        if (res != vk::Result::eSuccess) {
            throw std::runtime_error("failed to wait for fences: " + vk::to_string(res));
        }
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    recordCommandBuffer(imageIndex);

    vk::SubmitInfo submitInfo{};

    vk::Semaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = signalSemaphores;

    vk::Result res = device.resetFences(1, &inFlightFences[currentFrame]);
    if (res != vk::Result::eSuccess) {
        const auto message = "failed to reset fences: " + vk::to_string(res);
        throw std::runtime_error(message);
    }
    res = graphicsQueue.submit(1, &submitInfo, inFlightFences[currentFrame]);
    if (res != vk::Result::eSuccess) {
        const auto message = "failed to submit draw command buffer: " + vk::to_string(res);
        throw std::runtime_error(message);
    }

    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = signalSemaphores;

    vk::SwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    try {
        auto presentResult = presentQueue.presentKHR(presentInfo);
        if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (presentResult != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }
    }
    catch (const vk::OutOfDateKHRError& e) {
        framebufferResized = false;
        recreateSwapChain();
    }
    catch (const vk::IncompatibleDisplayKHRError& e) {
        framebufferResized = false;
        recreateSwapChain();
    }
    catch (const std::exception& e) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
}

vk::ShaderModule VulkanRenderer::createShaderModule(const std::vector<char>& code) {
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    vk::ShaderModule shaderModule = device.createShaderModule(createInfo);
    if (!shaderModule) {
        throw std::runtime_error("failed to create shader module!");
    }

    return shaderModule;
}

vk::SurfaceFormatKHR VulkanRenderer::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (/* availableFormat.format == vk::Format::eB8G8R8A8Srgb && */ availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

vk::PresentModeKHR VulkanRenderer::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
            return availablePresentMode;
        }
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eMailbox;
}

vk::Extent2D VulkanRenderer::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);

        vk::Extent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };

        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

        return actualExtent;
    }
}

SwapChainSupportDetails VulkanRenderer::querySwapChainSupport(vk::PhysicalDevice device) {
    SwapChainSupportDetails details;

    // This is silly, why is everything seperate
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);

    return details;
}

bool VulkanRenderer::isDeviceSuitable(vk::PhysicalDevice device) {
    QueueFamilyIndices indices = findQueueFamilies(device);

    vk::PhysicalDeviceProperties2 props2;
    vk::PhysicalDeviceMeshShaderPropertiesNV meshProps;
    props2.pNext = &meshProps;

    device.getProperties2(&props2);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if (extensionsSupported) {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

bool VulkanRenderer::checkDeviceExtensionSupport(vk::PhysicalDevice device) {
    std::vector<vk::ExtensionProperties> availableExtensions;
    availableExtensions = device.enumerateDeviceExtensionProperties();

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

QueueFamilyIndices VulkanRenderer::findQueueFamilies(vk::PhysicalDevice device) {
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies;
    queueFamilies = device.getQueueFamilyProperties();

    int i = 0;
    for (const auto& queueFamily : queueFamilies) {
        if (queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }

        if (device.getSurfaceSupportKHR(i, surface)) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) {
            break;
        }

        i++;
    }

    return indices;
}

std::vector<const char*> VulkanRenderer::getRequiredExtensions() {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    if (enableValidationLayers || performanceMarkers) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    return extensions;
}

bool VulkanRenderer::checkValidationLayerSupport() {
    std::vector<vk::LayerProperties> availableLayers;
    availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : validationLayers) {
        bool layerFound = false;

        for (const auto& layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}

std::string VulkanRenderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }

    size_t fileSize = (size_t)file.tellg();
    // old stuff that returns a vector
    /*
    std::vector<char> buffer;
    buffer.reserve(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    */
    file.seekg(0);
    std::string buffer( (std::istreambuf_iterator<char>(file)),
        (std::istreambuf_iterator<char>()) );

    return buffer;
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanRenderer::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    // Select prefix depending on flags passed to the callback
    std::string prefix("");

    if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
        prefix = "VERBOSE: ";
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        prefix = "INFO: ";
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        prefix = "WARNING: ";
    }
    else if (messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        prefix = "ERROR: ";
    }


    // Display message to default output (console/logcat)
    std::stringstream debugMessage;
    debugMessage << prefix << "[" << pCallbackData->messageIdNumber << "][" << pCallbackData->pMessageIdName << "] : " << pCallbackData->pMessage;

#if defined(__ANDROID__)
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        LOGE("%s", debugMessage.str().c_str());
    }
    else {
        LOGD("%s", debugMessage.str().c_str());
    }
#else
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        std::cerr << debugMessage.str() << "\n";
    }
    else {
        std::cout << debugMessage.str() << "\n";
    }
    fflush(stdout);
#endif


    // The return value of this callback controls whether the Vulkan call that caused the validation message will be aborted or not
    // We return VK_FALSE as we DON'T want Vulkan calls that cause a validation message to abort
    // If you instead want to have calls abort, pass in VK_TRUE and the function will return VK_ERROR_VALIDATION_FAILED_EXT 
    return VK_FALSE;
}
