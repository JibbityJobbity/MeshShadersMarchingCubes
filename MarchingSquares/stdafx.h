#pragma once

#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#define EOL '\n'
#elif __linux__
#define VK_USE_PLATFORM_XLIB_KHR
#define EOL '\n'
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

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <shaderc/shaderc.hpp>
#include <cstdint>
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
