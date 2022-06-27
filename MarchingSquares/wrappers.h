#ifndef WRAPPERS_H
#define WRAPPERS_H

#include <string>
#include <vulkan/vulkan.hpp>

struct DescriptorSetLayoutBinding {
	vk::DescriptorType type;
	vk::ShaderStageFlags stages;
	vk::DeviceSize size;
};

struct ShadercMacro {
	std::string key, value;
};

struct ShaderInfo {
	std::string name;
	std::string path;
	vk::ShaderStageFlagBits stage;
};

#endif
