#pragma once
#include <vulkan.hpp>
#include <vulkan_ext.h>
#include <fstream>

namespace mvk
{
    inline std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        return buffer;
    }

    inline vk::UniqueShaderModule loadShaderFromFile(vk::Device device, const std::string& path)
    {
        auto code = readFile(path);
        vk::ShaderModuleCreateInfo createInfo = {};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        return device.createShaderModuleUnique(createInfo);
    }

}