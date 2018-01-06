#pragma once

#include <vulkan.hpp>
#include <vulkan_ext.h>

namespace mvk
{
    class LogicalDeviceManager
    {
    public:
    protected:
        vk::PhysicalDevice physicalDevice;
        vk::UniqueSurfaceKHR surface;
        vk::UniqueDevice device;
    };
}