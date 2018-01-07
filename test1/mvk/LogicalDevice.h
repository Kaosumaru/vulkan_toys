#pragma once

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan.hpp>
#include <vulkan_ext.h>
#include <vector>

#include "ValidationLayer.h"

namespace mvk
{
    class LogicalDevice
    {
        const std::vector<const char*> deviceExtensions = {
            VK_KHR_SWAPCHAIN_EXTENSION_NAME
        };
    public:
        LogicalDevice(GLFWwindow* window) : _window(window) { init(); };
        ~LogicalDevice() {  };

        struct QueueFamilyIndices {
            int graphicsFamily = -1;
            int presentFamily = -1;

            bool isComplete() const { return graphicsFamily >= 0 && presentFamily >= 0; }
        };

        vk::Device device() { return *_device; }
        vk::SurfaceKHR surface() { return *_surface; }
        vk::PhysicalDevice physicalDevice() { return _physicalDevice; }
        const QueueFamilyIndices& indices() { return _indices; }

        uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    protected:
        void init() 
        {
            createInstance();
            _validationLayersManager.setupDebugCallback(*_instance);
            createSurface();
            pickPhysicalDevice();
            createLogicalDevice();
        }

        void createInstance();
        void createSurface();
        void pickPhysicalDevice();
        void createLogicalDevice();

        std::vector<const char*> getRequiredExtensions();
        QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice physicalDevice);


        vk::UniqueInstance _instance;
        mvk::ValidationLayers _validationLayersManager;

        vk::PhysicalDevice _physicalDevice;
        vk::PhysicalDeviceMemoryProperties _memoryProperties;
        QueueFamilyIndices _indices;

        vk::UniqueSurfaceKHR _surface;
        vk::UniqueDevice _device;

        GLFWwindow* _window;
    };
}