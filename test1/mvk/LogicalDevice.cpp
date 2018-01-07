#include "LogicalDevice.h"
#include "SwapChain.h"
#include <set>

using namespace mvk;

void LogicalDevice::createInstance() 
{
    if (!_validationLayersManager.isOK()) 
    {
        throw std::runtime_error("validation layers requested, but not available!");
    }

    vk::ApplicationInfo appInfo = {};
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    vk::InstanceCreateInfo createInfo = {};
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    _validationLayersManager.AddToCreateInfo(createInfo);

    _instance = vk::createInstanceUnique(createInfo);
    vkExtInitInstance(*_instance);
}

void LogicalDevice::createSurface() 
{
    VkSurfaceKHR surf;
    if (glfwCreateWindowSurface(*_instance, _window, nullptr, &surf) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create window surface!");
    }
    vk::SurfaceKHRDeleter deleter(*_instance);
    _surface = vk::UniqueSurfaceKHR(surf, deleter);
}

void LogicalDevice::pickPhysicalDevice() 
{
    auto devices = _instance->enumeratePhysicalDevices();

    if (devices.size() == 0) 
    {
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    auto checkDeviceExtensionSupport = [&](vk::PhysicalDevice device) 
    {
        auto availableExtensions = device.enumerateDeviceExtensionProperties();

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) 
            requiredExtensions.erase(extension.extensionName);

        return requiredExtensions.empty();
    };

    auto isDeviceSuitable = [&](vk::PhysicalDevice physicalDevice) 
    {
        bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);
        if (!extensionsSupported) return false;

        mvk::SwapChainSupportDetails swapChainSupport {physicalDevice, *_surface};
        if (!swapChainSupport.valid()) return false;

        return findQueueFamilies(physicalDevice).isComplete();
    };

    for (const auto& device : devices) 
    {
        if (isDeviceSuitable(device)) 
        {
            _physicalDevice = device;
            break;
        }
    }

    if (!_physicalDevice) 
    {
        throw std::runtime_error("failed to find a suitable GPU!");
    }

    _memoryProperties = _physicalDevice.getMemoryProperties();
}

LogicalDevice::QueueFamilyIndices LogicalDevice::findQueueFamilies(vk::PhysicalDevice physicalDevice)
{
    QueueFamilyIndices indices;

    auto queueFamilies = physicalDevice.getQueueFamilyProperties();

    int i = 0;
    for (const auto& queueFamily : queueFamilies) 
    {
        if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) 
        {
            indices.graphicsFamily = i;
        }

        auto presentSupport = physicalDevice.getSurfaceSupportKHR(i, *_surface);

        if (queueFamily.queueCount > 0 && presentSupport) 
        {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) break;
        i++;
    }
    return indices;
}

uint32_t LogicalDevice::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
{
    for (uint32_t i = 0; i < _memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (_memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return -1;
}

void LogicalDevice::createLogicalDevice() 
{
    QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
    _indices = indices;

    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

    float queuePriority = 1.0f;
    for (int queueFamily : uniqueQueueFamilies) 
    {
        vk::DeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    vk::PhysicalDeviceFeatures deviceFeatures = {};

    vk::DeviceCreateInfo createInfo = {};

    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    _validationLayersManager.AddToCreateInfo(createInfo);

    _device = _physicalDevice.createDeviceUnique(createInfo);
}

std::vector<const char*> LogicalDevice::getRequiredExtensions() 
{
    std::vector<const char*> extensions;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    for (uint32_t i = 0; i < glfwExtensionCount; i++) {
        extensions.push_back(glfwExtensions[i]);
    }

    _validationLayersManager.AddRequiredExtension(extensions);



    return extensions;
}