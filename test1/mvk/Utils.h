#pragma once
#include <vulkan.hpp>
#include <vulkan_ext.h>

namespace mvk
{
    class ValidationLayers
    {
    public:
#ifdef NDEBUG
        const bool enableValidationLayers = false;
#else
        const bool enableValidationLayers = true;
#endif

        void setupDebugCallback(vk::Instance& instance)
        {
            if (!enableValidationLayers) return;

            VkDebugReportCallbackCreateInfoEXT createInfo = {};
            createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
            createInfo.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
            createInfo.pfnCallback = debugCallback;

            callback = instance.createDebugReportCallbackEXTUnique(createInfo);
        }

        bool isOK()
        {
            return !enableValidationLayers || checkValidationLayerSupport();
        }


        template<typename T>
        void AddToCreateInfo(T& createInfo)
        {
            if (enableValidationLayers) {
                createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
                createInfo.ppEnabledLayerNames = validationLayers.data();
            }
            else {
                createInfo.enabledLayerCount = 0;
            }
        }

        void AddRequiredExtension(std::vector<const char*>& extensions)
        {
            if (enableValidationLayers) {
                extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
            }
        }
    private:
        const std::vector<const char*> validationLayers = {
            "VK_LAYER_LUNARG_standard_validation"
        };

        bool checkValidationLayerSupport() {
            auto availableLayers = vk::enumerateInstanceLayerProperties();

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

        static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
            VkDebugReportFlagsEXT flags,
            VkDebugReportObjectTypeEXT objType,
            uint64_t obj,
            size_t location,
            int32_t code,
            const char* layerPrefix,
            const char* msg,
            void* userData) {

            std::cerr << "validation layer: " << msg << std::endl;

            return VK_FALSE;
        }


        vk::UniqueDebugReportCallbackEXT callback;
    };

    struct SwapChainSupportDetails
    {
        SwapChainSupportDetails() {}
        SwapChainSupportDetails(vk::PhysicalDevice device, vk::SurfaceKHR surface)
        {
            capabilities = device.getSurfaceCapabilitiesKHR(surface);
            formats = device.getSurfaceFormatsKHR(surface);
            presentModes = device.getSurfacePresentModesKHR(surface);
        }

        bool valid() { return !formats.empty() && !presentModes.empty(); }

        vk::SurfaceFormatKHR chooseStandardSwapSurfaceFormat()
        {
            if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
                vk::SurfaceFormatKHR format;
                format.format = vk::Format::eB8G8R8A8Unorm;
                format.colorSpace = vk::ColorSpaceKHR::eSrgbNonlinear;
                return format;
            }

            for (const auto& availableFormat : formats) {
                if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                    return availableFormat;
                }
            }

            return formats[0];
        }

        vk::PresentModeKHR chooseStandardSwapPresentMode() {
            vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

            for (const auto& availablePresentMode : presentModes) {
                if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                    return availablePresentMode;
                }
                else if (availablePresentMode == vk::PresentModeKHR::eImmediate) {
                    bestMode = availablePresentMode;
                }
            }

            return bestMode;
        }

        vk::Extent2D chooseSwapExtent(uint32_t width, uint32_t height) {
            if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
                return capabilities.currentExtent;
            }
            else {
                vk::Extent2D actualExtent = {width, height};

                actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);


                return actualExtent;
            }
        }

        vk::SurfaceCapabilitiesKHR capabilities;
        std::vector<vk::SurfaceFormatKHR> formats;
        std::vector<vk::PresentModeKHR> presentModes;
    };

    struct SwapChainManager
    {
    public:
        SwapChainManager() {}
        ~SwapChainManager()
        {
            cleanup();
        }

        bool create(vk::Device device, vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface, uint32_t graphicsFamily, uint32_t presentFamily)
        {
            _device = device;
            _physicalDevice = physicalDevice;
            _surface = surface;
            _graphicsFamily = graphicsFamily;
            _presentFamily = presentFamily;
            if (!createSwapChain()) return false;
            createImageViews();

            return true;
        }

        void cleanup()
        {
            if (!valid()) return;

            _swapChainImageViews.clear();
            _swapChain.reset();
        }

        bool valid() { return _swapChain.get(); }

        const auto& swapChain() { return _swapChain; }
        const auto& imageViews() { return _swapChainImageViews; }
        vk::Format imageFormat() { return _swapChainImageFormat; }
        vk::Extent2D extent() { return _swapChainExtent; }
    private:
        bool createSwapChain() {
            mvk::SwapChainSupportDetails swapChainSupport {_physicalDevice, _surface};

            auto surfaceFormat = swapChainSupport.chooseStandardSwapSurfaceFormat();
            auto presentMode = swapChainSupport.chooseStandardSwapPresentMode();
            vk::Extent2D extent = swapChainSupport.chooseSwapExtent(WIDTH, HEIGHT);

            if (extent.width == 0 || extent.height == 0) return false;

            uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
            if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
                imageCount = swapChainSupport.capabilities.maxImageCount;
            }

            vk::SwapchainCreateInfoKHR createInfo = {};
            createInfo.surface = _surface;
            createInfo.minImageCount = imageCount;
            createInfo.imageFormat = surfaceFormat.format;
            createInfo.imageColorSpace = surfaceFormat.colorSpace;
            createInfo.imageExtent = extent;
            createInfo.imageArrayLayers = 1;
            createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

            uint32_t queueFamilyIndices[] = {_graphicsFamily, _presentFamily};

            //check if queues are same, so imageSharingMode could be disabled
            if (_graphicsFamily != _presentFamily) {
                createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
                createInfo.queueFamilyIndexCount = 2;
                createInfo.pQueueFamilyIndices = queueFamilyIndices;
            }
            else {
                createInfo.imageSharingMode = vk::SharingMode::eExclusive;
                createInfo.queueFamilyIndexCount = 0; // Optional
                createInfo.pQueueFamilyIndices = nullptr; // Optional
            }
            createInfo.preTransform = swapChainSupport.capabilities.currentTransform;

            createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
            createInfo.presentMode = presentMode;
            createInfo.clipped = VK_TRUE;
            createInfo.oldSwapchain = nullptr;

            _swapChain = _device.createSwapchainKHRUnique(createInfo);

            _swapChainImages = _device.getSwapchainImagesKHR(_swapChain.get()); //TODO
            _swapChainImageFormat = surfaceFormat.format;
            _swapChainExtent = extent;
            return true;
        }

        void createImageViews() {
            _swapChainImageViews.resize(_swapChainImages.size());
            for (size_t i = 0; i < _swapChainImages.size(); i++) {
                vk::ImageViewCreateInfo createInfo = {};
                createInfo.image = _swapChainImages[i];
                createInfo.viewType = vk::ImageViewType::e2D;
                createInfo.format = _swapChainImageFormat;
                createInfo.components.r = vk::ComponentSwizzle::eIdentity;
                createInfo.components.g = vk::ComponentSwizzle::eIdentity;
                createInfo.components.b = vk::ComponentSwizzle::eIdentity;
                createInfo.components.a = vk::ComponentSwizzle::eIdentity;
                createInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
                createInfo.subresourceRange.baseMipLevel = 0;
                createInfo.subresourceRange.levelCount = 1;
                createInfo.subresourceRange.baseArrayLayer = 0;
                createInfo.subresourceRange.layerCount = 1;

                _swapChainImageViews[i] = _device.createImageViewUnique(createInfo);
            }
        }

        vk::Device _device = nullptr;
        vk::PhysicalDevice _physicalDevice = nullptr;
        vk::SurfaceKHR _surface = nullptr;
        uint32_t _graphicsFamily = 0;
        uint32_t _presentFamily = 0;

        vk::UniqueSwapchainKHR _swapChain;
        std::vector<vk::Image> _swapChainImages;
        vk::Format _swapChainImageFormat;
        vk::Extent2D _swapChainExtent;

        std::vector<vk::UniqueImageView> _swapChainImageViews;
    };

    std::vector<char> readFile(const std::string& filename) {
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

    vk::UniqueShaderModule loadShaderFromFile(vk::Device device, const std::string& path)
    {
        auto code = readFile(path);
        vk::ShaderModuleCreateInfo createInfo = {};
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        return device.createShaderModuleUnique(createInfo);
    }

}