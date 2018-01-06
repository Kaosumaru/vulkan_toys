

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#include <vulkan.hpp>
#include <vulkan_ext.h>

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <set>

#include <type_traits>
#include <fstream>
#include <filesystem>

#include "function_traits.hpp"


const int WIDTH = 800;
const int HEIGHT = 600;

/*
Implement
- vkQueueWaitIdle "One thing i want to point out is that using queue wait idle on each frame is not the best thing to do. In fact the performance could be worst than GL."
- split code into logical subobjects

Read about
- vkSubpassDependency "It assumes that the transition occurs at the start of the pipeline, but we haven't acquired the image yet at that point! There are two ways to deal with this problem. "


X (seems to be pointless in game case) createInfo.oldSwapchain "You need to pass the previous swap chain to the oldSwapChain field in the VkSwapchainCreateInfoKHR struct and destroy the old swap chain as soon as you've finished using it."
*/



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





class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:


    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    GLFWwindow* window;

    vk::UniqueInstance instance;
    mvk::ValidationLayers validationLayersManager;

    vk::PhysicalDevice physicalDevice;
    vk::UniqueSurfaceKHR surface;
    vk::UniqueDevice device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::UniqueRenderPass renderPass;
    vk::UniquePipelineLayout pipelineLayout;
    vk::UniquePipeline graphicsPipeline;

    mvk::SwapChainManager _swapChainManager;
    std::vector<vk::UniqueFramebuffer> swapChainFramebuffers;

    vk::UniqueCommandPool commandPool;
    std::vector<vk::UniqueCommandBuffer> commandBuffers;

    struct SyncSemaphores
    {
        vk::UniqueSemaphore imageAvailableSemaphore;
        vk::UniqueSemaphore renderFinishedSemaphore;
    } sync;

    struct QueueFamilyIndices {
        int graphicsFamily = -1;
        int presentFamily = -1;

        bool isComplete() {
            return graphicsFamily >= 0 && presentFamily >= 0;
        }
    };

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        validationLayersManager.setupDebugCallback(*instance);
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSemaphores();
        createCommandPool();
        recreateFramebuffers();
    }

    void recreateFramebuffers() {
        device->waitIdle();

        cleanupFramebuffers();

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        if (!_swapChainManager.create(*device, physicalDevice, *surface, indices.graphicsFamily, indices.presentFamily)) return;

        //pipeline must be recreated since potentially viewport size & output format changed
        createRenderPass();
        createGraphicsPipeline();

        //we need to create new framebuffers with new renderpass data & image views
        createFramebuffers();
        //and new command buffers for these framebuffers
        createCommandBuffers();
    }

    void cleanupFramebuffers() {
        if (!_swapChainManager.valid()) return;
        swapChainFramebuffers.clear();

        commandBuffers.clear();

        graphicsPipeline.reset();
        pipelineLayout.reset();
        renderPass.reset();


        _swapChainManager.cleanup();
    }

    void cleanup() {
        cleanupFramebuffers();
        glfwDestroyWindow(window);
        glfwTerminate();
    }


    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        device->waitIdle();
    }

    void drawFrame() {
        vkQueueWaitIdle(presentQueue); //TODO possibly suboptimal
        
        if (!_swapChainManager.valid())
        {
            //TODO sleep a bit
            recreateFramebuffers();
            return;
        }

        auto& swapChain = _swapChainManager.swapChain();
        auto [result, imageIndex] = device->acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(), sync.imageAvailableSemaphore.get(), nullptr);
        if (shouldRecreateSwapChain(result)) return;

        vk::SubmitInfo submitInfo = {};

        //wait with command submission till requested image is ready
        //(we are waiting before VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT since previos pipeline steps possibly don't require image)
        vk::Semaphore waitSemaphores[] = {sync.imageAvailableSemaphore.get()};


        vk::PipelineStageFlags waitStages[] = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        //submit this command buffer
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[imageIndex].get(); //TODO check

        //notify renderFinishedSemaphore when render is done
        vk::Semaphore signalSemaphores[] = {sync.renderFinishedSemaphore.get()};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if (graphicsQueue.submit(1, &submitInfo, nullptr) != vk::Result::eSuccess) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }


        //present frame (return to swap chain)
        vk::PresentInfoKHR presentInfo = {};

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        vk::SwapchainKHR swapChains[] = {*swapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        presentInfo.pResults = nullptr; // Optional
        result = presentQueue.presentKHR(&presentInfo);

        if (shouldRecreateSwapChain(result)) return;

    }

    bool shouldRecreateSwapChain(vk::Result result)
    {
        if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
            recreateFramebuffers();
            return true;
        }
        else if (result != vk::Result::eSuccess) {
            throw std::runtime_error("failed to present swap chain image!");
        }
        return false;
    }



    void createInstance() {
        if (!validationLayersManager.isOK()) {
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

        validationLayersManager.AddToCreateInfo(createInfo);

        instance = vk::createInstanceUnique(createInfo);
        vkExtInitInstance(*instance);
    }

    void createSurface() {
        VkSurfaceKHR surf;
        if (glfwCreateWindowSurface(*instance, window, nullptr, &surf) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        vk::SurfaceKHRDeleter deleter(*instance);
        surface = vk::UniqueSurfaceKHR(surf, deleter);
    }

    void pickPhysicalDevice() {
        auto devices = instance->enumeratePhysicalDevices();

        if (devices.size() == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        auto checkDeviceExtensionSupport = [&](vk::PhysicalDevice device) {
            auto availableExtensions = device.enumerateDeviceExtensionProperties();

            std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

            for (const auto& extension : availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        };

        auto isDeviceSuitable = [&](vk::PhysicalDevice device) {
            bool extensionsSupported = checkDeviceExtensionSupport(device);

            if (!extensionsSupported) return false;

            mvk::SwapChainSupportDetails swapChainSupport {device, *surface};
            if (!swapChainSupport.valid()) return false;

            return findQueueFamilies(device).isComplete();
        };

        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                physicalDevice = device;
                break;
            }
        }

        if (!physicalDevice) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice physicalDevice) {
        QueueFamilyIndices indices;

        auto queueFamilies = physicalDevice.getQueueFamilyProperties();

        int i = 0;
        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueCount > 0 && queueFamily.queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }

            auto presentSupport = physicalDevice.getSurfaceSupportKHR(i, *surface);

            if (queueFamily.queueCount > 0 && presentSupport) {
                indices.presentFamily = i;
            }

            if (indices.isComplete()) {
                break;
            }

            i++;
        }
        return indices;
    }

    void createLogicalDevice() {
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<int> uniqueQueueFamilies = {indices.graphicsFamily, indices.presentFamily};

        float queuePriority = 1.0f;
        for (int queueFamily : uniqueQueueFamilies) {
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

        validationLayersManager.AddToCreateInfo(createInfo);

        device = physicalDevice.createDeviceUnique(createInfo);

        //vkExtInitDevice(device); - TODO this probably invalidates some functions...
        graphicsQueue = device->getQueue(indices.graphicsFamily, 0);
        presentQueue = device->getQueue(indices.presentFamily, 0);
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment = {};
        colorAttachment.format = _swapChainManager.imageFormat();
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;


        vk::AttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::SubpassDescription subpass = {};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        vk::SubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.srcAccessMask = {};
        dependency.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;

        vk::RenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;

        renderPass = device->createRenderPassUnique(renderPassInfo);
    }

    void createGraphicsPipeline() 
    {
        auto swapChainExtent = _swapChainManager.extent();

        auto vertShaderModule = mvk::loadShaderFromFile(*device, "shaders/vert.spv");
        auto fragShaderModule = mvk::loadShaderFromFile(*device, "shaders/frag.spv");

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo = {};
        vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertShaderStageInfo.module = vertShaderModule.get();
        vertShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo = {};
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragShaderStageInfo.module = fragShaderModule.get();
        fragShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

        //dummy vertex input
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo = {};
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.pVertexBindingDescriptions = nullptr; // Optional
        vertexInputInfo.vertexAttributeDescriptionCount = 0;
        vertexInputInfo.pVertexAttributeDescriptions = nullptr; // Optional

                                                                //topology TODO READ MORE
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly = {};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        //viewport/scissor
        vk::Viewport viewport = {};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        vk::Rect2D scissor = {};
        scissor.offset = {0, 0};
        scissor.extent = swapChainExtent;

        vk::PipelineViewportStateCreateInfo viewportState = {};
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        //rasterizer
        vk::PipelineRasterizationStateCreateInfo rasterizer = {};
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = vk::PolygonMode::eFill;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.frontFace = vk::FrontFace::eClockwise;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

                                                //multisampling
        vk::PipelineMultisampleStateCreateInfo multisampling = {};
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional

                                                   //blending
        vk::PipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = vk::BlendFactor::eOne; // Optional
        colorBlendAttachment.dstColorBlendFactor = vk::BlendFactor::eZero; // Optional
        colorBlendAttachment.colorBlendOp = vk::BlendOp::eAdd; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = vk::BlendFactor::eOne; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = vk::BlendFactor::eZero; // Optional
        colorBlendAttachment.alphaBlendOp = vk::BlendOp::eAdd; // Optional

        vk::PipelineColorBlendStateCreateInfo colorBlending = {};
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = vk::LogicOp::eCopy; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        //dynamic states (currently disabled)
#if 0
        vk::DynamicState dynamicStates[] = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eLineWidth
        };

        vk::PipelineDynamicStateCreateInfo dynamicState = {};
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates = dynamicStates;
#endif


        //create pipeline layout
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo = {};
        pipelineLayoutInfo.setLayoutCount = 0; // Optional
        pipelineLayoutInfo.pSetLayouts = nullptr; // Optional
        pipelineLayoutInfo.pushConstantRangeCount = 0; // Optional
        pipelineLayoutInfo.pPushConstantRanges = 0; // Optional

        pipelineLayout = device->createPipelineLayoutUnique(pipelineLayoutInfo);


        //create graphic pipeline
        vk::GraphicsPipelineCreateInfo pipelineInfo = {};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = nullptr; // Optional
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // Optional
        pipelineInfo.layout = *pipelineLayout;
        pipelineInfo.renderPass = *renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = nullptr; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional
        graphicsPipeline = device->createGraphicsPipelineUnique(nullptr, pipelineInfo);
    }

    void createFramebuffers() {
        auto swapChainExtent = _swapChainManager.extent();
        auto& swapChainImageViews = _swapChainManager.imageViews();
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vk::ImageView attachments[] = {
                *swapChainImageViews[i]
            };

            vk::FramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.renderPass = *renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
            swapChainFramebuffers[i] = device->createFramebufferUnique(framebufferInfo);
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        vk::CommandPoolCreateInfo poolInfo = {};
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily;

        commandPool = device->createCommandPoolUnique(poolInfo);
    }

    void createCommandBuffers() {
        auto swapChainExtent = _swapChainManager.extent();

        commandBuffers.resize(swapChainFramebuffers.size());

        vk::CommandBufferAllocateInfo allocInfo = {};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        commandBuffers = device->allocateCommandBuffersUnique(allocInfo);

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            vk::CommandBufferBeginInfo beginInfo = {};
            beginInfo.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
            beginInfo.pInheritanceInfo = nullptr; // Optional

            commandBuffers[i]->begin(beginInfo);

            vk::RenderPassBeginInfo renderPassInfo = {};
            renderPassInfo.renderPass = *renderPass;
            renderPassInfo.framebuffer = *swapChainFramebuffers[i];
            renderPassInfo.renderArea.offset = {0, 0};
            renderPassInfo.renderArea.extent = swapChainExtent;

            vk::ClearValue clearColor = vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
            renderPassInfo.clearValueCount = 1;
            renderPassInfo.pClearValues = &clearColor;

            commandBuffers[i]->beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[i]->bindPipeline(vk::PipelineBindPoint::eGraphics, *graphicsPipeline);
            commandBuffers[i]->draw(3, 1, 0, 0);
            commandBuffers[i]->endRenderPass();

            commandBuffers[i]->end();
        }
    }

    void createSemaphores() {
        sync.imageAvailableSemaphore = device->createSemaphoreUnique({});
        sync.renderFinishedSemaphore = device->createSemaphoreUnique({});
    }

    std::vector<const char*> getRequiredExtensions() {
        std::vector<const char*> extensions;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        for (uint32_t i = 0; i < glfwExtensionCount; i++) {
            extensions.push_back(glfwExtensions[i]);
        }

        validationLayersManager.AddRequiredExtension(extensions);



        return extensions;
    }

};


int main() {
    std::string path = std::experimental::filesystem::current_path().string();
    HelloTriangleApplication app;

    try {
        app.run();
    }
    catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

