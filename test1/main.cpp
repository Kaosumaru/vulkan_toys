

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#include <vulkan.hpp>
#include <vulkan_ext.h>
#include "mvk/LogicalDevice.h"
#include "mvk/Utils.h"
#include "mvk/SwapChain.h"

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <set>

#include <type_traits>

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









class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    GLFWwindow* window;

    std::unique_ptr<mvk::LogicalDevice> _vulkan;
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



    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        _vulkan = std::make_unique<mvk::LogicalDevice>(window);
        createQueues();
        createSemaphores();
        createCommandPool();
        recreateFramebuffers();
    }

    void recreateFramebuffers() {
        _vulkan->device().waitIdle();

        cleanupFramebuffers();

        if (!_swapChainManager.create(WIDTH, HEIGHT, *_vulkan)) return;

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

        _vulkan->device().waitIdle();
    }

    void drawFrame() {
        auto device = _vulkan->device();
        vkQueueWaitIdle(presentQueue); //TODO possibly suboptimal
        
        if (!_swapChainManager.valid())
        {
            //TODO sleep a bit
            recreateFramebuffers();
            return;
        }

        auto& swapChain = _swapChainManager.swapChain();
        auto [result, imageIndex] = device.acquireNextImageKHR(*swapChain, std::numeric_limits<uint64_t>::max(), sync.imageAvailableSemaphore.get(), nullptr);
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

    void createQueues()
    {
        auto device = _vulkan->device();
        auto& indices = _vulkan->indices();
        graphicsQueue = device.getQueue(indices.graphicsFamily, 0);
        presentQueue = device.getQueue(indices.presentFamily, 0);
    }

    void createRenderPass() 
    {
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

        renderPass = _vulkan->device().createRenderPassUnique(renderPassInfo);
    }

    void createGraphicsPipeline() 
    {
        auto device = _vulkan->device();
        auto swapChainExtent = _swapChainManager.extent();

        auto vertShaderModule = mvk::loadShaderFromFile(device, "shaders/vert.spv");
        auto fragShaderModule = mvk::loadShaderFromFile(device, "shaders/frag.spv");

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

        pipelineLayout = device.createPipelineLayoutUnique(pipelineLayoutInfo);


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
        graphicsPipeline = device.createGraphicsPipelineUnique(nullptr, pipelineInfo);
    }

    void createFramebuffers() 
    {
        auto device = _vulkan->device();
        auto swapChainExtent = _swapChainManager.extent();
        auto& swapChainImageViews = _swapChainManager.imageViews();
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) 
        {
            vk::ImageView attachments[] = { *swapChainImageViews[i] };

            vk::FramebufferCreateInfo framebufferInfo = {};
            framebufferInfo.renderPass = *renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = attachments;
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;
            swapChainFramebuffers[i] = device.createFramebufferUnique(framebufferInfo);
        }
    }

    void createCommandPool()
    {
        vk::CommandPoolCreateInfo poolInfo = {};
        poolInfo.queueFamilyIndex = _vulkan->indices().graphicsFamily;
        commandPool = _vulkan->device().createCommandPoolUnique(poolInfo);
    }

    void createCommandBuffers() 
    {
        auto device = _vulkan->device();
        auto swapChainExtent = _swapChainManager.extent();

        commandBuffers.resize(swapChainFramebuffers.size());

        vk::CommandBufferAllocateInfo allocInfo = {};
        allocInfo.commandPool = *commandPool;
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        commandBuffers = device.allocateCommandBuffersUnique(allocInfo);

        for (size_t i = 0; i < commandBuffers.size(); i++) 
        {
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

    void createSemaphores() 
    {
        auto device = _vulkan->device();
        sync.imageAvailableSemaphore = device.createSemaphoreUnique({});
        sync.renderFinishedSemaphore = device.createSemaphoreUnique({});
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

