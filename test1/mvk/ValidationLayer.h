#pragma once
#include <vulkan.hpp>
#include <vulkan_ext.h>
#include <iostream>

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


}