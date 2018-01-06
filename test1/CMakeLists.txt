cmake_minimum_required (VERSION 3.0)
project (ExampleVulkan)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ExampleVulkan main.cpp mvk/LogicalDeviceManager.cpp ../contrib/vulkan_ext_loader/vulkan_ext.c)

if (MSVC)
	add_definitions( "/MP /W3 /D_CRT_SECURE_NO_WARNINGS /wd4005 /wd4996 /wd4503 /nologo" )
    target_compile_options(ExampleVulkan PUBLIC "/std:c++17")
	#set_target_properties(ExampleVulkan PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif(MSVC)

find_package (Vulkan)
if (Vulkan_FOUND)
    message(STATUS "Found Vulkan, Including and Linking now")
    include_directories(${Vulkan_INCLUDE_DIRS})
    target_link_libraries (ExampleVulkan ${Vulkan_LIBRARIES})
    message(STATUS "Vulkan Include dirs: ${Vulkan_INCLUDE_DIRS} and libs: ${Vulkan_LIBRARIES}")
endif (Vulkan_FOUND)

#find_package(glfw3 3.2)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(../glfw "${CMAKE_CURRENT_BINARY_DIR}/glfw")
include_directories(../glfw)
include_directories(../contrib/glm)
include_directories(../contrib/Vulkan-Hpp/vulkan)
include_directories(../contrib/vulkan_ext_loader)


target_link_libraries (ExampleVulkan glfw)


#find_package(glm)
include_directories(../glm)