cmake_minimum_required(VERSION 3.26)
project(HermesArc)

#-------------------------------------------------------------------------------
# Find libtorch
# (Assumes libtorch is installed in C:\Users\xshuj\lib\libtorch)
list(APPEND CMAKE_PREFIX_PATH "C:/Users/xshuj/lib/libtorch/lib")
find_package(Torch REQUIRED)

#-------------------------------------------------------------------------------
# Find CUDA runtime (for cudart)
find_package(CUDA REQUIRED)

#-------------------------------------------------------------------------------
# Find Unity PluginAPI headers (for IUnityGraphicsD3D12.h, IUnityInterface.h)
set(UNITY_EDITOR_DIR "C:/Program Files/Unity/Hub/Editor/6000.0.37f1/Editor/Data")
find_path(UNITY_PLUGINAPI_INCLUDE
        NAMES IUnityGraphicsD3D12.h IUnityInterface.h IUnityGraphics.h
        HINTS
        "${UNITY_EDITOR_DIR}/PluginAPI"
)
message(STATUS "Unity => ${UNITY_PLUGINAPI_INCLUDE}")
if(NOT UNITY_PLUGINAPI_INCLUDE)
    message(FATAL_ERROR "Could not locate Unity PluginAPI headers; check UNITY_EDITOR_DIR")
endif()

#-------------------------------------------------------------------------------
# Find D3D12 and DXGI libraries (Windows SDK)
find_library(D3D12_LIB d3d12 HINTS "$ENV{WINDIR}/System32")
find_library(DXGI_LIB  dxgi  HINTS "$ENV{WINDIR}/System32")

#-------------------------------------------------------------------------------
# Compiler settings
set(CMAKE_CXX_STANDARD 17)

#-------------------------------------------------------------------------------
# Build the plugin
add_library(HermesArc  SHARED
        src/library.cpp
)

# Export symbols with HERMESARC_EXPORTS
target_compile_definitions(HermesArc
        PRIVATE HERMESARC_EXPORTS
)
target_compile_definitions(HermesArc
        PRIVATE
        D3D12_NO_HELPERS
)

# Include directories
target_include_directories(HermesArc
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${UNITY_PLUGINAPI_INCLUDE}
        ${CUDA_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(HermesArc
        PUBLIC
        ${TORCH_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
        ${D3D12_LIB}
        ${DXGI_LIB}
)

# Ensure proper DLL naming for Unity
set_target_properties(HermesArc PROPERTIES
        PREFIX ""    # no 'lib' prefix on Windows
        SUFFIX ".dll"
)
