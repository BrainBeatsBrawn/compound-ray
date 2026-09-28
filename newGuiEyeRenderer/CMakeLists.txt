# Set the local resources from this folder (similar to the resources set in all other CMake files)
set(gui_eye_renderer_sources
  libEyeRenderer.cpp # This example uses a now-deprecated C API to libcompoundray
  gui.cpp
  BasicController.h
  BasicController.cpp
  GLDisplay.cpp
)

# Mark a specific source file to be CUDA! From https://forums.developer.nvidia.com/t/how-to-compile-cuda-c-files-and-nvidia-optix-files-inside-the-same-visual-studio-project/31821/2
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../libEyeRenderer3/summing_kernel.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)
OPTIX_add_sample_executable (newGuiEyeRenderer target_name ${gui_eye_renderer_sources} OPTIONS -rdc true)

# #include <glad/glad.h> to come from ../support/
target_include_directories(${target_name} INTERFACE PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../support")
# libEyeRenderer.cpp needs access to MulticamScene.h etc
target_include_directories(${target_name} INTERFACE PRIVATE "../src")

# Note that we link dynamically to -lcudart, rt, dl, as this appeared
# to be necessary to get a CUDA reduction kernel to run after an Optix
# ray tracing frame run (${CUDA_LIBRARIES} contains static links).
target_link_libraries(${target_name} -lcudart -lrt -ldl ${glad_target} ${GLFW_LIB_NAME} compoundray)

set_target_properties (${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
