# Set the local resources from this folder (similar to the resources set in all other CMake files)
set(gui_eye_renderer_sources
  gui.cpp
  BasicController.h
  BasicController.cpp
  GLDisplay.cpp
)

# Include the 'add_eye_renderer' macro from the eye renderer and
# then actually add the correct files to the source list so we can use
# eye renderer library commands.
include("../libEyeRenderer3/CMakeIncludes.cmake")
add_eye_renderer("../libEyeRenderer3" gui_eye_renderer_sources)

# Mark a specific source file to be CUDA! From https://forums.developer.nvidia.com/t/how-to-compile-cuda-c-files-and-nvidia-optix-files-inside-the-same-visual-studio-project/31821/2
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/../libEyeRenderer3/summing_kernel.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)
OPTIX_add_sample_executable (newGuiEyeRenderer target_name ${gui_eye_renderer_sources} OPTIONS -rdc true)

# #include <glad/glad.h> to come from ../support/
target_include_directories(${target_name} INTERFACE PRIVATE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/../support")

# ${CUDA_LIBRARIES} contains static links. Note that libEyeRenderer
# now links dynamically to -lcudart, rt, dl, as this appeared to be
# necessary to get a CUDA reduction kernel to run after an Optix ray
# tracing frame run.
target_link_libraries(${target_name} -lcudart -lrt -ldl ${glad_target})

set_target_properties (${target_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
