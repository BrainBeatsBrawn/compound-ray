#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Include the eye renderer through the CMakeIncludes file instead
# of explicitly listing every file again.
set(local_eye_renderer_sources "")
include("CMakeIncludes.cmake")
add_eye_renderer("." local_eye_renderer_sources)

# Mark a specific source file to be CUDA! From https://forums.developer.nvidia.com/t/how-to-compile-cuda-c-files-and-nvidia-optix-files-inside-the-same-visual-studio-project/31821/2
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/summing_kernel.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)
OPTIX_add_sample_executable(EyeRenderer3 EyeRenderer3_target ${local_eye_renderer_sources} OPTIONS -rdc true)

target_include_directories(${EyeRenderer3_target} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/compound-ray>"
)

# Client to link CUDA
target_link_libraries(${EyeRenderer3_target} -lcudart -lrt -ldl)

# Install EyeRender3 and its dependencies giving it the export name compound-ray-config
install(TARGETS ${EyeRenderer3_target} ${sutil_target} ${glad_target}
        EXPORT compound-ray-config
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/compound-ray
      )

# Install headers
install(FILES libEyeRenderer.h GlobalParameters.h MulticamScene.h shaders.cu
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/compound-ray)
install(FILES
        cameras/DataRecordCamera.h
        cameras/CompoundEye.h
        cameras/CompoundEyeDataTypes.h
        cameras/GenericCamera.h
        cameras/GenericCameraDataTypes.h
        cameras/OrthographicCamera.h
        cameras/OrthographicCameraDataTypes.h
        cameras/PanoramicCamera.h
        cameras/PanoramicCameraDataTypes.h
        cameras/PerspectiveCamera.h
        cameras/PerspectiveCameraDataTypes.h
        cameras/ThreeSixtyCamera.h
        cameras/ThreeSixtyCameraDataTypes.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/compound-ray/cameras)

# Create the EXPORT so that compound-ray has its own .cmake file and find_package(compound-ray) should work
export(TARGETS ${EyeRenderer3_target} ${sutil_target} ${glad_target}
    NAMESPACE compound-ray::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/compound-ray-config.cmake"
)

install(EXPORT compound-ray-config
        FILE compound-ray-config.cmake
        NAMESPACE compound-ray::
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/compound-ray"
)
