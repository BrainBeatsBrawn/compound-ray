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

include_directories(".")

set(libcompoundray_sources
  ./RayComputeTypes.h
  ./MulticamScene.h
  ./MulticamScene.cpp
  ./cameras/GenericCamera.h
  ./cameras/GenericCamera.cpp
  ./cameras/PerspectiveCamera.h
  ./cameras/PerspectiveCamera.cpp
  ./cameras/PanoramicCamera.h
  ./cameras/PanoramicCamera.cpp
  ./cameras/OrthographicCamera.h
  ./cameras/OrthographicCamera.cpp
  ./cameras/CompoundEye.h
  ./cameras/CompoundEye.cpp
  ./cameras/DataRecordCamera.h
  ./shaders.cu
  ./summing_kernel.cu
)

# Mark a specific source file to be CUDA! From https://forums.developer.nvidia.com/t/how-to-compile-cuda-c-files-and-nvidia-optix-files-inside-the-same-visual-studio-project/31821/2
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/summing_kernel.cu
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
)

# libcompoundray is a new library that will only have MulticamScene in it (and *no* libEyeRenderer C API)
OPTIX_add_sample_executable(compoundray compoundray_target ${libcompoundray_sources} OPTIONS -rdc true)
message("generated ptx files for compoundray: ${compoundray_ptx_files}")
message ("compoundray_target is ${compoundray_target}")

target_include_directories(${compoundray_target} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/compoundray>"
)

# Client to link CUDA
target_link_libraries(${compoundray_target} -lcudart -lrt -ldl)

# Install the PTX files
install(FILES ${compoundray_ptx_files} DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})

# Install EyeRender3, compoundray and its dependencies giving it the export name compoundray-config
install(TARGETS ${compoundray_target} ${sutil_target}
        EXPORT compoundray-config
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/compoundray
      )

# Install headers
install(FILES RayComputeTypes.h MulticamScene.h shaders.cu
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/compoundray)
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
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/compoundray/cameras)

# Create the EXPORT so that compoundray has its own .cmake file and find_package(compoundray) should work
export(TARGETS ${compoundray_target} ${sutil_target}
    NAMESPACE compoundray::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/compoundray-config.cmake"
)

install(EXPORT compoundray-config
        FILE compoundray-config.cmake
        NAMESPACE compoundray::
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/compoundray"
)
