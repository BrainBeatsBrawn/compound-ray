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

set(sources
    ${CMAKE_CURRENT_BINARY_DIR}/../sampleConfig.h
    ${SAMPLES_CUDA_DIR}/whitted.cu
    ${SAMPLES_CUDA_DIR}/BufferView.h
    ${SAMPLES_CUDA_DIR}/GeometryData.h
    ${SAMPLES_CUDA_DIR}/Light.h
    ${SAMPLES_CUDA_DIR}/LocalGeometry.h
    ${SAMPLES_CUDA_DIR}/MaterialData.h
    ${SAMPLES_CUDA_DIR}/util.h
    Aabb.h
    Camera.cpp
    Camera.h
    CUDAOutputBuffer.h
    Exception.h
    Matrix.h
    PPMLoader.cpp
    PPMLoader.h
    Preprocessor.h
    Quaternion.h
    Record.h
    sutilapi.h
    sutil.cpp
    sutil.h
    Trackball.cpp
    Trackball.h
    vec_math.h
    hitscanprocessing.h
    hitscanprocessing.cpp
    )



include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CUDA_NVRTC_ENABLED)
  CUDA_COMPILE_PTX(ptx_files ${sources})
endif()

# Make the library.
add_library(${sutil_target} ${sources})

if( WIN32 )
  target_compile_definitions( ${sutil_target} PUBLIC GLAD_GLAPI_EXPORT )
endif()

target_link_libraries(${sutil_target} LINK_PRIVATE ${GLFW_LIB_NAME} ${CUDA_LIBRARIES})

target_include_directories( ${sutil_target} PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

# Use gcc rather than g++ to link if we are linking statically against libgcc_s and libstdc++
if(USING_GNU_C OR USING_GNU_CXX)
  if(GCC_LIBSTDCPP_HACK)
    set_target_properties(${sutil_target} PROPERTIES LINKER_LANGUAGE "C")
    target_link_libraries(${sutil_target} LINK_PRIVATE ${STATIC_LIBSTDCPP})
  endif()
endif()


if(CUDA_NVRTC_ENABLED)
  target_link_libraries(${sutil_target} LINK_PRIVATE ${CUDA_nvrtc_LIBRARY})
endif()
if(WIN32)
  target_link_libraries(${sutil_target} LINK_PRIVATE winmm.lib)
endif()

# Make the list of sources available to the parent directory for installation needs.
set(sutil_sources "${sources}" PARENT_SCOPE)

set_property(TARGET ${sutil_target} PROPERTY FOLDER "${OPTIX_IDE_FOLDER}")

# NB: install() of libsutil_7_sdk.so is carried out in
# libEyeRenderer/CMakeLists.txt as libEyeRender.so is exported along
# with glad and sutil_7_sdk

# Install headers
install(FILES Aabb.h Camera.h CUDAOutputBuffer.h Exception.h
        hitscanprocessing.h Matrix.h PPMLoader.h Preprocessor.h
        Quaternion.h Record.h sutilapi.h sutil.h Trackball.h vec_math.h
        WorkDistribution.h
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/compound-ray/sutil)
