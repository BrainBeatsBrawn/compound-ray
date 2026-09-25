//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include "libEyeRenderer.h"

#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>

#include <sampleConfig.h>

#include <cuda/Light.h>

#include <sutil/Camera.h>
#include <sutil/CUDAOutputBuffer.h>
#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/sutil.h>
#include <sutil/vec_math.h>

#include "MulticamScene.h"
#include "GlobalParameters.h"
#include "cameras/CompoundEyeDataTypes.h"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdint>
#include <vector>
#include <stdexcept>

//#define USE_IAS // WAR for broken direct intersection of GAS on non-RTX cards

#ifndef BUFFER_TYPE_CUDA_DEVICE
# ifndef BUFFER_TYPE_GL_INTEROP
#  ifndef BUFFER_TYPE_ZERO_COPY
#   ifndef BUFFER_TYPE_CUDA_P2P
#    define BUFFER_TYPE_ZERO_COPY 1
#   endif
#  endif
# endif
#endif

#ifdef BUFFER_TYPE_CUDA_DEVICE
#define BUFFER_TYPE 0
#endif
#ifdef BUFFER_TYPE_GL_INTEROP
#define BUFFER_TYPE 1
#endif
#ifdef BUFFER_TYPE_ZERO_COPY
#define BUFFER_TYPE 2
#endif
#ifdef BUFFER_TYPE_CUDA_P2P
#define BUFFER_TYPE 3
#endif

// By default, call the summing kernel within launchFrame so that it will always
// happen (sum_average_with_getCameraData = false).
//
// It's potentially useful to call the summing kernel from inside getCameraData
// (sum_average_with_getCameraData = true) to profile its execution (seems to take 5 ms
// for summing/data transfer at 2048 samples, dropping to 2 ms for 1024 samples on my
// i9/4080).
static constexpr bool sum_average_with_getCameraData = false;

MulticamScene* scene;

// An output buffer used by non-compound eye cameras. Annoyingly, CUDAOutputBuffer has lots of GL calls in it. libEyeRenderer only.
sutil::CUDAOutputBuffer<uchar4>* outputBuffer = nullptr;
// The width and height of the output buffer. libEyeRenderer only.
int32_t width = 0;
int32_t height = 0;

bool notificationsActive = true;

void multicamAlloc()
{
    outputBuffer = new sutil::CUDAOutputBuffer<uchar4>(static_cast<sutil::CUDAOutputBufferType>(BUFFER_TYPE), width, height);
    scene = new MulticamScene{};
}

void multicamDealloc()
{
    if (outputBuffer) { delete outputBuffer; }
    delete scene;
}

// Updates the params to acurately reflect the currently selected camera
void handleCameraUpdate()
{
    // Make sure the SBT of the scene is updated for the newly selected camera before launch,
    // also push any changed host-side camera SBT data over to the device.
    scene->reconfigureSBTforCurrentCamera(false);
}

// Launch Optix threads to render a camera view. Once this is done getCameraData() accesses the
// summed average values for a compound eye. Non-compound eye data is accessed with
// getFramePointer()
void launchFrame (MulticamScene* _scene )
{
    if (outputBuffer && (outputBuffer->width() * outputBuffer->height() > 0)) {
        _scene->params->frame_buffer = outputBuffer->map();
    } else {
        _scene->params->frame_buffer = nullptr;
    }

    // d_params is a (no-longer global) pointer to GPU RAM, params is a (no-longer global) pointer to CPU-side RAM
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(_scene->d_params),
                               _scene->params,
                               sizeof(globalParameters::LaunchParams),
                               cudaMemcpyHostToDevice,
                               0)); // stream

    if (_scene->hasCompoundEyes() && _scene->isCompoundEyeActive()) {
        CompoundEye* camera = (CompoundEye*) _scene->getCamera();

        auto csbt = _scene->compoundSbt();
        // Launch the ommatidial renderer
        auto cpl = _scene->compoundPipeline();
        auto ole = optixLaunch (cpl,                               // pipeline
                                0,                                 // stream
                                reinterpret_cast<CUdeviceptr>( _scene->d_params ), // pipelineParams
                                sizeof( globalParameters::LaunchParams ),  // pipelineParamsSize
                                csbt,                              // shader buffer table
                                camera->getOmmatidialCount(),      // launch width
                                camera->getSamplesPerOmmatidium(), // launch height
                                1);                                // launch depth
        OPTIX_CHECK (ole);

        {
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                std::stringstream ss;
                ss << "Post-launch CUDA error on synchronize with error " << (int)error << " '"
                   << cudaGetErrorString (error)
                   << "' (" __FILE__ << ":" << __LINE__ << ")\n";
                throw sutil::Exception (ss.str().c_str());
            }
        } // this is more or less CUDA_SYNC_CHECK();

        _scene->params->frame++;// Increase the frame number
        camera->setRandomsAsConfigured();// Make sure that random stream initialization is only ever done once

        if constexpr (sum_average_with_getCameraData == false) {
            // After the compoundray pipeline, can call the sample-summing CUDA kernel here
            camera->averageRecordFrame();
            CUDA_SYNC_CHECK();
        }
    }

    // Launch non-compound render (if required)
    if (_scene->require_noncompound_pipeline == true && width > 0 && height > 0) {
        OPTIX_CHECK (optixLaunch (_scene->pipeline(),
                                  0,      // stream
                                  reinterpret_cast<CUdeviceptr>(_scene->d_params),
                                  sizeof(globalParameters::LaunchParams),
                                  _scene->sbt(),
                                  width,  // launch width
                                  height, // launch height
                                  1));    // launch depth is 1
    }

    if (outputBuffer&& (outputBuffer->width() * outputBuffer->height() > 0)) {
        outputBuffer->unmap();
    }

    CUDA_SYNC_CHECK();
}

void cleanup() {} // no-op; cleanup is now all in MulticamScene deconstructor

//------------------------------------------------------------------------------
//
// API functions
//
//------------------------------------------------------------------------------
// General Running
//------------------------------------------------------------------------------
void setVerbosity (bool v) { notificationsActive = v; }

void loadGlTFscene (const char* filepath, sutil::Matrix4x4 root_transform)
{
    if (scene == nullptr) { throw sutil::Exception ("loadGlTFscene exception: scene is nullptr"); }
    scene->loadGlTFscene (filepath, root_transform);
}

void setRenderSize (int w, int h)
{
    width = w;
    height = h;
    // Resize our non-compound eye output buffer here
    outputBuffer->resize (width, height);
}

double renderFrame()
{
    handleCameraUpdate();

    auto then = std::chrono::steady_clock::now();
    launchFrame (scene);
    CUDA_SYNC_CHECK();
    std::chrono::duration<double, std::milli> render_time = std::chrono::steady_clock::now() - then;

    if (notificationsActive) {
        std::cout<<"[PyEye] Rendered frame in "<<render_time.count()<<"ms."<<std::endl;
    }

    return(render_time.count());
}

void saveFrameAs (const char* ppmFilename)
{
    sutil::ImageBuffer buffer;
    buffer.data = outputBuffer->getHostPointer();
    buffer.width = outputBuffer->width();
    buffer.height = outputBuffer->height();
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;

    sutil::displayBufferFile (ppmFilename, buffer, false);

    if (notificationsActive) {
        std::cout << "[PyEye] Saved render as '" << ppmFilename << "'\n";
    }
}

unsigned char* getFramePointer()
{
    if(notificationsActive) { std::cout << "[PyEye] Retrieving frame pointer...\n"; }
    return (unsigned char*)outputBuffer->getHostPointer();
}

void setRequireNoncompoundPipeline (const bool require_ncp)
{
    scene->require_noncompound_pipeline = require_ncp;
}

// Currently not revealed in libEyeRenderer.h...
void getFrame (unsigned char* frame)
{
    if(notificationsActive) { std::cout << "[PyEye] Retrieving frame...\n"; }
    size_t displaySize = outputBuffer->width() * outputBuffer->height();
    for (size_t i = 0; i < displaySize; i++) {
        unsigned char val = (unsigned char)(((float)i/(float)displaySize)*254);
        frame[displaySize*3 + 0] = val;
        frame[displaySize*3 + 1] = val;
        frame[displaySize*3 + 2] = val;
    }
}

void stop() {} // now a no-op. All cleanup is in MulticamScene deconstruction

//------------------------------------------------------------------------------
// Camera Control
//------------------------------------------------------------------------------
size_t getCameraCount() { return(scene->getCameraCount()); }

void nextCamera() { scene->nextCamera(); }

size_t getCurrentCameraIndex() { return(scene->getCameraIndex()); }

const char* getCurrentCameraName() { return(scene->getCamera()->getCameraName()); }

void previousCamera() { scene->previousCamera(); }

void gotoCamera (int index) { scene->setCurrentCamera(index); }

bool gotoCameraByName (char* name)
{
    scene->setCurrentCamera(0);
    for (auto i = 0u; i<scene->getCameraCount(); i++) {
        if (strcmp(name, scene->getCamera()->getCameraName()) == 0) {
            return true;
        }
        scene->nextCamera();
    }
    return false;
}

void setCameraPosition (float x, float y, float z)
{
    scene->getCamera()->setPosition(make_float3(x,y,z));
}

void getCameraPosition (float& x, float& y, float& z)
{
    const float3& camPos = scene->getCamera()->getPosition();
    x = camPos.x;
    y = camPos.y;
    z = camPos.z;
}

void setCameraLocalSpace (float lxx, float lxy, float lxz,
                          float lyx, float lyy, float lyz,
                          float lzx, float lzy, float lzz)
{
    scene->getCamera()->setLocalSpace (make_float3(lxx, lxy, lxz),
                                       make_float3(lyx, lyy, lyz),
                                       make_float3(lzx, lzy, lzz));
}

void rotateCameraAround (float angle, float x, float y, float z)
{
    scene->getCamera()->rotateAround (angle,  make_float3(x,y,z));
}

void rotateCameraLocallyAround (float angle, float x, float y, float z)
{
    scene->getCamera()->rotateLocallyAround (angle, make_float3(x,y,z));
}

void rotateCamerasAround (float angle, float x, float y, float z)
{
    size_t cc = scene->getCameraCount();
    for (size_t i = 0; i < cc; ++i) {
        scene->getCamera()->rotateAround(angle,  make_float3(x,y,z));
        scene->nextCamera();
    }
}

void rotateCamerasLocallyAround (float angle, float x, float y, float z)
{
    scene->rotateCamerasLocallyAround (angle, x, y, z);
}

void translateCamera (float x, float y, float z)
{
    scene->getCamera()->move(make_float3(x, y, z));
}

void translateCameraLocally (float x, float y, float z)
{
    scene->getCamera()->moveLocally(make_float3(x, y, z));
}

void translateCamerasLocally (float x, float y, float z)
{
    // For each camera in scene:
    size_t cc = scene->getCameraCount();
    for (size_t i = 0; i < cc; ++i) {
        scene->getCamera()->moveLocally (make_float3(x, y, z));
        scene->nextCamera();
    } // at end should have looped back to original cam
}

void resetCameraPose() { scene->getCamera()->resetPose(); }

void setCameraPose (float posX, float posY, float posZ, float rotX, float rotY, float rotZ)
{
    GenericCamera* c = scene->getCamera();
    c->resetPose();
    c->rotateAround(rotX, make_float3(1,0,0));
    c->rotateAround(rotY, make_float3(0,1,0));
    c->rotateAround(rotZ, make_float3(0,0,1));
    c->move(make_float3(posX, posY, posZ));
}

void setCameraPoseMatrix (const sutil::Matrix4x4& camera_localspace)
{
    scene->setCameraPoseMatrix (camera_localspace);
}

void getCameraData (std::vector<std::array<float, 3>>& cameraData) { scene->getCameraData (cameraData); }

//------------------------------------------------------------------------------
// Ommatidial Camera Control
//------------------------------------------------------------------------------
bool isCompoundEyeActive() { return scene->isCompoundEyeActive(); }
std::string getEyeDataPath() { return scene->getEyeDataPath(); }
void setCurrentEyeSamplesPerOmmatidium (int s) { scene->setCurrentEyeSamplesPerOmmatidium (s); }
int getCurrentEyeSamplesPerOmmatidium() { return scene->getCurrentEyeSamplesPerOmmatidium(); }
void changeCurrentEyeSamplesPerOmmatidiumBy (int s) { scene->changeCurrentEyeSamplesPerOmmatidiumBy (s); }
size_t getCurrentEyeOmmatidialCount() { return scene->getCurrentEyeOmmatidialCount(); }

void setOmmatidia (OmmatidiumPacket* omms, size_t count)
{
    // Break out if the current eye isn't compound
    if (!scene->isCompoundEyeActive()) { return; }

    // First convert the OmmatidiumPacket list into an array of Ommatidium objects
    std::vector<Ommatidium> ommVector(count);
    for(size_t i = 0; i<count; i++) {
        OmmatidiumPacket& omm = omms[i];
        ommVector[i].relativePosition  = make_float3(omm.posX, omm.posY, omm.posZ);
        ommVector[i].relativeDirection = make_float3(omm.dirX, omm.dirY, omm.dirZ);
        ommVector[i].acceptanceAngleRadians = omm.acceptanceAngle;
        ommVector[i].focalPointOffset = omm.focalpointOffset;
    }

    // Actually set the new ommatidial structure
    ((CompoundEye*)scene->getCamera())->setOmmatidia (ommVector.data(), count);
}

const char* getCurrentEyeDataPath()
{
    if (scene->isCompoundEyeActive()) {
        return ((CompoundEye*)scene->getCamera())->eyeDataPath.c_str();
    }
    return "\0";
}

void setCurrentEyeShaderName (char* name)
{
    if (scene->isCompoundEyeActive()) {
        ((CompoundEye*)scene->getCamera())->setShaderName (std::string(name)); // Set the shader
        scene->reconfigureSBTforCurrentCamera (true); // Reconfigure for the new shader
    }
}

bool isInsideHitGeometry(float x, float y, float z, char* name)
{
    return scene->isInsideHitGeometry (make_float3(x, y, z), std::string(name), false);
}

float3 getGeometryMaxBounds(char* name)
{
    return scene->getGeometryMaxBounds (std::string(name));
}

float3 getGeometryMinBounds(char* name)
{
    return scene->getGeometryMinBounds (std::string(name));
}
