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

#ifndef MULTICAM_SCENE_H
#define MULTICAM_SCENE_H

#include <cuda/BufferView.h>
#include <cuda/MaterialData.h>
#include <sutil/Aabb.h>
#include <sutil/Matrix.h>
#include <sutil/Preprocessor.h>
#include <sutil/sutilapi.h>
#include <sutil/hitscanprocessing.h>

#include <cuda_runtime.h>

#include <optix.h>

#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <array>
#include <cmath>
#include <sstream>
#include <chrono>
#include <map>
#include <fstream>

#include "RayComputeTypes.h"
#include "cameras/GenericCameraDataTypes.h"
#include "cameras/GenericCamera.h"
#include "cameras/CompoundEye.h"

#include "curand_kernel.h"

//#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined(WIN32)
#pragma warning(push)
#pragma warning(disable : 4267)
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined(WIN32)
#pragma warning(pop)
#endif

namespace cray
{
    // Compile time debugging choices
    static constexpr bool debug_gltf = false;
    static constexpr bool debug_cameras = true;
    static constexpr bool debug_pipeline = false;

    namespace local
    {
        const std::vector<std::string> splitString (const std::string& s, const std::string& delim)
        {
            std::vector<std::string> output;
            const size_t delimSize = delim.size();
            size_t lastDelimLoc = 0;
            size_t delimLoc = s.find (delim, 0);
            while (delimLoc != std::string::npos) {
                if (delimLoc != lastDelimLoc) {
                    output.push_back (s.substr (lastDelimLoc, delimLoc - lastDelimLoc));
                }
                lastDelimLoc = delimLoc + delimSize;
                delimLoc = s.find (delim, lastDelimLoc);
            }
            // Push either the whole thing if it's not found, or the last segment if there were delims
            output.push_back (s.substr (lastDelimLoc, s.size()));
            return output;
        }
    }

    std::vector<cray::Ommatidium> read_eye_file (const std::string& eye_data_path)
    {
        std::vector<cray::Ommatidium> ommVector = {};

        // Read the lines of the file
        std::ifstream eyeDataFile (eye_data_path, std::ifstream::in);
        if (eyeDataFile.is_open() == false) { return ommVector; }

        std::string line;
        size_t ommCount = 0;
        while (std::getline (eyeDataFile, line)) {
            std::vector<std::string> splitData = cray::local::splitString (line, " "); // position, direction, angle, offset
            cray::Ommatidium o = {
                { std::stof(splitData[0]), std::stof(splitData[1]), std::stof(splitData[2]) },
                { std::stof(splitData[3]), std::stof(splitData[4]), std::stof(splitData[5]) },
                std::stof(splitData[6]), std::stof(splitData[7])
            };
            ommVector.push_back(o);
            ommCount++;
        }
        std::cout <<  "Loaded " << ommCount << " ommatidia." << std::endl;
        eyeDataFile.close();

        return ommVector;
    }

    class MulticamScene
    {
    public:
        cray::LaunchParams* d_params = nullptr;
        cray::LaunchParams* params = nullptr; // hostside now

        struct MeshGroup
        {
            std::string name;
            sutil::Matrix4x4 transform;

            std::vector<cuda::BufferView<std::uint32_t>> indices;
            std::vector<cuda::BufferView<float3>> positions;
            std::vector<cuda::BufferView<float3>> normals;
            std::vector<cuda::BufferView<float2>> texcoords;
            std::vector<cuda::BufferView<float3>> host_colors_f3;
            std::vector<cuda::BufferView<float4>> host_colors_f4;
            std::vector<cuda::BufferView<ushort4>> host_colors_us4;
            std::vector<cuda::BufferView<uchar4>> host_colors_uc4;
            std::vector<int> host_color_types; // -1 = doesn't use vertex colours, 5126 = float4, 5123 = ushort4, 5121 = uchar4
            int host_color_container = -1; // -1 for unknown. 3 for vec3 (and use host_colors_f3)  4 for vec4 (use host_colors_f4 or _us4 or _uc4)

            std::vector<std::int32_t> material_idx;

            OptixTraversableHandle gas_handle = 0;
            CUdeviceptr d_gas_output = 0;

            sutil::Aabb object_aabb;
            sutil::Aabb world_aabb;
        };

        struct HitboxMeshGroup
        {
            std::string name;
            sutil::Matrix4x4 transform;

            std::vector<std::shared_ptr<std::vector<std::uint32_t>>> indices;
            std::vector<std::shared_ptr<std::vector<float3>>> positions;

            sutil::Aabb object_aabb;
            sutil::Aabb world_aabb;
        };

        struct Triangle
        {
            float p1, p2, p3;
        };

        MulticamScene()
        {
            this->params = new cray::LaunchParams{};
        }

        ~MulticamScene()
        {
            this->cleanup();
        }

        std::string getEyeDataPath()
        {
            try {
                return this->eye_data_paths.at (this->getCameraIndex());
            } catch (const std::out_of_range& e) {}
            return std::string("");
        }

        void setCurrentEyeSamplesPerOmmatidium (int s)
        {
            if (this->getCamera() != nullptr) {
                this->getCamera()->setSamplesPerOmmatidium (s);
            }
        }

        int getCurrentEyeSamplesPerOmmatidium()
        {
            if (this->getCamera() != nullptr) {
                return this->getCamera()->getSamplesPerOmmatidium();
            }
            return -1;
        }

        void changeCurrentEyeSamplesPerOmmatidiumBy (int s)
        {
            if (this->getCamera() != nullptr) {
                this->getCamera()->changeSamplesPerOmmatidiumBy (s);
            }
        }

        size_t getCurrentEyeOmmatidialCount()
        {
            if (this->getCamera() != nullptr) {
                return this->getCamera()->getOmmatidialCount();
            }
            return 0u;
        }

        static constexpr bool sum_average_with_getCameraData = false;

        void getCameraData (std::vector<std::array<float, 3>>& cameraData)
        {
            if (this->getCamera() == nullptr) { return; }

            if constexpr (sum_average_with_getCameraData == true) {
                // Alternative place to do the sample summing. Useful here, so that you can time
                // getCameraData() to work out how much time is taken to sum and transfer data to CPU
                this->getCamera()->averageRecordFrame();
            }
            size_t omcount = this->getCamera()->getOmmatidialCount();
            cameraData.resize (omcount);
            float3* _data =  this->getCamera()->getRecordFrame();
            for (size_t i = 0; i < omcount; ++i) {
                // copy _data[i] to cameraData[i] applying gamma correction
                // 1/2.2 = 0.45454545
                //cameraData[i] = { powf(_data[i].x, 1.0f/2.2f), powf(_data[i].y, 1.0f/2.2f), powf(_data[i].z, 1.0f/2.2f) };
                // Check for nans while running; somewhere in the averaging code, we sometimes obtain a NaN
                if (std::isnan (_data[i].x)) { // Only need to check one element for NaN
                    cameraData[i] = { 0.0f, 0.0f, 0.0f };
                } else {
                    cameraData[i] = { _data[i].x, _data[i].y, _data[i].z };
                }
            }
        }

        void rotateCamerasLocallyAround (float angle, float x, float y, float z)
        {
            size_t cc = this->getCameraCount();
            for (size_t i = 0; i < cc; ++i) {
                if (this->getCamera() != nullptr) {
                    this->getCamera()->rotateLocallyAround (angle, make_float3(x,y,z));
                }
                this->nextCamera();
            }
        }

        void setCameraPoseMatrix (const sutil::Matrix4x4& camera_localspace)
        {
            if (this->getCamera() != nullptr) {
                this->getCamera()->setLocalSpace (camera_localspace);
            }
        }

        // Launch Optix threads to render a camera view. Once this is done getCameraData() accesses the
        // summed average values for a compound eye. Non-compound eye data is accessed with
        // getFramePointer()
        void launchFrame()
        {
            // d_params is a (no-longer global) pointer to GPU RAM, params is a (no-longer global) pointer to CPU-side RAM
            CUDA_CHECK (cudaMemcpyAsync (reinterpret_cast<void*>(this->d_params),
                                         this->params,
                                         sizeof(cray::LaunchParams),
                                         cudaMemcpyHostToDevice,
                                         0)); // stream

            if (this->getCamera() != nullptr) {
                cray::CompoundEye* camera = (cray::CompoundEye*) this->getCamera();

                // Launch the ommatidial renderer
                auto ole = optixLaunch (m_compound_pipeline,
                                        0,                                 // stream
                                        reinterpret_cast<CUdeviceptr> (this->d_params), // pipelineParams
                                        sizeof (cray::LaunchParams),       // pipelineParamsSize
                                        &this->m_compound_sbt,             // shader binding table
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

                this->params->frame++; // Increase the frame number
                camera->setRandomsAsConfigured(); // Make sure that random stream initialization is only ever done once

                if constexpr (sum_average_with_getCameraData == false) {
                    // After the compoundray pipeline, we usually call the sample-summing CUDA kernel here
                    camera->averageRecordFrame();
                    CUDA_SYNC_CHECK();
                }
            }

            // No need for outputBuffer unmap here any more

            CUDA_SYNC_CHECK();
        }

        double renderFrame()
        {
            // Make sure the SBT of the scene is updated for the newly selected camera before launch,
            // also push any changed host-side camera SBT data over to the device.
            this->reconfigureSBTforCurrentCamera (false);

            auto then = std::chrono::steady_clock::now();
            this->launchFrame();
            CUDA_SYNC_CHECK();
            std::chrono::duration<double, std::milli> render_time = std::chrono::steady_clock::now() - then;
            return render_time.count();
        }

        void initLaunchParams();

        void loadScene (const std::string& filename, const sutil::Matrix4x4& root_transform);

        void loadGlTFscene (const char* filepath, sutil::Matrix4x4 root_transform)
        {
            this->loadScene (filepath, root_transform);
            this->finalize();
            this->initLaunchParams();
        }

        // Obtain access to a mesh of positions (to scan over a landscape)
        const std::vector<cuda::BufferView<float3> >* getMeshPositions (size_t idx)
        {
            if (idx >= this->m_meshes.size()) { return nullptr; }
            return &this->m_meshes[idx]->positions;
        }
        const std::vector<cuda::BufferView<float3> >* getMeshNormals (size_t idx)
        {
            if (idx >= this->m_meshes.size()) { return nullptr; }
            return &this->m_meshes[idx]->normals;
        }

        std::uint32_t addMesh (std::shared_ptr<MeshGroup> mesh)
        {
            m_meshes.push_back (mesh);
            return (this->m_meshes.size() - 1u);
        }
        void addMaterial (const MaterialData::Pbr& mtl) { m_materials.push_back (mtl); }
        void addBuffer (const std::uint64_t buf_size, const void* data);
        void addImage (const std::int32_t width, const std::int32_t height, const std::int32_t bits_per_component,
                       const std::int32_t num_components, const void* data);
        void addSampler (cudaTextureAddressMode address_s, cudaTextureAddressMode address_t,
                         cudaTextureFilterMode  filter_mode, const std::int32_t image_idx);

        CUdeviceptr getBuffer (std::int32_t buffer_index) const;
        cudaArray_t getImage (std::int32_t image_index) const;
        cudaTextureObject_t getSampler (std::int32_t sampler_index) const;

        void finalize();
        void cleanup();

        //// Camera functions

        // Create a new compound eye
        // Returns the position of the compound camera in the array for later reference
        std::int32_t addCamera (const std::string& cam_name,
                                const std::vector<Ommatidium>* ommVec,
                                const std::string& eye_data_path,
                                const float3& position,
                                const float3& rightAxis,
                                const float3& upAxis,
                                const float3& forwardAxis)
        {
            cray::CompoundEye* camera = new cray::CompoundEye (cam_name, ommVec->size(), eye_data_path);
            camera->setPosition (position);
            camera->setLocalSpace (rightAxis, upAxis, forwardAxis);
            camera->copyOmmatidia (ommVec->data()); // Copies ommVec data to GPU
            return this->addCamera (camera, ommVec, eye_data_path);
        }

        // Create a new compound eye
        // Returns the position of the compound camera in the array for later reference
        std::int32_t addCamera (cray::CompoundEye* cameraPtr, const std::vector<Ommatidium>* ommVec, const std::string& eye_data_path)
        {
            // New camera index. m_compoundEyes is a map with sequential index
            auto cam_idx = static_cast<std::int32_t>(this->m_compoundEyes.size());

            this->m_compoundEyes[cam_idx] = cameraPtr;
            this->m_ommVecs[cam_idx] = *ommVec; // copies ommVec data into a Multicam class member attribute.
            this->eye_data_paths[cam_idx] = eye_data_path;

            if constexpr (debug_cameras == true) {
                std::cout << "Inserted ommVec of size " << m_ommVecs[cam_idx].size()
                          << " into m_ommVecs[" << cam_idx << "] with eye_data_path " << eye_data_path << ".\n";
            }
            return cam_idx;
        }

        void removeCameras()
        {
            std::int32_t sz = getCameraCount();
            for (std::int32_t i = 0; i < sz; ++i) { delete this->m_compoundEyes[i]; }
            this->m_compoundEyes.clear();
            this->m_ommVecs.clear();
            this->eye_data_paths.clear();
        }

        cray::CompoundEye* getCamera() const
        {
            if (!m_compoundEyes.empty()) {
                try {
                    return m_compoundEyes.at (this->currentCamera);
                } catch (const std::out_of_range& e) {
                    return nullptr;
                }
            }
            return nullptr;
        }

        void setCurrentCamera (const std::int32_t index)
        {
            const std::int32_t s = this->getCameraCount();
            this->currentCamera = (index % s + s) % s;
        }

        const std::int32_t getCameraCount() const { return static_cast<std::int32_t>(this->m_compoundEyes.size()); };

        const std::int32_t getCameraIndex() const { return this->currentCamera; }

        void nextCamera() { this->setCurrentCamera (this->currentCamera + 1); }

        void previousCamera() { this->setCurrentCamera (this->currentCamera - 1); }

        void changeCompoundSampleRateBy (int change);

        sutil::Aabb aabb() const { return m_scene_aabb; }
        OptixDeviceContext context() const { return m_context; }
        const std::vector<MaterialData::Pbr>& materials() const { return m_materials; }
        const std::vector<std::shared_ptr<MeshGroup>>& meshes() const { return m_meshes; }

        void createContext();
        void buildMeshAccels (std::uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT);
        void buildInstanceAccel (int rayTypeCount = cray::RAY_TYPE_COUNT);

        // Changes the Shader Binding Table to reflect the current camera (assumes all camera records are allocated)
        void reconfigureSBTforCurrentCamera (bool force);

        // Scene manipulation
        bool isInsideHitGeometry (float3 worldPos, std::string name, bool debug = false);
        float3 getGeometryMaxBounds (std::string name);
        float3 getGeometryMinBounds (std::string name);

        std::string m_backgroundShader = "__miss__default_background";

        std::vector<sutil::hitscan::TriangleMesh> m_hitboxMeshes; // Stores all triangle meshes public, because why the hell not?
        // The CPU side vector of ommatidia used to create each CompoundEye in m_compoundEyes
        std::map<std::int32_t, std::vector<Ommatidium>> m_ommVecs;

        // The eye data file, specified as "compound-structure" for compound eyes. One for each eye.
        std::map<std::int32_t, std::string> eye_data_paths;

        // Obtain a copy of the meshes for external program to do a simple rendering
        std::vector<std::shared_ptr<MeshGroup> > getMeshes() { return m_meshes; }
        std::vector<MaterialData::Pbr> getMaterials() { return m_materials; }

    private:
        void createPTXModule();
        void createProgramGroups();
        void createSBTmissAndHit (OptixShaderBindingTable& sbt);
        void createCompoundPipeline();

        // A map of cameras/compound eyes. Contains pointers to all compound eyes (could be renamed back to m_cameras)
        std::map<std::int32_t, cray::CompoundEye*> m_compoundEyes;

        std::vector<std::shared_ptr<MeshGroup> > m_meshes;
        std::vector<MaterialData::Pbr>       m_materials;
        std::vector<CUdeviceptr>             m_buffers;
        std::vector<cudaTextureObject_t>     m_samplers;
        std::vector<cudaArray_t>             m_images;
        sutil::Aabb                          m_scene_aabb;

        OptixDeviceContext                   m_context                  = 0;
        OptixModule                          m_ptx_module               = 0;
        OptixShaderBindingTable              m_sbt                      = {};
        OptixShaderBindingTable              m_compound_sbt             = {};
        OptixPipeline                        m_compound_pipeline        = 0;
        OptixPipelineCompileOptions          m_pipeline_compile_options = {};

        OptixProgramGroup                    m_compound_raygen_group    = 0;
        OptixProgramGroup                    m_radiance_miss_group      = 0;
        OptixProgramGroup                    m_occlusion_miss_group     = 0;
        OptixProgramGroup                    m_radiance_hit_group       = 0;
        OptixProgramGroup                    m_occlusion_hit_group      = 0;
        OptixProgramGroupOptions             program_group_options      = {};

        OptixTraversableHandle               m_ias_handle               = 0;
        CUdeviceptr                          m_d_ias_output_buffer      = 0;

        std::int32_t                        currentCamera               = 0;
        std::int32_t                        lastPipelinedCamera         = std::numeric_limits<std::int32_t>::max();
    };
} // namespace

#endif
