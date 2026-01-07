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
#include <fstream>
#include <limits>

#include "GlobalParameters.h"
#include "cameras/GenericCameraDataTypes.h"
#include "cameras/GenericCamera.h"
#include "cameras/PerspectiveCamera.h"
#include "cameras/ThreeSixtyCamera.h"
#include "cameras/OrthographicCamera.h"
#include "cameras/CompoundEye.h"

#include "curand_kernel.h"

//#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined( WIN32 )
#pragma warning( push )
#pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#pragma warning( pop )
#endif


using namespace sutil;

class MulticamScene
{
public:
    struct MeshGroup
    {
        std::string                       name;
        Matrix4x4                         transform;

        std::vector<cuda::BufferView<uint32_t>> indices;
        std::vector<cuda::BufferView<float3> >  positions;
        std::vector<cuda::BufferView<float3> >  normals;
        std::vector<cuda::BufferView<float2> >  texcoords;
        std::vector<cuda::BufferView<float3 >>  host_colors_f3;
        std::vector<cuda::BufferView<float4 >>  host_colors_f4;
        std::vector<cuda::BufferView<ushort4>>  host_colors_us4;
        std::vector<cuda::BufferView<uchar4 >>  host_colors_uc4;
        std::vector<int> host_color_types; // -1 = doesn't use vertex colours, 5126 = float4, 5123 = ushort4, 5121 = uchar4
        int              host_color_container = -1; // -1 for unknown. 3 for vec3 (and use host_colors_f3)  4 for vec4 (use host_colors_f4 or _us4 or _uc4)

        std::vector<int32_t>              material_idx;

        OptixTraversableHandle            gas_handle = 0;
        CUdeviceptr                       d_gas_output = 0;

        Aabb                              object_aabb;
        Aabb                              world_aabb;
    };

    struct HitboxMeshGroup
    {
        std::string name;
        Matrix4x4 transform;

        std::vector<std::shared_ptr<std::vector<uint32_t>>> indices;
        std::vector<std::shared_ptr<std::vector<float3>>> positions;

        Aabb object_aabb;
        Aabb world_aabb;
    };

    struct Triangle
    {
        float p1, p2, p3;
    };

    ~MulticamScene();

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

    // Return index of the added camera
    int addCamera  ( GenericCamera* cameraPtr  );
    // Returns the position of the compound camera in the array for later reference
    uint32_t addCompoundCamera  (int camera_index, CompoundEye* cameraPtr, std::vector<Ommatidium>& ommVec);
    uint32_t addMesh    ( std::shared_ptr<MeshGroup> mesh )    {
        m_meshes.push_back( mesh );
        return (this->m_meshes.size() - 1u);
    }
    void addMaterial( const MaterialData::Pbr& mtl    )    { m_materials.push_back( mtl );     }
    void addBuffer  ( const uint64_t buf_size, const void* data );
    void addImage(
        const int32_t width,
        const int32_t height,
        const int32_t bits_per_component,
        const int32_t num_components,
        const void*   data
        );
    void addSampler(
        cudaTextureAddressMode address_s,
        cudaTextureAddressMode address_t,
        cudaTextureFilterMode  filter_mode,
        const int32_t          image_idx
        );

    CUdeviceptr                    getBuffer ( int32_t buffer_index  )const;
    cudaArray_t                    getImage  ( int32_t image_index   )const;
    cudaTextureObject_t            getSampler( int32_t sampler_index )const;

    void                           finalize();
    void                           cleanup();

    //// Camera functions
    // Gets a pointer to the current camera
    GenericCamera*                            getCamera();
    void                                      setCurrentCamera(const int index);
    const size_t                              getCameraCount() const;
    const size_t                              getCameraIndex() const { return currentCamera; }
    void                                      nextCamera();
    void                                      previousCamera();

    //// Compound eye functions (note: similar to others here)
    const bool                                hasCompoundEyes() const      { return ommatidialCameraCount() > 0; }
    const uint32_t                            ommatidialCameraCount() const{ return m_compoundEyes.size(); }
    void                                      checkIfCurrentCameraIsCompound();// Updates flag accessed below
    const bool                                isCompoundEyeActive() const  { return m_selectedCameraIsCompound; }
    void                                      changeCompoundSampleRateBy(int change);


    OptixPipeline                             pipeline()const              { return m_pipeline;   }
    const OptixShaderBindingTable*            sbt()const                   { return &m_sbt;       }
    OptixTraversableHandle                    traversableHandle() const    { return m_ias_handle; }
    sutil::Aabb                               aabb() const                 { return m_scene_aabb; }
    OptixDeviceContext                        context() const              { return m_context;    }
    const std::vector<MaterialData::Pbr>&     materials() const            { return m_materials;  }
    const std::vector<std::shared_ptr<MeshGroup>>& meshes() const          { return m_meshes;     }

    void createContext();
    void buildMeshAccels( uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT );
    void buildInstanceAccel( int rayTypeCount = globalParameters::RAY_TYPE_COUNT );

    // Changes the SBT to refelct the current camera (assumes all camera records are allocated)
    void reconfigureSBTforCurrentCamera(bool force);

    // OptixPipeline is a ptr to an opaque struct
    OptixPipeline compoundPipeline() const { return m_compound_pipeline; }
    const OptixShaderBindingTable* compoundSbt() const { return &m_compound_sbt; }

    // Scene manipulation
    bool isInsideHitGeometry(float3 worldPos, std::string name, bool debug = false);
    float3 getGeometryMaxBounds(std::string name);
    float3 getGeometryMinBounds(std::string name);

    std::string                          m_backgroundShader         = "__miss__default_background";

    std::vector<sutil::hitscan::TriangleMesh>         m_hitboxMeshes; // Stores all triangle meshes public, because why the hell not?
    // The CPU side vector of ommatidia used to create each CompoundEye in m_compoundEyes
    std::map<int, std::vector<Ommatidium>> m_ommVecs;

    // The eye data file, specified as "compound-structure" for compound eyes
    std::string eye_data_path = "";

    // The newGuiEyeRenderer requires an additional optix pipeline for the panoramic
    // rendering and ALSO to render compound eyes.
    bool require_noncompound_pipeline = false;

    // Obtain a copy of the meshes for external program to do a simple rendering
    std::vector<std::shared_ptr<MeshGroup> > getMeshes() { return m_meshes; }
    std::vector<MaterialData::Pbr> getMaterials() { return m_materials; }

private:
    void createPTXModule();
    void createProgramGroups();
    void createPipeline();
    void createSBTmissAndHit(OptixShaderBindingTable& sbt);

    void createCompoundPipeline();

    std::map<int, GenericCamera*>        m_cameras;// cameras is a map of pointers to Camera objects.
    std::vector<std::shared_ptr<MeshGroup> > m_meshes;
    std::vector<MaterialData::Pbr>       m_materials;
    std::vector<CUdeviceptr>             m_buffers;
    std::vector<cudaTextureObject_t>     m_samplers;
    std::vector<cudaArray_t>             m_images;
    sutil::Aabb                          m_scene_aabb;

    OptixDeviceContext                   m_context                  = 0;
    OptixShaderBindingTable              m_sbt                      = {};
    OptixPipelineCompileOptions          m_pipeline_compile_options = {};
    OptixPipeline                        m_pipeline                 = 0;
    OptixModule                          m_ptx_module               = 0;

    // Compound eye stuff (A lot of these are stored precomp values so they don't have to be recomputed every frame)

    // Contains pointers to all compound eyes (shared with the m_cameras vector). Might make more
    // sense for this to be map<int, CompoundEye*>
    std::map<int, CompoundEye*>          m_compoundEyes;

    OptixShaderBindingTable              m_compound_sbt             = {};
    OptixPipeline                        m_compound_pipeline        = 0;
    OptixProgramGroup                    m_compound_raygen_group    = 0;
    bool                                 m_selectedCameraIsCompound = false;

    OptixProgramGroup                    m_raygen_prog_group = 0;
    OptixProgramGroupDesc                raygen_prog_group_desc = {};
    OptixProgramGroupOptions             program_group_options = {};

    OptixProgramGroup                    m_pinhole_raygen_prog_group= 0;
    OptixProgramGroup                    m_ortho_raygen_prog_group  = 0;
    OptixProgramGroup                    m_radiance_miss_group      = 0;
    OptixProgramGroup                    m_occlusion_miss_group     = 0;
    OptixProgramGroup                    m_radiance_hit_group       = 0;
    OptixProgramGroup                    m_occlusion_hit_group      = 0;
    OptixTraversableHandle               m_ias_handle               = 0;
    CUdeviceptr                          m_d_ias_output_buffer      = 0;

    size_t                               currentCamera              = 0;
    size_t                               lastPipelinedCamera        = std::numeric_limits<size_t>::max();
};


void loadScene (const std::string& filename, MulticamScene& scene, const Matrix4x4& root_transform);

#endif
