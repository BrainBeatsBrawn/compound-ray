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
//        This file was originally based on the "Scene.cpp" file
//        that comes within sutil of the NVidia OptiX SDK, but has
//        been changed by Blayze Millward to be more aligned to the
//        design schema of the insect eye perspective renderer.

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <sutil/Exception.h>
#include <sutil/Matrix.h>
#include <sutil/Quaternion.h>
#include <sutil/Record.h>
#include <sutil/sutil.h>

#include "MulticamScene.h"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#if defined( WIN32 )
#pragma warning( push )
#pragma warning( disable : 4267 )
#endif
#include <support/tinygltf/tiny_gltf.h>
#if defined( WIN32 )
#pragma warning( pop )
#endif

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <iostream>

// Using a handful of GL definitions
#ifndef GL_MIRRORED_REPEAT
# define GL_MIRRORED_REPEAT 0x8370
#endif

#ifndef GL_CLAMP_TO_EDGE
# define GL_CLAMP_TO_EDGE 0x812F
#endif

#ifndef GL_NEAREST
# define GL_NEAREST 0x2600
#endif

namespace
{
    // Compile time debugging choices
    static constexpr bool debug_gltf = false;
    static constexpr bool debug_cameras = false;
    static constexpr bool debug_pipeline = false;

    float3 make_float3_from_double( double x, double y, double z )
    {
        return make_float3( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ) );
    }

    float4 make_float4_from_double( double x, double y, double z, double w )
    {
        return make_float4( static_cast<float>( x ), static_cast<float>( y ), static_cast<float>( z ), static_cast<float>( w ) );
    }

    typedef Record<globalParameters::HitGroupData> HitGroupRecord;

    static constexpr bool debug_allow_context_log = false;
    void context_log_cb( unsigned int level, const char* tag, const char* message, void* /*cbdata */)
    {
        if constexpr (debug_allow_context_log) {
            std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
                      << message << "\n";
        }
    }

    static constexpr bool debug_bufferview = false;
    static constexpr bool debug_bufferview_byteoffsets = false;
    static constexpr bool debug_bufferview_full = false;
    /*
     * This function obtains a CUDA BufferView of the data that is associated with the glTF accessor
     * in @model with index @accessor_idx. The glTF accessor provides access to/additional metadata
     * about memory described by a glTF BufferView (which should not be confused with the CUDA
     * BufferView object returned by this function). @scene provides access to the buffer of data
     * that underlies the BufferViews.
     */
    template<typename T>
    cuda::BufferView<T> bufferViewFromGLTF(const tinygltf::Model& model, MulticamScene& scene, const int32_t accessor_idx)
    {
        if (accessor_idx == -1) { return cuda::BufferView<T>(); }

        const tinygltf::Accessor& gltf_accessor      = model.accessors[accessor_idx];
        const tinygltf::BufferView& gltf_buffer_view = model.bufferViews[gltf_accessor.bufferView];

        const int32_t elmt_cmpt_byte_size = tinygltf::GetComponentSizeInBytes(gltf_accessor.componentType);
        const int32_t cmpts_in_type  = tinygltf::GetNumComponentsInType(gltf_accessor.type);

        if constexpr (debug_bufferview_full) {
            std::cout << "elmt_cmpt_byte_size from accessor.componentType: " << elmt_cmpt_byte_size << std::endl;
            std::cout << "cmpts_in_type from accessor.type: " << cmpts_in_type << std::endl;
            std::cout << "count from accessor: " << gltf_accessor.count << std::endl;
        }

        if (cmpts_in_type == -1 || elmt_cmpt_byte_size == -1) { throw Exception ("gltf accessor not supported"); }
        if ((cmpts_in_type * elmt_cmpt_byte_size) > sizeof(T)) { throw Exception ("bufferViewFromGLTF: sizeof(T) < accessor data type size"); }

        const CUdeviceptr buffer_base = scene.getBuffer (gltf_buffer_view.buffer);
        cuda::BufferView<T> buffer_view;
        if constexpr (debug_bufferview_byteoffsets) {
            std::cout << "gltf_accessor.byteOffset    = " << gltf_accessor.byteOffset << std::endl;
            std::cout << "gltf_buffer_view.byteOffset = " << gltf_buffer_view.byteOffset << std::endl;
            std::cout << "gltf_accessor.byteOffset % 16   = " << (gltf_accessor.byteOffset % 16) << std::endl;
            std::cout << "gltf_buffer_view.byteOffset % 16 = " << (gltf_buffer_view.byteOffset % 16) << std::endl;
        }
        buffer_view.data           = buffer_base + gltf_buffer_view.byteOffset + gltf_accessor.byteOffset;

        if constexpr (debug_bufferview_byteoffsets) {
            std::cout << "data pointer % 16: " << (static_cast<size_t>(buffer_view.data) % 16) << std::endl;
        }

        buffer_view.byte_stride    = static_cast<uint16_t>(gltf_buffer_view.byteStride);
        // A cuda::BufferView::count is a count of objects of type T (which is the one that really makes sense)
        buffer_view.count          = static_cast<uint32_t>(gltf_accessor.count);
        buffer_view.elmt_byte_size = static_cast<uint16_t>(elmt_cmpt_byte_size * cmpts_in_type);

        if constexpr (debug_bufferview) {
            std::cout << "Returning buffer_view with .count: " << buffer_view.count
                      << " .elmt_byte_size: " <<  buffer_view.elmt_byte_size
                      << " sizeof(T): " << sizeof(T)
                      << " .byte_stride: " << buffer_view.byte_stride
                      << " number of objects: " << buffer_view.count
                      << " consuming: " << buffer_view.size_bytes() << " bytes in RAM"
                      << std::endl;
        }
        return buffer_view;
    }

    const bool isObjectsExtraValueTrue (const tinygltf::Value& extras, const char* key)
    {
        tinygltf::Value v = extras.Get(key);
        if(v.IsBool())
        {
            return v.Get<bool>();
        }

        if(v.IsString())
        {
            std::string valueStr = v.Get<std::string>();
            std::transform(valueStr.begin(), valueStr.end(), valueStr.begin(), [](unsigned char c){ return std::tolower(c); });
            return (valueStr.compare("true") == 0);
        }
        return false;
    }
    const std::vector<std::string> splitString(const std::string& s, const std::string& deliminator)
    {
        std::vector<std::string> output;
        const size_t delimSize = deliminator.size();
        size_t lastDelimLoc = 0;
        size_t delimLoc = s.find(deliminator, 0);
        while(delimLoc != std::string::npos)
        {
            if(delimLoc != lastDelimLoc)
                output.push_back(s.substr(lastDelimLoc, delimLoc-lastDelimLoc));
            lastDelimLoc = delimLoc + delimSize;
            delimLoc = s.find(deliminator, lastDelimLoc);
        }
        // Push either the whole thing if it's not found, or the last segment if there were deliminators
        output.push_back(s.substr(lastDelimLoc, s.size()));
        return output;
    }

    // Global function called from loadScene
    void processGLTFNode(
        MulticamScene& scene,
        const tinygltf::Model& model,
        const tinygltf::Node& gltf_node,
        const Matrix4x4& parent_matrix,
        const std::string& glTFdir
        )
    {
        const Matrix4x4 translation = gltf_node.translation.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::translate( make_float3_from_double(
                                  gltf_node.translation[0],
                                  gltf_node.translation[1],
                                  gltf_node.translation[2]
                                  ) );

        const Matrix4x4 rotation = gltf_node.rotation.empty() ?
        Matrix4x4::identity() :
        Quaternion(
            static_cast<float>( gltf_node.rotation[3] ),
            static_cast<float>( gltf_node.rotation[0] ),
            static_cast<float>( gltf_node.rotation[1] ),
            static_cast<float>( gltf_node.rotation[2] )
            ).rotationMatrix();

        const Matrix4x4 scale = gltf_node.scale.empty() ?
        Matrix4x4::identity() :
        Matrix4x4::scale( make_float3_from_double(
                              gltf_node.scale[0],
                              gltf_node.scale[1],
                              gltf_node.scale[2]
                              ) );

        std::vector<float> gltf_matrix;
        for( double x : gltf_node.matrix )
            gltf_matrix.push_back( static_cast<float>( x ) );
        const Matrix4x4 matrix = gltf_node.matrix.empty() ?
        Matrix4x4::identity() :
        Matrix4x4( reinterpret_cast<float*>( gltf_matrix.data() ) ).transpose();

        const Matrix4x4 node_xform = parent_matrix * matrix * translation * rotation * scale ;

        if( gltf_node.camera != -1 )
        {
            // We're dealing with cameras
            const auto& gltf_camera = model.cameras[ gltf_node.camera ];
            if constexpr (debug_gltf == true) {
                std::cout << "============================"<<std::endl<<"Processing camera '" << gltf_camera.name << "'" << std::endl
                          << "\ttype: " << gltf_camera.type << std::endl;
            }
            // Get configured camera information and local axis
            const float3 upAxis      = make_float3 (node_xform * make_float4_from_double (0.0f, 1.0f,  0.0f, 0.0f)); //  uy
            const float3 forwardAxis = make_float3 (node_xform * make_float4_from_double (0.0f, 0.0f, -1.0f, 0.0f)); // -uz
            const float3 rightAxis   = make_float3 (node_xform * make_float4_from_double (1.0f, 0.0f,  0.0f, 0.0f)); //  ux

            if constexpr (debug_cameras == true) {
                std::cout << "\tUP axis: (" << upAxis.x <<"," << upAxis.y << "," << upAxis.z << ")" << std::endl;
                std::cout << "\tFWD axis: (" << forwardAxis.x <<"," << forwardAxis.y << "," << forwardAxis.z << ")" << std::endl;
                std::cout << "\tR axis: (" << rightAxis.x <<"," << rightAxis.y << "," << rightAxis.z << ")" << std::endl;
            }

            // eye is 'position' - a transform of the origin
            const float3 eye     = make_float3( node_xform*make_float4_from_double( 0.0f, 0.0f,  0.0f, 1.0f ) );
            const float  yfov   = static_cast<float>( gltf_camera.perspective.yfov ) * 180.0f / static_cast<float>( M_PI );
            if constexpr (debug_cameras == true) {
                std::cout << "\teye posn: " << eye.x    << ", " << eye.y    << ", " << eye.z    << std::endl;
                std::cout << "\tfov     : " << yfov     << std::endl;
                std::cout << "\taspect  : " << gltf_camera.perspective.aspectRatio << std::endl;
            }
            // Form camera objects
            if( gltf_camera.type == "orthographic" )
            {
                OrthographicCamera* camera = new OrthographicCamera(gltf_camera.name);
                camera->setPosition(eye);
                camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
                camera->setXYscale(gltf_camera.orthographic.xmag, gltf_camera.orthographic.ymag);
                int cidx = scene.addCamera(camera);
                if constexpr (debug_cameras == true) {
                    std::cout << "Added orthographic camera " << cidx << std::endl;
                }
                return;
            }

            if(isObjectsExtraValueTrue(gltf_camera.extras, "panoramic"))
            {
                if constexpr (debug_cameras == true) {
                    std::cout << "This camera has special indicator 'panoramic' specified, adding panoramic camera..."<<std::endl;
                }
                PanoramicCamera* camera = new PanoramicCamera(gltf_camera.name);
                camera->setPosition(eye);
                camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
                int cidx = scene.addCamera(camera);
                if constexpr (debug_cameras == true) {
                    std::cout << "Added panorama camera " << cidx << std::endl;
                }
                return;
            }

            if(isObjectsExtraValueTrue(gltf_camera.extras, "compound-eye"))
            {
                if constexpr (debug_cameras == true) {
                    std::cout << "This camera has special indicator 'compound-eye' specified, adding compound eye based camera..."<<std::endl;
                }
                std::string eyeDataPath = gltf_camera.extras.Get("compound-structure").Get<std::string>();
                std::string projectionShader = gltf_camera.extras.Get("compound-projection").Get<std::string>();
                if constexpr (debug_cameras == true) {
                    std::cout << "  Camera internal projection type: "<<projectionShader<<std::endl;
                    std::cout << "  Camera eye data path: "<<eyeDataPath<<std::endl;
                }

                if(eyeDataPath == "")
                {
                    std::cerr << "ERROR: Eye data path empty or non-existant." << std::endl;
                    return;
                }
                if(projectionShader == "")
                {
                    std::cerr << "ERROR: Projection shader specifier empty or non-existant." << std::endl;
                    return;
                }

                // Try and load the file as an absolute (or relative to the execution of the eye)
                std::ifstream eyeDataFile(eyeDataPath, std::ifstream::in);
                std::string usedEyeDataPath; // Track the actual complete path that was used
                if(!eyeDataFile.is_open())
                {
                    if constexpr (debug_cameras == true) {
                        std::cerr << "WARNING: Unable to open \"" << eyeDataPath << "\", attempting to open at relative address..."<<std::endl;
                    }
                    // Try and load the file relatively to the gltf file
                    std::string relativeEyeDataPath = glTFdir + eyeDataPath; // Just append the eye data path
                    eyeDataFile.open(relativeEyeDataPath, std::ifstream::in);
                    if(!eyeDataFile.is_open())
                    {
                        std::cerr << "ERROR: Unable to open \"" << relativeEyeDataPath << "\", read cancelled."<<std::endl;
                        scene.eye_data_path = relativeEyeDataPath;
                        return;
                    }else{
                        if constexpr (debug_cameras == true) {
                            std::cout << "Reading from " << relativeEyeDataPath << "..." << std::endl;
                        }
                        usedEyeDataPath = relativeEyeDataPath;
                        scene.eye_data_path = usedEyeDataPath;
                    }
                }else{
                    if constexpr (debug_cameras == true) {
                        std::cout << "Reading from " << eyeDataPath << "..." << std::endl;
                    }
                    usedEyeDataPath = eyeDataPath;
                    scene.eye_data_path = usedEyeDataPath;
                }

                // Read the lines of the file
                std::string line;
                std::vector<Ommatidium> ommVector;// Stores the ommatidia
                size_t ommCount = 0;
                while(std::getline(eyeDataFile, line))
                {
                    std::vector<std::string> splitData = splitString(line, " ");// position, direction, angle, offset
                    Ommatidium o = {{std::stof(splitData[0]), std::stof(splitData[1]), std::stof(splitData[2])}, {std::stof(splitData[3]), std::stof(splitData[4]), std::stof(splitData[5])}, std::stof(splitData[6]), std::stof(splitData[7]) };
                    ommVector.push_back(o);
                    ommCount++;
                }
                std::cout <<  "  Loaded " << ommCount << " ommatidia." << std::endl;

                if(ommCount == 0)
                {
                    std::cerr << "  ERROR: Zero ommatidia loaded. Are you specifying the right path? (Check previous 'Reading from...' output)" << std::endl;
                    return;
                }

                // Create a new compound eye
                CompoundEye* camera = new CompoundEye(gltf_camera.name, projectionShader, ommVector.size(), usedEyeDataPath);
                camera->setPosition(eye);
                camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
                int cidx = scene.addCamera(camera);
                camera->copyOmmatidia(ommVector.data());
                scene.addCompoundCamera(cidx, camera, ommVector);

                eyeDataFile.close();

                return;
            }

            PerspectiveCamera* camera = new PerspectiveCamera(gltf_camera.name);
            camera->setPosition(eye);
            camera->setLocalSpace(rightAxis, upAxis, forwardAxis);
            camera->setYFOV(yfov);
            int cidx = scene.addCamera( camera );
            if constexpr (debug_cameras == true) {
                std::cout << "Added perspective camera..." << cidx << std::endl;
            }
        }
        else if( gltf_node.mesh != -1 && isObjectsExtraValueTrue(model.meshes[gltf_node.mesh].extras, "hitbox") )
        {
            // Process a hitbox mesh
            const auto& gltf_mesh = model.meshes[ gltf_node.mesh ];
            if constexpr (debug_gltf == true) {
                std::cerr << "Processing glTF mesh as Hitbox mesh: '" << gltf_mesh.name << "'\n";
                std::cerr << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;
            }
            //// Add a triangle mesh to the hitbox mesh list
            sutil::hitscan::TriangleMesh tm;
            tm.name = gltf_mesh.name;
            tm.transform = node_xform;
            sutil::hitscan::populateTriangleMesh(tm, gltf_mesh, model); // Populate the triangle
            sutil::hitscan::calculateObjectAabb(tm);
            sutil::hitscan::calculateWorldAabbUsingTransformAndObjectAabb(tm);
            //tm.print(); // Print for debugging
            scene.m_hitboxMeshes.push_back(tm); // Add it to the list
        }
        else if( gltf_node.mesh != -1 )
        {
            const auto& gltf_mesh = model.meshes[ gltf_node.mesh ];
            if constexpr (debug_gltf == true) {
                std::cerr << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";
                std::cerr << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;
            }
            for( auto& gltf_primitive : gltf_mesh.primitives )
            {
                if( gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES ) // Ignore non-triangle meshes
                {
                    // TODO: Add support for GL_LINE_STRIP mode here.
                    std::cerr << "\tNon-triangle primitive: skipping\n";
                    continue;
                }

                auto mesh = std::make_shared<MulticamScene::MeshGroup>();

                // Add the mesh to the mesh list
                int m_idx = scene.addMesh (mesh);
                if constexpr (debug_gltf == true) {
                    std::cout << "\tThis is m_meshes index " << m_idx << std::endl;
                }

                mesh->name = gltf_mesh.name;
                mesh->indices.push_back( bufferViewFromGLTF<uint32_t>( model, scene, gltf_primitive.indices ) );
                mesh->material_idx.push_back( gltf_primitive.material );
                mesh->transform = node_xform;
                if constexpr (debug_gltf == true) {
                    std::cerr << "\t\tNum triangles is indices.count/3: " << mesh->indices.back().count / 3 << std::endl;
                }
                assert( gltf_primitive.attributes.find( "POSITION" ) !=  gltf_primitive.attributes.end() );
                const int32_t pos_accessor_idx =  gltf_primitive.attributes.at( "POSITION" );
                mesh->positions.push_back( bufferViewFromGLTF<float3>( model, scene, pos_accessor_idx ) );
                if constexpr (debug_gltf == true) {
                    std::cerr << "\t\tNum vertices(positions count/3): " << mesh->positions.back().count / 3 << std::endl;
                }

                const auto& pos_gltf_accessor = model.accessors[ pos_accessor_idx ];
                mesh->object_aabb = Aabb(
                    make_float3_from_double(
                        pos_gltf_accessor.minValues[0],
                        pos_gltf_accessor.minValues[1],
                        pos_gltf_accessor.minValues[2]
                        ),
                    make_float3_from_double(
                        pos_gltf_accessor.maxValues[0],
                        pos_gltf_accessor.maxValues[1],
                        pos_gltf_accessor.maxValues[2]
                        ) );
                mesh->world_aabb = mesh->object_aabb;
                mesh->world_aabb.transform( node_xform );

                auto normal_accessor_iter = gltf_primitive.attributes.find( "NORMAL" ) ;
                if( normal_accessor_iter  !=  gltf_primitive.attributes.end() )
                {
                    if constexpr (debug_gltf == true) {
                        std::cerr << "\t\tHas vertex normals: true\n";
                    }
                    mesh->normals.push_back( bufferViewFromGLTF<float3>( model, scene, normal_accessor_iter->second ) );
                }
                else
                {
                    if constexpr (debug_gltf == true) {
                        std::cerr << "\t\tHas vertex normals: false\n";
                    }
                    mesh->normals.push_back( bufferViewFromGLTF<float3>( model, scene, -1 ) );
                }

                auto texcoord_accessor_iter = gltf_primitive.attributes.find( "TEXCOORD_0" ) ;

                if (texcoord_accessor_iter != gltf_primitive.attributes.end()) {
                    if constexpr (debug_gltf == true) { std::cerr << "\t\tHas texcoords: true\n"; }
                    mesh->texcoords.push_back (bufferViewFromGLTF<float2> (model, scene, texcoord_accessor_iter->second));
                } else {
                    if constexpr (debug_gltf == true) { std::cerr << "\t\tHas texcoords: false\n"; }
                    mesh->texcoords.push_back (bufferViewFromGLTF<float2> (model, scene, -1));
                }

                auto vertex_colours_accessor_iter = gltf_primitive.attributes.find( "COLOR_0" ) ;

                if(vertex_colours_accessor_iter != gltf_primitive.attributes.end() ) // TODO: UNFIX
                {
                    if constexpr (debug_gltf == true) {
                        std::cerr << "\t\tHas vertex colours: true (so we're using them)\n";
                    }
                    // TODO: Add support for vec3 vertex colours here.
                    // Check that the vertex colours are 4-component:
                    const tinygltf::Accessor& vertex_colours_gltf_accessor = model.accessors[ vertex_colours_accessor_iter->second ];

                    if (vertex_colours_gltf_accessor.type == TINYGLTF_TYPE_VEC4) {

                        // const tinygltf::BufferView& colour_buffer_view = model.bufferViews[ vertex_colours_gltf_accessor.bufferView ]; // currently unused
                        // const tinygltf::Buffer& colour_buffer = model.buffers[ colour_buffer_view.buffer ]; // currently unused

                        // Determine the type and component type of the vertex_colours_gltf_accessor
                        // const int numComponents = tinygltf::GetNumComponentsInType(vertex_colours_gltf_accessor.type); // currently unused
                        int componentType = vertex_colours_gltf_accessor.componentType;

                        // TODO: Consider using `isObjectsExtraValueTrue(model.meshes[gltf_node.mesh].extras, "vertex-colours") here
                        //       to determine whether to use vertex colours or not.
                        switch(componentType)
                        {
                        case TINYGLTF_COMPONENT_TYPE_FLOAT:
                            if constexpr (debug_gltf == true) {
                                std::cerr << "\t\t\tColour vec4 component type is float.\n";
                            }
                            mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, vertex_colours_accessor_iter->second ) );

                            // We must populate the other buffers so that indices align
                            mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                            mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                            mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1) );
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                            if constexpr (debug_gltf == true) {
                                std::cerr << "\t\t\tColour vec4 component type is unsigned short.\n";
                            }
                            mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, vertex_colours_accessor_iter->second ) );

                            // We must populate the other buffers so that indices align
                            mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                            mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                            mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1) );
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                            if constexpr (debug_gltf == true) {
                                std::cerr << "\t\t\tColour vec4 component type is unsigned byte.\n";
                            }
                            mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, vertex_colours_accessor_iter->second ) );

                            // We must populate the other buffers so that indices align
                            mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                            mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                            mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                            break;
                        default:
                            if constexpr (debug_gltf == true) {
                                std::cerr << "\t\t\tColour vec4 component type is not supported.\n";
                            }
                            // We must populate the other buffers so that indices align
                            mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1 ) );
                            mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                            mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                            mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                            componentType = -1;
                            break;
                        }
                        mesh->host_color_types.push_back(componentType);
                        if constexpr (debug_gltf == true) {
                            std::cerr << "\t\tmesh->host_color_types.push_back(" << componentType << ");\n";
                        }
                        if (mesh->host_color_container == -1) {
                            mesh->host_color_container = 4;
                        }
                        if (mesh->host_color_container != 4) {
                            std::cerr << "\t\t\tBAD vec4 colour container size!.\n";
                        }
                    }
                    else if (vertex_colours_gltf_accessor.type == TINYGLTF_TYPE_VEC3)
                    {
                        if constexpr (debug_gltf == true) {
                            std::cerr << "\t\t\tWarning: Vertex colours are of type vec3.\n";
                        }
                        // const tinygltf::BufferView& colour_buffer_view = model.bufferViews[ vertex_colours_gltf_accessor.bufferView ]; // unused
                        // const tinygltf::Buffer& colour_buffer = model.buffers[ colour_buffer_view.buffer ]; // unused

                        // Determine the type and component type of the vertex_colours_gltf_accessor
                        // const int numComponents = tinygltf::GetNumComponentsInType(vertex_colours_gltf_accessor.type); // unused
                        int componentType = vertex_colours_gltf_accessor.componentType;

                        // TODO: Consider using `isObjectsExtraValueTrue(model.meshes[gltf_node.mesh].extras, "vertex-colours") here
                        //       to determine whether to use vertex colours or not.
                        switch(componentType)
                        {
                        case TINYGLTF_COMPONENT_TYPE_FLOAT:
                            if constexpr (debug_gltf == true) {
                                std::cerr << "\t\t\tColour vec3 component type is float.\n";
                            }
                            mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, vertex_colours_accessor_iter->second ) );

                            // We must populate the other buffers so that indices align
                            mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                            mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                            mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1) );
                            break;
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                        default:
                            std::cerr << "\t\t\tThis vec3 component type is not supported.\n";
                            // We must populate the other buffers so that indices align
                            mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1 ) );
                            mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                            mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                            mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                            componentType = -1;
                            break;
                        }
                        mesh->host_color_types.push_back(componentType);
                        if constexpr (debug_gltf == true) {
                            std::cerr << "\t\tmesh->host_color_types.push_back(" << componentType << ");\n";
                        }

                        if (mesh->host_color_container == -1) {
                            mesh->host_color_container = 3;
                        }
                        if (mesh->host_color_container != 3) {
                            std::cerr << "\t\t\tBAD vec3 colour container size!.\n";
                        }
                    }
                    else
                    {
                        std::cerr << "\t\t\tWarning: Vertex colours are not of type vec3 or vec4. Ignoring vertex colours.\n";
                        mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1 ) );
                        mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                        mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                        mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                        if constexpr (debug_gltf == true) {
                            std::cerr << "\t\tmesh->host_color_types.push_back(-1);\n";
                        }
                        mesh->host_color_types.push_back(-1);
                    }

                }
                else
                {
                    if constexpr (debug_gltf == true) {
                        std::cerr << "\t\tHas vertex colours: false\n";
                    }
                    mesh->host_color_types.push_back(-1);
                    if constexpr (debug_gltf == true) {
                        std::cerr << "\t\tmesh->host_color_types.push_back(-1);\n";
                    }
                    // We must populate the other buffers so that indices align
                    mesh->host_colors_uc4.push_back( bufferViewFromGLTF<uchar4>( model, scene, -1 ) );
                    mesh->host_colors_f3.push_back( bufferViewFromGLTF<float3>( model, scene, -1) );
                    mesh->host_colors_f4.push_back( bufferViewFromGLTF<float4>( model, scene, -1) );
                    mesh->host_colors_us4.push_back( bufferViewFromGLTF<ushort4>( model, scene, -1) );
                }
            }
        }
        else if( !gltf_node.children.empty() )
        {
            for( int32_t child : gltf_node.children )
            {
                processGLTFNode( scene, model, model.nodes[child], node_xform, glTFdir);
            }
        }
    }

} // end anon namespace


// Load a scene from filename. Apply root_transform (which may be identity, or a transform to
// convert from y-up (GLTF) to z-up (Blender-agreeable)
void loadScene (const std::string& filename, MulticamScene& scene, const Matrix4x4& root_transform)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile( &model, &err, &warn, filename );
    if( !warn.empty() )
        std::cerr << "glTF WARNING: " << warn << std::endl;
    if( !ret )
    {
        std::cerr << "Failed to load GLTF scene '" << filename << "': " << err << std::endl;
        throw Exception( err.c_str() );
    }

    // Calculate and store the path to the file bar the file iteself for relative includes
    std::string glTFdir = "";
    std::size_t slashPos = filename.find_last_of("/\\")+1; // (+1 to include the slash)
    if(slashPos != std::string::npos)
        glTFdir = filename.substr(0,slashPos);

    // Retrieve background shader information if it exists
    if constexpr (debug_gltf == true) {
        std::cout << "Searching for background shader..." << std::endl;
    }
    for(auto modelScene : model.scenes)
    {
        std::string bgShader = modelScene.extras.Get("background-shader").Get<std::string>();
        if constexpr (debug_gltf == true) {
            std::cout << "\tBackground shader string detected: \"" << bgShader << "\"" << std::endl;
        }

        if(bgShader != "")
        {
            scene.m_backgroundShader = "__miss__" + bgShader;
        }
    }
    if constexpr (debug_gltf == true) {
            std::cout << "Background shader set to: \"" << scene.m_backgroundShader << "\"" << std::endl;
    }


    //
    // Process buffer data first -- buffer views will reference this list
    //
    for( const auto& gltf_buffer : model.buffers )
    {
        const uint64_t buf_size = gltf_buffer.data.size();
        if constexpr (debug_gltf == true) {
            std::cerr << "Processing glTF buffer '" << gltf_buffer.name << "'\n"
                      << "\tbyte size: " << buf_size << "\n"
                      << "\turi      : " << (buf_size > 128u ? gltf_buffer.uri.substr(0, 128) + std::string("...") : gltf_buffer.uri) << std::endl;
        }
        scene.addBuffer( buf_size,  gltf_buffer.data.data() );
    }

    //
    // Images -- just load all up front for simplicity
    //
    for( const auto& gltf_image : model.images )
    {
        if constexpr (debug_gltf == true) {
            std::cerr << "Processing image '" << gltf_image.name << "'\n"
                      << "\t(" << gltf_image.width << "x" << gltf_image.height << ")x" << gltf_image.component << "\n"
                      << "\tbits: " << gltf_image.bits << std::endl;
        }
        assert( gltf_image.component == 4 );
        assert( gltf_image.bits      == 8 || gltf_image.bits == 16 );

        scene.addImage(
            gltf_image.width,
            gltf_image.height,
            gltf_image.bits,
            gltf_image.component,
            gltf_image.image.data()
            );
    }

    //
    // Textures -- refer to previously loaded images
    //
    for( const auto& gltf_texture : model.textures )
    {
        if( gltf_texture.sampler == -1 )
        {
            scene.addSampler( cudaAddressModeWrap, cudaAddressModeWrap, cudaFilterModeLinear, gltf_texture.source );
            continue;
        }

        const auto& gltf_sampler = model.samplers[ gltf_texture.sampler ];

        const cudaTextureAddressMode address_s = gltf_sampler.wrapS == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
        gltf_sampler.wrapS == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
        cudaAddressModeWrap;
        const cudaTextureAddressMode address_t = gltf_sampler.wrapT == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
        gltf_sampler.wrapT == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
        cudaAddressModeWrap;
        const cudaTextureFilterMode  filter    = gltf_sampler.minFilter == GL_NEAREST     ? cudaFilterModePoint   :
        cudaFilterModeLinear;
        scene.addSampler( address_s, address_t, filter, gltf_texture.source );
    }

    //
    // Materials
    //
    for( auto& gltf_material : model.materials )
    {
        if constexpr (debug_gltf == true) {
            std::cerr << "Processing glTF material: '" << gltf_material.name << "'\n";
        }
        MaterialData::Pbr mtl;

        {
            const auto base_color_it = gltf_material.values.find( "baseColorFactor" );
            if( base_color_it != gltf_material.values.end() )
            {
                const tinygltf::ColorValue c = base_color_it->second.ColorFactor();
                mtl.base_color = make_float4_from_double( c[0], c[1], c[2], c[3] );
                if constexpr (debug_gltf == true) {
                    std::cerr
                    << "\tBase color: ("
                    << mtl.base_color.x << ", "
                    << mtl.base_color.y << ", "
                    << mtl.base_color.z << ")\n";
                }
            }
            else
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tUsing default base color factor\n";
                }
            }
        }

        {
            const auto base_color_it = gltf_material.values.find( "baseColorTexture" );
            if( base_color_it != gltf_material.values.end() )
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tFound base color texture: " << base_color_it->second.TextureIndex() << "\n";
                }
                mtl.base_color_tex = scene.getSampler( base_color_it->second.TextureIndex() );
            }
            else
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tNo base color texture, mtl.base_color_tex = 0\n";
                }
                mtl.base_color_tex = 0;
            }
        }

        {
            const auto roughness_it = gltf_material.values.find( "roughnessFactor" );
            if( roughness_it != gltf_material.values.end() )
            {
                mtl.roughness = static_cast<float>( roughness_it->second.Factor() );
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tRougness:  " << mtl.roughness <<  "\n";
                }
            }
            else
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tUsing default roughness factor\n";
                }
            }
        }

        {
            const auto metallic_it = gltf_material.values.find( "metallicFactor" );
            if( metallic_it != gltf_material.values.end() )
            {
                mtl.metallic = static_cast<float>( metallic_it->second.Factor() );
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tMetallic:  " << mtl.metallic <<  "\n";
                }
            }
            else
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tUsing default metallic factor\n";
                }
            }
        }

        {
            const auto metallic_roughness_it = gltf_material.values.find( "metallicRoughnessTexture" );
            if( metallic_roughness_it != gltf_material.values.end() )
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tFound metallic roughness tex: " << metallic_roughness_it->second.TextureIndex() << "\n";
                }
                mtl.metallic_roughness_tex = scene.getSampler( metallic_roughness_it->second.TextureIndex() );
            }
            else
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tNo metallic roughness tex\n";
                }
            }
        }

        {
            const auto normal_it = gltf_material.additionalValues.find( "normalTexture" );
            if( normal_it != gltf_material.additionalValues.end() )
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tFound normal color tex: " << normal_it->second.TextureIndex() << "\n";
                }
                mtl.normal_tex = scene.getSampler( normal_it->second.TextureIndex() );
            }
            else
            {
                if constexpr (debug_gltf == true) {
                    std::cerr << "\tNo normal tex\n";
                }
            }
        }

        scene.addMaterial( mtl );
    }

    //
    // Process nodes
    //
    std::vector<int32_t> root_nodes( model.nodes.size(), 1 );
    for (auto& gltf_node : model.nodes) {
        for (int32_t child : gltf_node.children) {
            root_nodes[child] = 0;
        }
    }

    for (size_t i = 0; i < root_nodes.size(); ++i) {
        if (!root_nodes[i]) { continue; }
        auto& gltf_node = model.nodes[i];
        processGLTFNode (scene, model, gltf_node, root_transform, glTFdir);
    }
}


void MulticamScene::addBuffer( const uint64_t buf_size, const void* data )
{
    CUdeviceptr buffer = 0;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &buffer ), buf_size ) );

    CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( buffer ),
                    data,
                    buf_size,
                    cudaMemcpyHostToDevice
                    ) );
    m_buffers.push_back( buffer );
}


void MulticamScene::addImage(
    const int32_t width,
    const int32_t height,
    const int32_t bits_per_component,
    const int32_t num_components,
    const void* data
    )
{
    // Allocate CUDA array in device memory
    int32_t               pitch;
    cudaChannelFormatDesc channel_desc;
    if( bits_per_component == 8 )
    {
        pitch        = width*num_components*sizeof(uint8_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else if( bits_per_component == 16 )
    {
        pitch        = width*num_components*sizeof(uint16_t);
        channel_desc = cudaCreateChannelDesc<uchar4>();
    }
    else
    {
        throw Exception( "Unsupported bits/component in glTF image" );
    }


    cudaArray_t   cuda_array = nullptr;
    CUDA_CHECK( cudaMallocArray(
                &cuda_array,
                &channel_desc,
                width,
                height
                ) );

    CUDA_CHECK( cudaMemcpy2DToArray(cuda_array,  // destination
                                    0,           // X offset
                                    0,           // Y offset
                                    data,        // source
                                    pitch,       // source pitch
                                    pitch,       // width
                                    height,      // height
                                    cudaMemcpyHostToDevice) );
    m_images.push_back( cuda_array );
}


void MulticamScene::addSampler(
    cudaTextureAddressMode address_s,
    cudaTextureAddressMode address_t,
    cudaTextureFilterMode  filter,
    const int32_t          image_idx
    )
{
    cudaResourceDesc res_desc = {};
    res_desc.resType          = cudaResourceTypeArray;
    res_desc.res.array.array  = getImage( image_idx );

    cudaTextureDesc tex_desc     = {};
    tex_desc.addressMode[0]      = address_s == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
    address_s == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
    cudaAddressModeWrap;
    tex_desc.addressMode[1]      = address_t == GL_CLAMP_TO_EDGE   ? cudaAddressModeClamp  :
    address_t == GL_MIRRORED_REPEAT ? cudaAddressModeMirror :
    cudaAddressModeWrap;
    tex_desc.filterMode          = filter    == GL_NEAREST         ? cudaFilterModePoint   :
    cudaFilterModeLinear;
    tex_desc.readMode            = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords    = 1;
    tex_desc.maxAnisotropy       = 1;
    tex_desc.maxMipmapLevelClamp = 99;
    tex_desc.minMipmapLevelClamp = 0;
    tex_desc.mipmapFilterMode    = cudaFilterModePoint;
    tex_desc.borderColor[0]      = 1.0f;
    tex_desc.sRGB                = 0; // TODO: glTF assumes sRGB for base_color -- handle in shader

    // Create texture object
    cudaTextureObject_t cuda_tex = 0;
    CUDA_CHECK( cudaCreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr ) );
    m_samplers.push_back( cuda_tex );
}


CUdeviceptr MulticamScene::getBuffer( int32_t buffer_index ) const
{
    return m_buffers[ buffer_index ];
}


cudaArray_t MulticamScene::getImage( int32_t image_index ) const
{
    return m_images[ image_index ];
}


cudaTextureObject_t MulticamScene::getSampler( int32_t sampler_index ) const
{
    return m_samplers[ sampler_index ];
}


void MulticamScene::finalize()
{
    GenericCamera* c = getCamera();

    createContext();
    buildMeshAccels();
    buildInstanceAccel();
    //createPTXModule(m_compound_ptx_module, "ommatidialShader.cu");
    //createPTXModule(m_ptx_module, "shaders.cu");
    createPTXModule();
    createProgramGroups();
    createPipeline();
    createCompoundPipeline();
    // Now handle the creation of the standard SBT table:
    createSBTmissAndHit(m_sbt);

    // Now handle the creation of the compound SBT table
    CompoundEye::InitiateCompoundRecord(m_compound_sbt, m_compound_raygen_group, c->getRecordPtr());// Initialize the compound record
    createSBTmissAndHit(m_compound_sbt); // Create the miss and hit bindings

    // Make sure the raygenRecord is pointed at and valid memory:
    c->forcePackAndCopyRecord(m_raygen_prog_group);
    m_sbt.raygenRecord = c->getRecordPtr();

    m_scene_aabb.invalidate();
    for( const auto& mesh: m_meshes ) {
        m_scene_aabb.include( mesh->world_aabb );
    }

    checkIfCurrentCameraIsCompound();
    //if( !m_cameras.empty() )
    //    m_cameras.front().setLookat( m_scene_aabb.center() );
}


MulticamScene::~MulticamScene()
{
    cleanup();
}

void MulticamScene::cleanup()
{
    //TODO: destroy the camera vector properly
    CompoundEye::FreeCompoundRecord();
}

//------------------------------------------------------------------------------
//
//  CAMERA FUNCTIONS
//
//------------------------------------------------------------------------------

int MulticamScene::addCamera(GenericCamera* cameraPtr)
{
    int i = m_cameras.size();
    m_cameras[i] = cameraPtr;
    checkIfCurrentCameraIsCompound();
    return i;
}
GenericCamera* MulticamScene::getCamera()
{
    if(!m_cameras.empty())
    {
        return m_cameras[currentCamera];
    }

    if constexpr (debug_cameras == true) {
        std::cerr << "Initializing default camera" << std::endl;
    }
    //cam.setFovY( 45.0f );
    //cam.setLookat( m_scene_aabb.center() );
    //cam.setEye   ( m_scene_aabb.center() + make_float3( 0.0f, 0.0f, 1.5f*m_scene_aabb.maxExtent() ) );

    PerspectiveCamera* cam = new PerspectiveCamera("Default Camera");
    this->addCamera(cam);
    return getCamera();

}
void MulticamScene::setCurrentCamera(const int index)
{
    const int s = int(getCameraCount());
    currentCamera = (index%s + s)%s;
    checkIfCurrentCameraIsCompound();
}
const size_t MulticamScene::getCameraCount() const
{
    return m_cameras.size();
}
void MulticamScene::nextCamera()
{
    setCurrentCamera(currentCamera+1);
}
void MulticamScene::previousCamera()
{
    setCurrentCamera(currentCamera-1);
}

//------------------------------------------------------------------------------
//
//  COMPOUND EYE FUNCTIONS
//
//------------------------------------------------------------------------------
uint32_t MulticamScene::addCompoundCamera(int cam_idx, CompoundEye* cameraPtr, std::vector<Ommatidium>& ommVec)
{
    m_compoundEyes[cam_idx] = cameraPtr;
    m_ommVecs[cam_idx] = ommVec;
    if constexpr (debug_cameras == true) {
        std::cout << "Inserted ommVec of size " << m_ommVecs[cam_idx].size()
                  << " into m_ommVecs[" << cam_idx << "].\n";
    }
    return (m_compoundEyes.size()-1);
}
void MulticamScene::checkIfCurrentCameraIsCompound()
{
    GenericCamera* cam = getCamera();
    bool out = false;
    for (auto ce : m_compoundEyes) { out |= cam == ce.second; }
    m_selectedCameraIsCompound = out;
}

//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------

void MulticamScene::createContext()
{
    // Initialize CUDA
    CUDA_CHECK( cudaFree( nullptr ) );

    CUcontext          cuCtx = nullptr;  // zero means take the current context
    OPTIX_CHECK( optixInit() );
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction       = &context_log_cb;
    options.logCallbackLevel          = 4;
    OPTIX_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_context ) );
}

namespace {
    template <typename T = char>
    class CuBuffer
    {
    public:
        CuBuffer( size_t count = 0 ) { alloc( count ); }
        ~CuBuffer() { free(); }
        void alloc( size_t count )
        {
            free();
            m_allocCount = m_count = count;
            if( m_count )
            {
                CUDA_CHECK( cudaMalloc( &m_ptr, m_allocCount * sizeof( T ) ) );
            }
        }
        void allocIfRequired( size_t count )
        {
            if( count <= m_count )
            {
                m_count = count;
                return;
            }
            alloc( count );
        }
        CUdeviceptr get() const { return reinterpret_cast<CUdeviceptr>( m_ptr ); }
        CUdeviceptr get( size_t index ) const { return reinterpret_cast<CUdeviceptr>( m_ptr + index ); }
        void        free()
        {
            m_count      = 0;
            m_allocCount = 0;
            CUDA_CHECK( cudaFree( m_ptr ) );
            m_ptr = nullptr;
        }
        CUdeviceptr release()
        {
            CUdeviceptr current = reinterpret_cast<CUdeviceptr>( m_ptr );
            m_count             = 0;
            m_allocCount        = 0;
            m_ptr               = nullptr;
            return current;
        }
        void upload( const T* data )
        {
            CUDA_CHECK( cudaMemcpy( m_ptr, data, m_count * sizeof( T ), cudaMemcpyHostToDevice ) );
        }

        void download( T* data ) const
        {
            CUDA_CHECK( cudaMemcpy( data, m_ptr, m_count * sizeof( T ), cudaMemcpyDeviceToHost ) );
        }
        void downloadSub( size_t count, size_t offset, T* data ) const
        {
            assert( count + offset < m_allocCount );
            CUDA_CHECK( cudaMemcpy( data, m_ptr + offset, count * sizeof( T ), cudaMemcpyDeviceToHost ) );
        }
        size_t count() const { return m_count; }
        size_t reservedCount() const { return m_allocCount; }
        size_t byteSize() const { return m_allocCount * sizeof( T ); }

    private:
        size_t m_count      = 0;
        size_t m_allocCount = 0;
        T*     m_ptr        = nullptr;
    };
}  // namespace

void MulticamScene::buildMeshAccels( uint32_t triangle_input_flags )
{
    // Problem:
    // The memory requirements of a compacted GAS are unknown prior to building the GAS.
    // Hence, compaction of a GAS requires to build the GAS first and allocating memory for the compacted GAS afterwards.
    // This causes a device-host synchronization point, potentially harming performance.
    // This is most likely the case for small GASes where the actual building and compaction of the GAS is very fast.
    // A naive algorithm processes one GAS at a time with the following steps:
    // 1. compute memory sizes for the build process (temporary buffer size and build buffer size)
    // 2. allocate temporary and build buffer
    // 3. build the GAS (with temporary and build buffer) and compute the compacted size
    // If compacted size is smaller than build buffer size (i.e., compaction is worth it):
    // 4. allocate compacted buffer (final output buffer)
    // 5. compact GAS from build buffer into compact buffer
    //
    // Idea of the algorithm:
    // Batch process the building and compaction of multiple GASes to avoid host-device synchronization.
    // Ideally, the number of synchronization points would be linear with the number of batches rather than the number of GASes.
    // The main constraints for selecting batches of GASes are:
    // a) the peak memory consumption when batch processing GASes, and
    // b) the amount of memory for the output buffer(s), containing the compacted GASes. This is also part of a), but is also important after the build process.
    // For the latter we try to keep it as minimal as possible, i.e., the total memory requirements for the output should equal the sum of the compacted sizes of the GASes.
    // Hence, it should be avoided to waste memory by allocating buffers that are bigger than what is required for a compacted GAS.
    //
    // The peak memory consumption effectively defines the efficiency of the algorithm.
    // If memory was unlimited, compaction isn't needed at all.
    // A lower bound for the peak memory consumption during the build is the output of the process, the size of the compacted GASes.
    // Peak memory consumption effectively defines the memory pool available during the batch building and compaction of GASes.
    //
    // The algorithm estimates the size of the compacted GASes by a give compaction ratio as well as the computed build size of each GAS.
    // The compaction ratio is defined as: size of compacted GAS / size of build output of GAS.
    // The validity of this estimate therefore depends on the assumed compaction ratio.
    // The current algorithm assumes a fixed compaction ratio.
    // Other strategies could be:
    // - update the compaction ration on the fly by do statistics on the already processed GASes to have a better guess for the remaining batches
    // - multiple compaction rations by type of GAS (e.g., motion vs static), since the type of GAS impacts the compaction ratio
    // Further, compaction may be skipped for GASes that do not benefit from compaction (compaction ratio of 1.0).
    //
    // Before selecting GASes for a batch, all GASes are sorted by size (their build size).
    // Big GASes are handled before smaller GASes as this will increase the likelihood of the peak memory consumption staying close to the minimal memory consumption.
    // This also increase the benefit of batching since small GASes that benefit most from avoiding synchronizations are built "together".
    // The minimum batch size is one GAS to ensure forward process.
    //
    // Goal:
    // Estimate the required output size (the minimal peak memory consumption) and work within these bounds.
    // Batch process GASes as long as they are expected to fit into the memory bounds (non strict).
    //
    // Assumptions:
    // The inputs to each GAS are already in device memory and are needed afterwards.
    // Otherwise this could be factored into the peak memory consumption.
    // E.g., by uploading the input data to the device only just before building the GAS and releasing it right afterwards.
    //
    // Further, the peak memory consumption of the application / system is influenced by many factors unknown to this algorithm.
    // E.g., if it is known that a big pool of memory is needed after GAS building anyways (e.g., textures that need to be present on the device),
    // peak memory consumption will be higher eventually and the GAS build process could already make use of a bigger memory pool.
    //
    // TODOs:
    // - compaction ratio estimation / updating
    // - handling of non-compactable GASes
    // - integration of GAS input data upload / freeing
    // - add optional hard limits / check for hard memory limits (shrink batch size / abort, ...)
    //////////////////////////////////////////////////////////////////////////

    // Magic constants:

    // see explanation above
    constexpr double initialCompactionRatio = 0.5;

    // It is assumed that trace is called later when the GASes are still in memory.
    // We know that the memory consumption at that time will at least be the compacted GASes + some CUDA stack space.
    // Add a "random" 250MB that we can use here, roughly matching CUDA stack space requirements.
    constexpr size_t additionalAvailableMemory = 250 * 1024 * 1024;

    //////////////////////////////////////////////////////////////////////////

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    struct GASInfo {
        std::vector<OptixBuildInput> buildInputs;
        OptixAccelBufferSizes gas_buffer_sizes;
        std::shared_ptr<MeshGroup> mesh;
    };
    std::multimap<size_t, GASInfo> gases;
    size_t totalTempOutputSize = 0;

    for(size_t i=0; i<m_meshes.size(); ++i)
    {
        auto& mesh = m_meshes[i];

        const size_t num_subMeshes =  mesh->indices.size();
        std::vector<OptixBuildInput> buildInputs(num_subMeshes);

        assert(mesh->positions.size() == num_subMeshes &&
               mesh->normals.size()   == num_subMeshes &&
               mesh->texcoords.size() == num_subMeshes);// &&
        //mesh->vertex_colours.size() == num_subMeshes);

        for(size_t i = 0; i < num_subMeshes; ++i)
        {
            OptixBuildInput& triangle_input                      = buildInputs[i];
            memset(&triangle_input, 0, sizeof(OptixBuildInput));
            triangle_input.type                                  = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangle_input.triangleArray.vertexFormat            = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangle_input.triangleArray.vertexStrideInBytes     = mesh->positions[i].byte_stride ? mesh->positions[i].byte_stride : sizeof(float3),
            triangle_input.triangleArray.numVertices             = mesh->positions[i].count;
            triangle_input.triangleArray.vertexBuffers           = &(mesh->positions[i].data);
            triangle_input.triangleArray.indexFormat             = mesh->indices[i].elmt_byte_size == 2 ? OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3 : OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triangle_input.triangleArray.indexStrideInBytes      = mesh->indices[i].byte_stride ? mesh->indices[i].byte_stride : mesh->indices[i].elmt_byte_size*3;
            triangle_input.triangleArray.numIndexTriplets        = mesh->indices[i].count / 3;
            triangle_input.triangleArray.indexBuffer             = mesh->indices[i].data;
            triangle_input.triangleArray.flags                   = &triangle_input_flags;
            triangle_input.triangleArray.numSbtRecords           = 1;
        }

        OptixAccelBufferSizes gas_buffer_sizes;
        OPTIX_CHECK( optixAccelComputeMemoryUsage( m_context, &accel_options, buildInputs.data(),
                                                   static_cast<unsigned int>( num_subMeshes ), &gas_buffer_sizes ) );

        totalTempOutputSize += gas_buffer_sizes.outputSizeInBytes;
        GASInfo g = {std::move( buildInputs ), gas_buffer_sizes, mesh};
        gases.emplace( gas_buffer_sizes.outputSizeInBytes, g );
    }

    size_t totalTempOutputProcessedSize = 0;
    size_t usedCompactedOutputSize = 0;
    double compactionRatio = initialCompactionRatio;

    CuBuffer<char> d_temp;
    CuBuffer<char> d_temp_output;
    CuBuffer<size_t> d_temp_compactedSizes;

    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    while( !gases.empty() )
    {
        // The estimated total output size that we end up with when using compaction.
        // It defines the minimum peak memory consumption, but is unknown before actually building all GASes.
        // Working only within these memory constraints results in an actual peak memory consumption that is very close to the minimal peak memory consumption.
        size_t remainingEstimatedTotalOutputSize =
        ( size_t )( ( totalTempOutputSize - totalTempOutputProcessedSize ) * compactionRatio );
        size_t availableMemPoolSize = remainingEstimatedTotalOutputSize + additionalAvailableMemory;
        // We need to fit the following things into availableMemPoolSize:
        // - temporary buffer for building a GAS (only during build, can be cleared before compaction)
        // - build output buffer of a GAS
        // - size (actual number) of a compacted GAS as output of a build
        // - compacted GAS

        size_t batchNGASes                    = 0;
        size_t batchBuildOutputRequirement    = 0;
        size_t batchBuildMaxTempRequirement   = 0;
        size_t batchBuildCompactedRequirement = 0;
        for( auto it = gases.rbegin(); it != gases.rend(); it++ )
        {
            batchBuildOutputRequirement += it->second.gas_buffer_sizes.outputSizeInBytes;
            batchBuildCompactedRequirement += ( size_t )( it->second.gas_buffer_sizes.outputSizeInBytes * compactionRatio );
            // roughly account for the storage of the compacted size, although that goes into a separate buffer
            batchBuildOutputRequirement += 8ull;
            // make sure that all further output pointers are 256 byte aligned
            batchBuildOutputRequirement = roundUp<size_t>( batchBuildOutputRequirement, 256ull );
            // temp buffer is shared for all builds in the batch
            batchBuildMaxTempRequirement = std::max( batchBuildMaxTempRequirement, it->second.gas_buffer_sizes.tempSizeInBytes );
            batchNGASes++;
            if( ( batchBuildOutputRequirement + batchBuildMaxTempRequirement + batchBuildCompactedRequirement ) > availableMemPoolSize )
                break;
        }

        // d_temp may still be available from a previous batch, but is freed later if it is "too big"
        d_temp.allocIfRequired( batchBuildMaxTempRequirement );

        // trash existing buffer if it is more than 10% bigger than what we need
        // if it is roughly the same, we keep it
        if( d_temp_output.byteSize() > batchBuildOutputRequirement * 1.1 )
            d_temp_output.free();
        d_temp_output.allocIfRequired( batchBuildOutputRequirement );

        // this buffer is assumed to be very small
        // trash d_temp_compactedSizes if it is at least 20MB in size and at least double the size than required for the next run
        if( d_temp_compactedSizes.reservedCount() > batchNGASes * 2 && d_temp_compactedSizes.byteSize() > 20 * 1024 * 1024 )
            d_temp_compactedSizes.free();
        d_temp_compactedSizes.allocIfRequired( batchNGASes );

        // sum of build output size of GASes, excluding alignment
        // size_t batchTempOutputSize = 0; // unused
        // sum of size of compacted GASes
        size_t batchCompactedSize = 0;

        auto it = gases.rbegin();
        for( size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i )
        {
            emitProperty.result = d_temp_compactedSizes.get( i );
            GASInfo& info = it->second;

            OPTIX_CHECK( optixAccelBuild( m_context, 0,   // CUDA stream
                                          &accel_options,
                                          info.buildInputs.data(),
                                          static_cast<unsigned int>( info.buildInputs.size() ),
                                          d_temp.get(),
                                          d_temp.byteSize(),
                                          d_temp_output.get( tempOutputAlignmentOffset ),
                                          info.gas_buffer_sizes.outputSizeInBytes,
                                          &info.mesh->gas_handle,
                                          &emitProperty,  // emitted property list
                                          1               // num emitted properties
                             ) );

            tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
            it++;
        }

        // trash d_temp if it is at least 20MB in size
        if( d_temp.byteSize() > 20 * 1024 * 1024 )
            d_temp.free();

        // download all compacted sizes to allocate final output buffers for these GASes
        std::vector<size_t> h_compactedSizes( batchNGASes );
        d_temp_compactedSizes.download( h_compactedSizes.data() );

        //////////////////////////////////////////////////////////////////////////
        // TODO:
        // Now we know the actual memory requirement of the compacted GASes.
        // Based on that we could shrink the batch if the compaction ratio is bad and we need to strictly fit into the/any available memory pool.
        bool canCompact = false;
        it = gases.rbegin();
        for( size_t i = 0; i < batchNGASes; ++i )
        {
            GASInfo& info = it->second;
            if( info.gas_buffer_sizes.outputSizeInBytes > h_compactedSizes[i] )
            {
                canCompact = true;
                break;
            }
            it++;
        }

        if( canCompact )
        {
            //////////////////////////////////////////////////////////////////////////
            // "batch allocate" the compacted buffers
            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                batchCompactedSize += h_compactedSizes[i];
                CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &info.mesh->d_gas_output ), h_compactedSizes[i] ) );
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;
                it++;
            }

            it = gases.rbegin();
            for( size_t i = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                OPTIX_CHECK( optixAccelCompact( m_context, 0, info.mesh->gas_handle, info.mesh->d_gas_output,
                                                h_compactedSizes[i], &info.mesh->gas_handle ) );
                it++;
            }
        }
        else
        {
            it = gases.rbegin();
            for( size_t i = 0, tempOutputAlignmentOffset = 0; i < batchNGASes; ++i )
            {
                GASInfo& info = it->second;
                info.mesh->d_gas_output = d_temp_output.get( tempOutputAlignmentOffset );
                batchCompactedSize += h_compactedSizes[i];
                totalTempOutputProcessedSize += info.gas_buffer_sizes.outputSizeInBytes;

                tempOutputAlignmentOffset += roundUp<size_t>( info.gas_buffer_sizes.outputSizeInBytes, 256ull );
                it++;
            }
            d_temp_output.release();
        }

        usedCompactedOutputSize += batchCompactedSize;

        gases.erase( it.base(), gases.end() );
    }
}


///TODO
struct Instance
{
    float transform[12];
};

void MulticamScene::buildInstanceAccel( int rayTypeCount )
{
    const size_t num_instances = m_meshes.size();

    std::vector<OptixInstance> optix_instances( num_instances );

    unsigned int sbt_offset = 0;
    for( size_t i = 0; i < m_meshes.size(); ++i )
    {
        auto  mesh = m_meshes[i];
        auto& optix_instance = optix_instances[i];
        memset( &optix_instance, 0, sizeof( OptixInstance ) );

        optix_instance.flags             = OPTIX_INSTANCE_FLAG_NONE;
        optix_instance.instanceId        = static_cast<unsigned int>( i );
        optix_instance.sbtOffset         = sbt_offset;
        optix_instance.visibilityMask    = 1;
        optix_instance.traversableHandle = mesh->gas_handle;
        memcpy( optix_instance.transform, mesh->transform.getData(), sizeof( float ) * 12 );

        sbt_offset += static_cast<unsigned int>( mesh->indices.size() ) * rayTypeCount;  // one sbt record per GAS build input per RAY_TYPE
    }

    const size_t instances_size_in_bytes = sizeof( OptixInstance ) * num_instances;
    CUdeviceptr  d_instances;
    CUDA_CHECK( cudaMalloc( reinterpret_cast<void**>( &d_instances ), instances_size_in_bytes ) );
    CUDA_CHECK( cudaMemcpy(
                    reinterpret_cast<void*>( d_instances ),
                    optix_instances.data(),
                    instances_size_in_bytes,
                    cudaMemcpyHostToDevice
                    ) );

    OptixBuildInput instance_input = {};
    instance_input.type                       = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instance_input.instanceArray.instances    = d_instances;
    instance_input.instanceArray.numInstances = static_cast<unsigned int>( num_instances );

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags                  = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation                   = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes ias_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage(
                     m_context,
                     &accel_options,
                     &instance_input,
                     1, // num build inputs
                     &ias_buffer_sizes
                     ) );

    CUdeviceptr d_temp_buffer;
    CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &d_temp_buffer ),
                    ias_buffer_sizes.tempSizeInBytes
                    ) );
    CUDA_CHECK( cudaMalloc(
                    reinterpret_cast<void**>( &m_d_ias_output_buffer ),
                    ias_buffer_sizes.outputSizeInBytes
                    ) );

    OPTIX_CHECK( optixAccelBuild(
                     m_context,
                     nullptr,                  // CUDA stream
                     &accel_options,
                     &instance_input,
                     1,                  // num build inputs
                     d_temp_buffer,
                     ias_buffer_sizes.tempSizeInBytes,
                     m_d_ias_output_buffer,
                     ias_buffer_sizes.outputSizeInBytes,
                     &m_ias_handle,
                     nullptr,            // emitted property list
                     0                   // num emitted properties
                     ) );

    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_temp_buffer ) ) );
    CUDA_CHECK( cudaFree( reinterpret_cast<void*>( d_instances   ) ) );
}

void MulticamScene::createPTXModule()
{

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel   = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;

    m_pipeline_compile_options = {};
    m_pipeline_compile_options.usesMotionBlur            = false;
    m_pipeline_compile_options.traversableGraphFlags     = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    m_pipeline_compile_options.numPayloadValues          = globalParameters::NUM_PAYLOAD_VALUES;
    m_pipeline_compile_options.numAttributeValues        = 2; // todo
    m_pipeline_compile_options.exceptionFlags            = OPTIX_EXCEPTION_FLAG_NONE; // should be optix_exception_flag_stack_overflow;
    m_pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    const std::string ptx = getPtxString( "EyeRenderer3", "shaders.cu" );

    m_ptx_module  = {};
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixModuleCreate(
                         m_context,
                         &module_compile_options,
                         &m_pipeline_compile_options,
                         ptx.c_str(),
                         ptx.size(),
                         log,
                         &sizeof_log,
                         &m_ptx_module
                         ) );
}


void MulticamScene::createProgramGroups()
{
    char log[2048];
    size_t sizeof_log = sizeof( log );

    {
        // Create the ommatidial raygen group
        OptixProgramGroupDesc compound_prog_group_desc    = {};
        compound_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        compound_prog_group_desc.raygen.module            = m_ptx_module;
        compound_prog_group_desc.raygen.entryFunctionName = "__raygen__ommatidium";

        if constexpr (debug_pipeline) {
            std::cout << "MulticamScene::createProgramGroups(): optixProgramGroupCreate for "
                      << compound_prog_group_desc.raygen.entryFunctionName << std::endl;
        }
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                             m_context,
                             &compound_prog_group_desc,
                             1,                             // num program groups
                             &program_group_options,
                             log,
                             &sizeof_log,
                             &m_compound_raygen_group
                             )
            );
    }

    {
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = m_ptx_module;
        raygen_prog_group_desc.raygen.entryFunctionName = GenericCamera::DEFAULT_RAYGEN_PROGRAM;

        if constexpr (debug_pipeline) {
            std::cout << "MulticamScene::createProgramGroups(): optixProgramGroupCreate for "
                      << raygen_prog_group_desc.raygen.entryFunctionName << std::endl;
        }
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                             m_context,
                             &raygen_prog_group_desc,
                             1,                             // num program groups
                             &program_group_options,
                             log,
                             &sizeof_log,
                             &m_raygen_prog_group
                             )
            );
    }


    //
    // Miss
    //
    {
        OptixProgramGroupDesc miss_prog_group_desc = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = m_ptx_module;
        miss_prog_group_desc.miss.entryFunctionName = m_backgroundShader.c_str();
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                             m_context,
                             &miss_prog_group_desc,
                             1,                             // num program groups
                             &program_group_options,
                             log,
                             &sizeof_log,
                             &m_radiance_miss_group
                             )
            );

        memset( &miss_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = nullptr;  // NULL miss program for occlusion rays
        miss_prog_group_desc.miss.entryFunctionName = nullptr;
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                             m_context,
                             &miss_prog_group_desc,
                             1,                             // num program groups
                             &program_group_options,
                             log,
                             &sizeof_log,
                             &m_occlusion_miss_group
                             )
            );
    }

    //
    // Hit group
    //
    {
        OptixProgramGroupDesc hit_prog_group_desc = {};
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
        sizeof_log = sizeof( log );
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                             m_context,
                             &hit_prog_group_desc,
                             1,                             // num program groups
                             &program_group_options,
                             log,
                             &sizeof_log,
                             &m_radiance_hit_group
                             )
            );

        memset( &hit_prog_group_desc, 0, sizeof( OptixProgramGroupDesc ) );
        hit_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hit_prog_group_desc.hitgroup.moduleCH            = m_ptx_module;
        hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
        sizeof_log = sizeof( log );
        OPTIX_CHECK( optixProgramGroupCreate(
                         m_context,
                         &hit_prog_group_desc,
                         1,                             // num program groups
                         &program_group_options,
                         log,
                         &sizeof_log,
                         &m_occlusion_hit_group
                         )
            );
    }
}


void MulticamScene::createPipeline()
{
    if constexpr (debug_pipeline == true) {
        std::cout << "MulticamScene::createPipeline(): Generating Projection pipeline..." << std::endl;
    }
    OptixProgramGroup program_groups[] =
    {
        m_raygen_prog_group,
        m_radiance_miss_group,
        m_occlusion_miss_group,
        m_radiance_hit_group,
        m_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 2;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                         m_context,
                         &m_pipeline_compile_options,
                         &pipeline_link_options,
                         program_groups,
                         sizeof( program_groups ) / sizeof( program_groups[0] ),
                         log,
                         &sizeof_log,
                         &m_pipeline
                         ) );
}

void MulticamScene::createCompoundPipeline()
{
    if constexpr (debug_pipeline == true) {
        std::cout << "MulticamScene::createCompoundPipeline(): Generating Compound pipeline..." << std::endl;
    }
    OptixProgramGroup program_groups[] =
    {
        m_compound_raygen_group,
        m_radiance_miss_group,
        m_occlusion_miss_group,
        m_radiance_hit_group,
        m_occlusion_hit_group
    };

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth          = 2;

    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK_LOG( optixPipelineCreate(
                         m_context,
                         &m_pipeline_compile_options,
                         &pipeline_link_options,
                         program_groups,
                         sizeof( program_groups ) / sizeof( program_groups[0] ),
                         log,
                         &sizeof_log,
                         &m_compound_pipeline
                         ) );
}

void MulticamScene::reconfigureSBTforCurrentCamera(bool force)
{
    GenericCamera* c = getCamera();
    char log[2048];
    size_t sizeof_log = sizeof( log );

    // Here, we regenerate the raygen pipeline if the camera has changed types:
    if(getCameraIndex() != lastPipelinedCamera || lastPipelinedCamera == std::numeric_limits<size_t>::max() || force)
    {
        lastPipelinedCamera = currentCamera;// update the pointer
        raygen_prog_group_desc.raygen.entryFunctionName = c->getEntryFunctionName();
        if constexpr (debug_pipeline == true) {
            std::cout<< "ALERT: Regenerating pipeline with raygen entry function '"<<c->getEntryFunctionName()<<"'."<<std::endl;
        }
        // THIS is where the projection shader is set up
        optixProgramGroupDestroy(m_raygen_prog_group);
        OPTIX_CHECK_LOG( optixProgramGroupCreate(
                             m_context,
                             &raygen_prog_group_desc,
                             1,                             // num program groups
                             &program_group_options,
                             log,
                             &sizeof_log,
                             &m_raygen_prog_group
                             )
            );

        c->forcePackAndCopyRecord(m_raygen_prog_group);
        m_sbt.raygenRecord = c->getRecordPtr();

        // Redirect the static compound eye pipeline record toward the current camera's record since the currently selected camera has changed
        // TODO: The raygen group reference might not be needed here. Find out.
        CompoundEye::RedirectCompoundDataPointer(m_compound_raygen_group, c->getRecordPtr());

        optixPipelineDestroy(m_pipeline);
        createPipeline();
        //createCompoundPipeline(); // but only if something?

    } else {
        // Just sync the camera's on-device memory (but only on a host-side change):
        c->packAndCopyRecordIfChanged(m_raygen_prog_group);
    }
}

void MulticamScene::createSBTmissAndHit(OptixShaderBindingTable& sbt)
{
    // Per-camera raygen Records are handled by each camera

    // Miss Record
    {
        const size_t miss_record_size = sizeof( EmptyRecord );
        CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &sbt.missRecordBase ),
                        miss_record_size*globalParameters::RAY_TYPE_COUNT
                        ) );

        EmptyRecord ms_sbt[ globalParameters::RAY_TYPE_COUNT ];
        OPTIX_CHECK( optixSbtRecordPackHeader( m_radiance_miss_group,  &ms_sbt[0] ) );
        OPTIX_CHECK( optixSbtRecordPackHeader( m_occlusion_miss_group, &ms_sbt[1] ) );

        CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( sbt.missRecordBase ),
                        ms_sbt,
                        miss_record_size*globalParameters::RAY_TYPE_COUNT,
                        cudaMemcpyHostToDevice
                        ) );
        sbt.missRecordStrideInBytes = static_cast<uint32_t>( miss_record_size );
        sbt.missRecordCount     = globalParameters::RAY_TYPE_COUNT;
    }

    // Hitgroup Records
    {
        std::vector<HitGroupRecord> hitgroup_records;
        for( const auto& mesh : m_meshes )
        {
            for( size_t i = 0; i < mesh->material_idx.size(); ++i )
            {
                HitGroupRecord rec = {};
                OPTIX_CHECK( optixSbtRecordPackHeader( m_radiance_hit_group, &rec ) );
                rec.data.geometry_data.type                    = GeometryData::TRIANGLE_MESH;
                rec.data.geometry_data.triangle_mesh.positions = mesh->positions[i];
                rec.data.geometry_data.triangle_mesh.normals   = mesh->normals[i];
                rec.data.geometry_data.triangle_mesh.texcoords = mesh->texcoords[i];
                rec.data.geometry_data.triangle_mesh.indices   = mesh->indices[i];

                rec.data.geometry_data.triangle_mesh.dev_color_type = mesh->host_color_types[i];
                rec.data.geometry_data.triangle_mesh.color_container = mesh->host_color_container; // specifies vec3 or vec4 colors
                rec.data.geometry_data.triangle_mesh.dev_colors_f3 = mesh->host_colors_f3[i];
                rec.data.geometry_data.triangle_mesh.dev_colors_f4 = mesh->host_colors_f4[i];
                rec.data.geometry_data.triangle_mesh.dev_colors_us4 = mesh->host_colors_us4[i];
                rec.data.geometry_data.triangle_mesh.dev_colors_uc4 = mesh->host_colors_uc4[i];

                const int32_t mat_idx  = mesh->material_idx[i];
                if( mat_idx >= 0 )
                    rec.data.material_data.pbr = m_materials[ mat_idx ];
                else
                    rec.data.material_data.pbr = MaterialData::Pbr();
                hitgroup_records.push_back( rec );

                OPTIX_CHECK( optixSbtRecordPackHeader( m_occlusion_hit_group, &rec ) );
                hitgroup_records.push_back( rec );
            }
        }

        const size_t hitgroup_record_size = sizeof( HitGroupRecord );
        CUDA_CHECK( cudaMalloc(
                        reinterpret_cast<void**>( &sbt.hitgroupRecordBase ),
                        hitgroup_record_size*hitgroup_records.size()
                        ) );

        CUDA_CHECK( cudaMemcpy(
                        reinterpret_cast<void*>( sbt.hitgroupRecordBase ),
                        hitgroup_records.data(),
                        hitgroup_record_size*hitgroup_records.size(),
                        cudaMemcpyHostToDevice
                        ) );

        sbt.hitgroupRecordStrideInBytes = static_cast<unsigned int>( hitgroup_record_size );
        sbt.hitgroupRecordCount         = static_cast<unsigned int>( hitgroup_records.size() );
    }
}

//// Additional scene features
bool MulticamScene::isInsideHitGeometry(float3 worldPos, std::string name, bool debug)
{
    if(debug) { std::cout << "Atempting hitscan against \"" << name << "\"\n"; }

    // Search through each of the m_hitboxMeshes until we find the hitbox mesh we care about
    sutil::hitscan::TriangleMesh* hitboxMesh = nullptr;

    for(unsigned int i = 0u; i<m_hitboxMeshes.size(); i++)
    {
        if(m_hitboxMeshes[i].name == name)
        {
            hitboxMesh = &m_hitboxMeshes[i];
            break;
        }
    }

    if(hitboxMesh == nullptr)
    {
        std::cerr << "WARNING: No hitbox with the given name \"" << name << "\" is present in the scene." << std::endl;
        return false;
    }

    if(debug) { std::cout << "\tMesh acquired.\n"; }

    // First quickly check if the point is within the mesh's AABB:
    //if(!hitboxMesh->worldAabb.contains(worldPos))
    //  return false; // If it doesn't contain it, then it certainly ain't gunna be in the model.

    if(debug) { std::cout << "\tPoint within mesh bounds.\n"; }

    // Perform a within-mesh hitscan against the selected mesh
    return sutil::hitscan::isPointWithinMesh(*hitboxMesh, worldPos);
}

// TODO: Each of these below (and the one above) should share a "get geometry by name" method.
float3 MulticamScene::getGeometryMaxBounds(std::string name)
{
    for(unsigned int i = 0u; i<m_hitboxMeshes.size(); i++)
        if(m_hitboxMeshes[i].name == name)
            return m_hitboxMeshes[i].worldAabb.m_max;

    for(unsigned int i = 0u; i<m_meshes.size(); i++)
        if(m_meshes[i]->name == name)
            return m_meshes[i]->world_aabb.m_max;

    return make_float3(0.0f);
}
float3 MulticamScene::getGeometryMinBounds(std::string name)
{
    for(unsigned int i = 0u; i<m_hitboxMeshes.size(); i++)
        if(m_hitboxMeshes[i].name == name)
            return m_hitboxMeshes[i].worldAabb.m_min;

    for(unsigned int i = 0u; i<m_meshes.size(); i++)
        if(m_meshes[i]->name == name)
            return m_meshes[i]->world_aabb.m_min;

    return make_float3(0.0f);
}
