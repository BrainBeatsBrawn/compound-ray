/*
 * Compoundray's OptiX ray casting functions
 */

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
// This codebase has been edited by Blayze Millward for the CompoundRay
// system and does not reflect the original codebase provided by NVIDIA
// corporation.
//
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "RayComputeTypes.h"

#include <stdint.h>
#include <stdio.h>

#include "cameras/GenericCameraDataTypes.h"
#include "cameras/CompoundEyeDataTypes.h"

// cuRand
#include <curand_kernel.h>

__constant__ float FWHM_SD_RATIO = 2.35482004503094938202313865291f;

extern "C"
{
__constant__ cray::LaunchParams params;
}


//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
//
//------------------------------------------------------------------------------

__device__ float3 schlick (const float3 spec_color, const float V_dot_H)
{
    return spec_color + (make_float3 (1.0f) - spec_color) * powf (1.0f - V_dot_H, 5.0f);
}

__device__ float vis (const float N_dot_L, const float N_dot_V, const float alpha)
{
    const float alpha_sq = alpha * alpha;
    const float ggx0 = N_dot_L * sqrtf (N_dot_V * N_dot_V * (1.0f - alpha_sq) + alpha_sq);
    const float ggx1 = N_dot_V * sqrtf (N_dot_L * N_dot_L * (1.0f - alpha_sq) + alpha_sq);
    return 2.0f * N_dot_L * N_dot_V / (ggx0 + ggx1);
}

__device__ float ggxNormal (const float N_dot_H, const float alpha)
{
    const float alpha_sq = alpha * alpha;
    const float N_dot_H_sq = N_dot_H * N_dot_H;
    const float x = N_dot_H_sq * (alpha_sq - 1.0f) + 1.0f;
    return alpha_sq / (M_PIf * x * x);
}

//------------------------------------------------------------------------------
//
// Tracing functions
//
//------------------------------------------------------------------------------

// This wraps optixTrace to perform a rayCast (ray time is set to 0)
static __forceinline__ __device__ void traceRadiance (OptixTraversableHandle handle,
                                                      float3                 ray_origin,
                                                      float3                 ray_direction,
                                                      float                  tmin,
                                                      float                  tmax,
                                                      cray::PayloadRadiance* payload)
{
    uint32_t u0 = 0u, u1 = 0u, u2 = 0u, u3 = 0u;
    optixTrace (handle,
                ray_origin, ray_direction,
                tmin,
                tmax,
                0.0f,                    // rayTime
                OptixVisibilityMask (1),
                OPTIX_RAY_FLAG_NONE,
                cray::RAY_TYPE_RADIANCE, // SBT offset
                cray::RAY_TYPE_COUNT,    // SBT stride
                cray::RAY_TYPE_RADIANCE, // missSBTIndex
                u0, u1, u2, u3);

     payload->result.x = __int_as_float (u0);
     payload->result.y = __int_as_float (u1);
     payload->result.z = __int_as_float (u2);
     payload->depth = u3;
}

// Wraps optixTrace to find the first occlusion. used in __closesthit__radiance()
static __forceinline__ __device__ bool traceOcclusion (OptixTraversableHandle handle,
                                                       float3                 ray_origin,
                                                       float3                 ray_direction,
                                                       float                  tmin,
                                                       float                  tmax)
{
    uint32_t occluded = 0u;
    optixTrace (handle,
                ray_origin,
                ray_direction,
                tmin,
                tmax,
                0.0f,                    // rayTime
                OptixVisibilityMask (1),
                OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
                cray::RAY_TYPE_OCCLUSION,      // SBT offset
                cray::RAY_TYPE_COUNT,          // SBT stride
                cray::RAY_TYPE_OCCLUSION,      // missSBTIndex
                occluded);
    return occluded;
}

// Used in miss programs
__forceinline__ __device__ void setPayloadResult (float3 p)
{
    optixSetPayload_0 (__float_as_int (p.x));
    optixSetPayload_1 (__float_as_int (p.y));
    optixSetPayload_2 (__float_as_int (p.z));
}

// Used in __closesthit__occlusion()
__forceinline__ __device__ void setPayloadOcclusion (bool occluded)
{
    optixSetPayload_0 (static_cast<uint32_t>(occluded));
}

// Used to make colours for the __raygen__ programs that write to params.framebuffer
__forceinline__ __device__ uchar4 make_color (const float3& c)
{
    const float gamma = 2.2f;
    return make_uchar4 (static_cast<uint8_t>(powf (clamp (c.x, 0.0f, 1.0f), 1.0f / gamma) * 255.0f),
                        static_cast<uint8_t>(powf (clamp (c.y, 0.0f, 1.0f), 1.0f / gamma) * 255.0f),
                        static_cast<uint8_t>(powf (clamp (c.z, 0.0f, 1.0f), 1.0f / gamma) * 255.0f), 255u);
}

//------------------------------------------------------------------------------
//
//  Ommatidial Ray Generation Programs.
//
//  For Seb, this () is the only one that matters. Rather than writing out to params.frame_buffer,
//  it writes its result into posedData.specializedData.d_compoundBuffer (which can then be averaged
//  down by another kernel - a CUDA one)
//
//------------------------------------------------------------------------------

__device__ inline float3 rotatePoint (const float3 point, const float angle, const float3 axis)
{
    return (cos(angle) * point + sin(angle) * cross(axis, point) + (1.0f - cos(angle)) * dot(axis, point) * axis);
}

__device__ float3 generateOffsetRay (const float ommatidialAxisAngle, const float splayAngle, const float3 ommatidialAxis)
{
    // Rotate the ommatidial axis about a perpendicular vector by splay angle
    float3 perpAxis = cross (make_float3 (0.0f, 1.0f, 0.0f), ommatidialAxis);
    // Check that the perpAxis isn't zero (because ommatidialAxis was pointing directly up) (could probably be done with a memcmp for speed)
    perpAxis = (perpAxis.x + perpAxis.y + perpAxis.z == 0.0f) ? make_float3 (0.0f, 0.0f, 1.0f) : normalize (perpAxis);
    // Rotate by the splay angle
    const float3 splayedAxis = rotatePoint (ommatidialAxis, splayAngle, perpAxis);
    // Rotate the new axis around the original ommatidial axis by the ommatidialAxisAngle
    return rotatePoint (splayedAxis, ommatidialAxisAngle, ommatidialAxis);
}

/*
 * The ommatidium raycaster. This is selected in MulticamScene.cpp where an "ommatidial raygen
 * group" (a program group) is created. The entryFunctionName is set to "__raygen__ommatidium" which
 * results in this OptiX program being used. This function is part of a "compoundPipeline" in
 * MulticamScene which is launched by MulticamScene::launchFrame IF the scene has compoundeye
 * cameras and the current camera is of type cray::CompoundEye.
 */
extern "C" __global__ void __raygen__ommatidium()
{
    const uint3 launch_idx = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();
    const uint32_t ommatidialIndex = launch_idx.x;
    const int id = launch_dims.x * launch_idx.y + launch_idx.x;

    // This gets the compound record, which points to the current camera's record.
    const cray::RecordPointer* recordPointer = (cray::RecordPointer*)optixGetSbtDataPointer();
    // This contains the actual posed eye data
    const cray::RaygenPosedContainer<cray::CompoundEyeData> posedData = ((cray::RaygenRecord<cray::RaygenPosedContainer<cray::CompoundEyeData>>*)(recordPointer->d_record))->data;

    cray::Ommatidium* allOmmatidia = (cray::Ommatidium*)(posedData.specializedData.d_ommatidialArray);
    cray::Ommatidium ommatidium = *(allOmmatidia + ommatidialIndex); // This ommatidium

    // Get the relative direction of the ommatidial axis
    const float3 relativeOmmatidialAxis = ommatidium.relativeDirection;
    const float3 relativeOmmatidialPosition = ommatidium.relativePosition;

    // A local copy of the cuRand state (to be) stored in shared memory
    curandState localState;
    // A reference to the original cuRand state stored in shared memory
    curandState& sharedState = ((curandState*)(posedData.specializedData.d_randomStates))[id];
    if (!posedData.specializedData.randomsConfigured) {
        curand_init (42, id, 0, &localState); // Initialize the state if it needs to be
    } else {
        localState = sharedState; // Pull down the random state of this ommatidium
    }

    // Calculate the s.d. to scale a standard normal random value up to so that it matches the acceptance angle
    const float standardDeviation = ommatidium.acceptanceAngleRadians / FWHM_SD_RATIO;
    // Angle away from the ommatidial axis
    float splayAngle = curand_normal (&localState) * standardDeviation;
    // Angle around the ommatidial axis (note that it only needs to rotate through 180 degrees because splayAngle can be negative)
    float ommatidialAxisAngle = curand_uniform (&localState) * M_PIf;
    // Copy the RNG state back into the buffer for use next time
    sharedState = localState;
    // Generate a pair of angles away from the ommatidial axis
    const float3 relativeDir = generateOffsetRay (ommatidialAxisAngle, splayAngle, relativeOmmatidialAxis);
    // Move the start of the ray into the eye along the ommatidial axis by focalPointOffset
    const float3 relativePos = relativeOmmatidialPosition - normalize (relativeOmmatidialAxis) * ommatidium.focalPointOffset;
    // Transform ray information into world-space
    const float3 ray_origin = posedData.position + posedData.localSpace.xAxis * relativePos.x + posedData.localSpace.yAxis * relativePos.y + posedData.localSpace.zAxis * relativePos.z;
    const float3 ray_direction = posedData.localSpace.xAxis * relativeDir.x + posedData.localSpace.yAxis * relativeDir.y + posedData.localSpace.zAxis * relativeDir.z;

    cray::PayloadRadiance payload;
    payload.result = make_float3 (0.0f);
    payload.importance = 1.0f;
    payload.depth = 0.0f;

    traceRadiance (params.handle,
                   ray_origin,
                   ray_direction,
                   ommatidium.focalPointOffset, // tmin, the surface of the top of the ommatidial lens
                   1e16f, // tmax
                   &payload);

    // Add results to this eye's compound buffer.
    // This mixes in the feedback from each sample ray with respect to its position in the rendering volume.
    // For instance, if each ommatidium is to make 20 samples then each launch of this shader is one sample and only
    // contributes 0.05/1 to the final colour in the compound buffer.
    // Scale it down as these will be summed in the projection shader
    ((float3*)posedData.specializedData.d_compoundBuffer)[id] = payload.result * (1.0f / posedData.specializedData.samplesPerOmmatidium);
}

//------------------------------------------------------------------------------
//
//  Miss programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__default_background()
{
    const float3 dir = normalize (optixGetWorldRayDirection());
    setPayloadResult (make_float3 ((atan2 (dir.z, dir.x) + M_PIf) / (M_PIf * 2.0f), (asin (dir.y) + M_PIf / 2.0f) / M_PIf, 0.0f));
    const float border = 0.01;
    if (abs(dir.x) < border || abs(dir.y) < border || abs(dir.z) < border) {
        setPayloadResult (make_float3 (0.0f));
    }
}

extern "C" __global__ void __miss__white() { setPayloadResult (make_float3 (1.0f)); }
extern "C" __global__ void __miss__black() { setPayloadResult (make_float3 (0.0f)); }

extern "C" __global__ void __miss__simple_sky()
{
    const float3 dir = normalize (optixGetWorldRayDirection());
    const float mix = min (max (0.0f, (asin (dir.y) * 2.0f) / M_PIf), 1.0f);
    const float3 upper = make_float3 (1.0f, 31.0f, 117.0f) / 255.0f;
    const float3 lower = make_float3 (143.0f, 179.0f, 203.0f) / 255.0f * 0.8f;
    setPayloadResult (lower * (1.0f - mix) + upper * mix);
}

// Simple sky unless your ommatidium direction is (0,0,0) (which appears to guarantee that the ray
// direction is either (0,0,0) or contains a NaN, then set black
extern "C" __global__ void __miss__simple_sky_black_down()
{
    const float3 p = optixGetWorldRayDirection();

    if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
        setPayloadResult (make_float3 (0.0f));
    } else if (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f) {
        setPayloadResult (make_float3 (0.0f));
    } else {
        const float3 dir = normalize (p);
        const float mix = min (max (0.0f, (asin (dir.y) * 2.0f) / M_PIf), 1.0f);
        const float3 upper = make_float3 (1.0f, 31.0f, 117.0f) / 255.0f;
        const float3 lower = make_float3 (143.0f, 179.0f, 203.0f) / 255.0f * 0.8f;
        setPayloadResult (lower * (1.0f - mix) + upper * mix);
    }
}

// Show a single colour for a miss, or another colour for a special ray direction of 000/contains NaN
__device__ void coloured_miss_coloured_down (const float3& clr_miss, const float3& clr_down)
{
    const float3 p = optixGetWorldRayDirection();
    if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z)) {
        setPayloadResult (clr_down);
    } else if (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f) {
        setPayloadResult (clr_down);
    } else {
        setPayloadResult (clr_miss);
    }
}

extern "C" __global__ void __miss__white_but_black_down()
{
    coloured_miss_coloured_down (make_float3 (1.0f), make_float3 (0.0f));
}

extern "C" __global__ void __miss__lightgrey_but_black_down()
{
    coloured_miss_coloured_down (make_float3 (0.95f), make_float3 (0.0f));
}

//------------------------------------------------------------------------------
//
//  Older Hit Programs (still in use)
//
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__occlusion() { setPayloadOcclusion (true); }

extern "C" __global__ void __closesthit__radiance()
{
    float3 result = make_float3 (0.0f);
    const cray::HitGroupData* hit_group_data = reinterpret_cast<cray::HitGroupData*>(optixGetSbtDataPointer());
    const LocalGeometry geom = getLocalGeometry (hit_group_data->geometry_data);

    //
    // Retrieve material data
    //
    float3 base_color = make_float3 (0.0f, 0.0f, 1.0f); // initialize to fully blue

    if (geom.UC) {
        // If we must 'use color' then we just convert (gamma-correct) the color specified in geom.C
        // Either with a call to this function: base_color = linearize (make_float3 (geom.C.x, geom.C.y, geom.C.z));
        // or, just inline right here (without creating an intermediate float3):
        base_color.x = powf (geom.C.x, 2.2f);
        base_color.y = powf (geom.C.y, 2.2f);
        base_color.z = powf (geom.C.z, 2.2f);

    } else if (hit_group_data != nullptr) {
        // No 'use color' commandment, but we have hit_group_data
        base_color = make_float3 (1.0f, 0.0f, 0.0f); // re-initialize to fully red

        cudaTextureObject_t mbc = hit_group_data->material_data.pbr.base_color_tex;
        if (mbc) {
            float4 my_tex = tex2D<float4> (mbc, geom.UV.x, geom.UV.y);
            float3 my_tex3 = make_float3 (my_tex);
            base_color = my_tex3;
        } else {
            // This obtains the colour we want when there is no base_color_tex
            base_color = make_float3 (hit_group_data->material_data.pbr.base_color);
        }

    } else {
        // No geom.C and no hit_group_data. Set pure green base_color to assist debug
        base_color = make_float3 (0.0f, 1.0f, 0.0f);
        setPayloadResult (base_color);
        return;
    }
    if (!params.lighting) {
        setPayloadResult (base_color);
        return;
    }

    result = base_color;

    float metallic  = hit_group_data->material_data.pbr.metallic;
    float roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex = make_float4 (1.0f);
    if (hit_group_data->material_data.pbr.metallic_roughness_tex) {
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = tex2D<float4> (hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y);
    }
    roughness *= mr_tex.y;
    metallic  *= mr_tex.z;
    //
    // Convert to material params
    //
    const float F0 = 0.04f;
    const float3 diff_color = base_color * (1.0f - F0) * (1.0f - metallic);
    const float3 spec_color = lerp (make_float3 (F0), base_color, metallic);
    const float alpha = roughness * roughness;
    //
    // compute direct lighting
    //
    float3 N = geom.N;
    if (hit_group_data->material_data.pbr.normal_tex) {
        const float4 NN = 2.0f * tex2D<float4> (hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y) - make_float4 (1.0f);
        N = normalize (NN.x * normalize (geom.dpdu) + NN.y * normalize (geom.dpdv) + NN.z * geom.N);
    }

    // There are four lights in compound ray apparently, hardcoded in. traceOcclusion for any to get a crash...
    // Light sampling: with many lights, shade lights_per_hit of them chosen uniformly at random and scale by
    // count / lights_per_hit, so the expected result equals shading every light.
    const std::uint32_t n_lights = params.lights.count;
    const std::uint32_t n_shaded = (params.lights_per_hit == 0u || params.lights_per_hit >= n_lights) ? n_lights : params.lights_per_hit;
    const float light_weight = n_shaded > 0u ? static_cast<float> (n_lights) / static_cast<float> (n_shaded) : 0.0f;
    const uint3 li = optixGetLaunchIndex();
    unsigned int seed = tea<4> (li.x + optixGetLaunchDimensions().x * li.y, params.frame);
    for (std::uint32_t s = 0; s < n_shaded; ++s) {
        const std::uint32_t i = (n_shaded == n_lights) ? s : min (static_cast<std::uint32_t> (rnd (seed) * n_lights), n_lights - 1u);
        Light::Point light = params.lights[i];
        // TODO: optimize
        const float L_dist = length (light.position - geom.P);
        const float3 L = (light.position - geom.P) / L_dist;
        const float3 V = -normalize (optixGetWorldRayDirection());
        const float3 H = normalize (L + V);
        const float N_dot_L = dot (N, L);
        const float N_dot_V = dot (N, V);
        const float N_dot_H = dot (N, H);
        const float V_dot_H = dot (V, H);
        if (N_dot_L > 0.0f && N_dot_V > 0.0f) {
            const float tmin = 0.001f;          // TODO
            const float tmax = L_dist - 0.001f; // TODO
            const bool occluded = traceOcclusion (params.handle, geom.P, L, tmin, tmax);
            if (!occluded) {
                const float3 F = schlick (spec_color, V_dot_H);
                const float G_vis = vis (N_dot_L, N_dot_V, alpha);
                const float D = ggxNormal (N_dot_H, alpha);
                const float3 diff = (1.0f - F) * diff_color / M_PIf;
                const float3 spec = F * G_vis * D;
                result += light_weight * light.color * light.intensity * N_dot_L * (diff + spec);
            }
        }
    }

    setPayloadResult (result);
}
