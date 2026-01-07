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

#pragma once

#include <stdint.h>
#include <cuda.h>
#include <optix.h>
#include <sutil/Preprocessor.h>

namespace cuda
{
    // A BufferView of things of type T, where T may be a native type like int, float, etc or a type
    // like float3, float2 etc. If T is float3, then the 'elements' are float if T is float then the
    // elements are float also.
    template <typename T>
    struct BufferView
    {
        // A pointer to the data
        CUdeviceptr data CONST_STATIC_INIT(0);
        // count of objects of size elmt_byte_size stored in this BufferView
        uint32_t count CONST_STATIC_INIT(0);
        // A byte stride from object to object, which may be greater than elmt_byte_size
        uint16_t byte_stride CONST_STATIC_INIT(0);
        // Size of the objects in bytes, might be smaller than T, into which type each object will be placed
        uint16_t elmt_byte_size CONST_STATIC_INIT(0);

        // Is the BufferView aligned for Optix functions? That requires data to be 16 byte aligned (OPTIX_SBT_RECORD_ALIGNMENT)
        SUTIL_HOSTDEVICE bool isAligned() const { return (data % OPTIX_SBT_RECORD_ALIGNMENT == 0); }
        // Is this BufferView's data pointer valid?
        SUTIL_HOSTDEVICE bool isValid() const { return static_cast<bool>(data); }
        // bool operator returns true if the BufferView has valid, aligned data
        SUTIL_HOSTDEVICE operator bool() const { return isValid() && isAligned(); }
        // array operator for data access
        SUTIL_HOSTDEVICE const T& operator[] (uint32_t idx) const { return *reinterpret_cast<T*>(data + idx * actual_stride()); }
        // The stride is either the byte_stride if defined (larger than sizeof(T)) or sizeof(T)
        SUTIL_HOSTDEVICE size_t actual_stride() const { return (byte_stride ? byte_stride : elmt_byte_size); }
        // Return the buffers data consumption in bytes
        SUTIL_HOSTDEVICE uint32_t size_bytes() const { return count * actual_stride(); }
    };

} // namespace cuda

#ifndef __CUDA_ARCH__ // compile cuda::CopiedBufferView only for the host
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

namespace cuda
{
    // small helper class to copy a BufferView to local CPU ram, most likely for debugging
    template <typename T>
    struct CopiedBufferView
    {
        cuda::BufferView<T> bv; // The CUDA buffer view of data

        std::vector<T> bv_data = {};

        CopiedBufferView (const cuda::BufferView<T>& _bv)
        {
            this->bv = _bv;

            if (sizeof(T) < bv.elmt_byte_size) {
                std::cerr << "CopiedBufferView: Can't fit elements into type T" << std::endl;
                return;
            }

            this->bv_data.resize (bv.count, T{0});
            size_t nbytes = bv.count * bv.actual_stride();

            if (sizeof(T) == bv.elmt_byte_size) {
                // Simplest case, can direct copy into bv_data
                void* dst = this->bv_data.data();
                auto mc_rtn = cudaMemcpy (dst, reinterpret_cast<void*>(bv.data), nbytes, cudaMemcpyDeviceToHost);
                if (mc_rtn) {
                    std::cerr << "CopiedBufferView: cudaMemcpy returned error: " << mc_rtn << std::endl;
                }
            } else {
                // More complex case
                // Copy to a temporary buffer of size rawbytes, then extract these into type T
                std::vector<uint8_t> rawbytes (nbytes, 0u);
                void* dst = rawbytes.data();
                auto mc_rtn = cudaMemcpy (dst, reinterpret_cast<void*>(bv.data), nbytes, cudaMemcpyDeviceToHost);
                if (mc_rtn) {
                    std::cerr << "CopiedBufferView: cudaMemcpy returned error: " << mc_rtn << std::endl;
                } else {
                    uint32_t j = 0u;
                    for (uint32_t i = 0u; i < bv.count; ++i) {
                        if constexpr (sizeof(T) == 4) {
                            uint32_t tmp = 0u;
                            for (uint16_t ii = 0u; ii < bv.elmt_byte_size; ii++) {
                                //std::cout << "tmp |= rawbytes[" << j << "](" << static_cast<uint32_t>(rawbytes[j])
                                //          << ") << " << (8 * ii) << " or tmp |= " <<  (static_cast<uint32_t>(rawbytes[j]) << (8 * ii)) << "...\n";
                                tmp |= (static_cast<uint32_t>(rawbytes[j++]) << (8 * ii));
                            }
                            this->bv_data[i] = static_cast<T>(tmp);
                            //std::cout << "...so bv_data["<<i<<"] = " <<  bv_data[i] << std::endl;
                        } else if constexpr (sizeof(T) == 8) {
                            uint64_t tmp = 0u;
                            for (uint16_t ii = 0u; ii < bv.elmt_byte_size; ii++) {
                                tmp |= static_cast<uint64_t>(rawbytes[j++]) << (8 * ii);
                            }
                            this->bv_data[i] = static_cast<T>(tmp);
                        } else {
                            std::cerr << "CopiedBufferView: sizeof(T) error: " << std::endl;
                        }
                    }
                }
            }
        }

        const T operator[] (uint32_t idx) const
        {
            if (idx < bv_data.size()) {
                return bv_data[idx];
            } else { return T{}; }
        }
    };
}
#endif
