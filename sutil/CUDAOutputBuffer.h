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

// NB: Seb has hacked this to be GL free (So no GL_INTEROP any more). Used only by
// newGuiEyeRenderer/gui.cpp.

#pragma once

#include <sutil/Exception.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "sutil.h"

namespace sutil
{

    enum class CUDAOutputBufferType
    {
        CUDA_DEVICE = 0, // not preferred, typically slower than ZERO_COPY
        GL_INTEROP  = 1, // NOT SUPPORTED single device only, preferred for single device NOT SUPPORTED
        ZERO_COPY   = 2, // general case, preferred for multi-gpu if not fully nvlink connected
        CUDA_P2P    = 3  // NOT SUPPORTED fully connected only, preferred for fully nvlink connected NOT SUPPORTED
    };


    template <typename PIXEL_FORMAT>
    class CUDAOutputBuffer
    {
    public:
        CUDAOutputBuffer( CUDAOutputBufferType type, int32_t width, int32_t height );
        ~CUDAOutputBuffer();

        void setDevice( int32_t device_idx ) { m_device_idx = device_idx; }
        void setStream( CUstream stream    ) { m_stream     = stream;     }

        void resize( int32_t width, int32_t height );

        // Allocate or update device pointer as necessary for CUDA access
        PIXEL_FORMAT* map();
        void unmap();

        int32_t width()  { return m_width;  }
        int32_t height() { return m_height; }
        int32_t area() { return m_width * m_height; }

        // Get output buffer
        PIXEL_FORMAT*  getHostPointer();

        CUDAOutputBufferType getType () const { return this->m_type; }

        void makeCurrent() { CUDA_CHECK( cudaSetDevice( m_device_idx ) ); }

    private:
        CUDAOutputBufferType       m_type;

        int32_t                    m_width             = 0u;
        int32_t                    m_height            = 0u;

        cudaGraphicsResource*      m_cuda_gfx_resource = nullptr;

    public:
        PIXEL_FORMAT*              m_host_zcopy_pixels = nullptr;
        PIXEL_FORMAT*              m_device_pixels     = nullptr;
        std::vector<PIXEL_FORMAT>  m_host_pixels;

    private:
        CUstream                   m_stream            = 0u;
        int32_t                    m_device_idx        = 0;
    };


    template <typename PIXEL_FORMAT>
    CUDAOutputBuffer<PIXEL_FORMAT>::CUDAOutputBuffer (CUDAOutputBufferType type, int32_t width, int32_t height)
        : m_type(type)
    {
        if (type == CUDAOutputBufferType::GL_INTEROP) {
            throw sutil::Exception ("CUDAOutputBuffer: GL_INTEROP not supported");
        }
        resize (width, height);
    }

    template <typename PIXEL_FORMAT>
    CUDAOutputBuffer<PIXEL_FORMAT>::~CUDAOutputBuffer()
    {
        try {
            makeCurrent();
            if (m_type == CUDAOutputBufferType::CUDA_DEVICE) {
                CUDA_CHECK( cudaFree( reinterpret_cast<void*>( m_device_pixels ) ) );
            } else if (m_type == CUDAOutputBufferType::CUDA_P2P ) {
                throw sutil::Exception ("CUDAOutputBuffer: CUDA_P2P not supported");
            } else if (m_type == CUDAOutputBufferType::ZERO_COPY ) {
                CUDA_CHECK( cudaFreeHost( reinterpret_cast<void*>( m_host_zcopy_pixels ) ) );
            } else if (m_type == CUDAOutputBufferType::GL_INTEROP ) {
                throw sutil::Exception ("CUDAOutputBuffer: GL_INTEROP not supported");
            }
        } catch(std::exception& e ) {
            std::cerr << "CUDAOutputBuffer destructor caught exception: " << e.what() << std::endl;
        }
    }

    template <typename PIXEL_FORMAT>
    void CUDAOutputBuffer<PIXEL_FORMAT>::resize( int32_t width, int32_t height )
    {
        if (m_width == width && m_height == height) { return; }

        m_width = width;
        m_height = height;

        makeCurrent();

        if (m_type == CUDAOutputBufferType::CUDA_DEVICE) {
            CUDA_CHECK (cudaFree (reinterpret_cast<void*>(m_device_pixels)));
            CUDA_CHECK (cudaMalloc (reinterpret_cast<void**>(&m_device_pixels), m_width * m_height * sizeof(PIXEL_FORMAT)));
        }

        if (m_type == CUDAOutputBufferType::GL_INTEROP || m_type == CUDAOutputBufferType::CUDA_P2P) {
            throw sutil::Exception ("CUDAOutputBuffer: GL_INTEROP/CUDA_P2P not supported");
        }

        if (m_type == CUDAOutputBufferType::ZERO_COPY) {
            CUDA_CHECK (cudaFreeHost (reinterpret_cast<void*>(m_host_zcopy_pixels)));
            CUDA_CHECK (cudaHostAlloc (reinterpret_cast<void**>(&m_host_zcopy_pixels),
                                       m_width * m_height * sizeof(PIXEL_FORMAT),
                                       cudaHostAllocPortable | cudaHostAllocMapped));
            CUDA_CHECK (cudaHostGetDevicePointer (reinterpret_cast<void**>(&m_device_pixels),
                                                  reinterpret_cast<void*>(m_host_zcopy_pixels), 0));
        }

        if (!m_host_pixels.empty()) { m_host_pixels.resize (m_width * m_height); }
    }

    template <typename PIXEL_FORMAT>
    PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::map()
    {
        if (m_type == CUDAOutputBufferType::GL_INTEROP ) {
            throw sutil::Exception ("CUDAOutputBuffer: GL_INTEROP not supported");
        } // else no need to do anything
        return m_device_pixels;
    }

    template <typename PIXEL_FORMAT>
    void CUDAOutputBuffer<PIXEL_FORMAT>::unmap()
    {
        makeCurrent();
        if( m_type == CUDAOutputBufferType::GL_INTEROP  ) {
            throw sutil::Exception ("CUDAOutputBuffer: GL_INTEROP not supported");
        } else { // ZERO_COPY or CUDA_DEVICE:
            CUDA_CHECK (cudaStreamSynchronize (m_stream));
        }
    }

    template <typename PIXEL_FORMAT>
    PIXEL_FORMAT* CUDAOutputBuffer<PIXEL_FORMAT>::getHostPointer()
    {
        if (m_type == CUDAOutputBufferType::CUDA_DEVICE) {

            m_host_pixels.resize (m_width * m_height);

            makeCurrent();
            CUDA_CHECK (cudaMemcpy (static_cast<void*>(m_host_pixels.data()), map(),
                                    m_width * m_height * sizeof(PIXEL_FORMAT), cudaMemcpyDeviceToHost));
            unmap();

            return m_host_pixels.data();

        } else if (m_type == CUDAOutputBufferType::ZERO_COPY) {
            return m_host_zcopy_pixels;

        } else {
            throw sutil::Exception ("CUDAOutputBuffer: GL_INTEROP/CUDA_P2P not supported");
        }
    }

} // end namespace sutil
