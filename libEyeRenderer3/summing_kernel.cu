/*
 * An implementation of a summing kernel to add up the samples (which are already
 * divided by n_ommatidial_samples) to get the average.
 *
 * Adapted by Seb James from an example of the summing algorithm at
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 *
 * \author Seb James
 * \date November, 2024
 */
#include "summing_kernel.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

// Number of threads in a warp for a contemporary NVidia GPU
static constexpr int warpthreads = 32;
// Ideal number of threads per block for a contemporary NVidia GPU (10 series to 40 series and similar)
static constexpr int threadsperblock = 512;
// Mask value for __shfl_down_sync
static constexpr unsigned int all_in_warp = 0xffffffff;

// Check a CUDA error code, exiting on failure
inline void gpuAssert (cudaError_t code, const char* file, int line)
{
   if (code != cudaSuccess) {
       std::cerr << "CUDA_SUM: " << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
       exit (code);
   }
}

// In each warp reduce three values per thread
__inline__ __device__ float3 warpReduceSum (float valR, float valG, float valB)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        valR += __shfl_down_sync(all_in_warp, valR, offset);
        valG += __shfl_down_sync(all_in_warp, valG, offset);
        valB += __shfl_down_sync(all_in_warp, valB, offset);
    }
    return make_float3 (valR, valG, valB);
}

// Run by the 32 threads of a warp
__inline__ __device__ float3 blockReduceSum (float3 val)
{
    static __shared__ float3 shared[warpthreads];       // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val.x, val.y, val.z); // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }      // Write reduced value to shared memory
    __syncthreads();                           // Wait for all partial reductions to complete

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float3(0.0f, 0.0f, 0.0f);

    // Final reduction within first warp
    if (wid == 0) { val = warpReduceSum (val.x, val.y, val.z); }

    return val;
}

/*
 * Perform a sum of n_arrays, each containing n_elements float3-sized data points,
 * arranged in the device memory pointed to by in.
 *
 * The elements of one array are dispersed across the memory. Element 0 of array 0 is at
 * memory location 0; element 1 of array 0 is at memory location (0 + n_elements) and so
 * on. This results in a need to translate the thread index (called tidx in the code)
 * which has an x component that indexes elements and a y component that indexes arrays
 * into a memory index (midx).
 *
 * The result is written to out, which should be a region of device memory of size
 * gridDim.x * n_arrays * sizeof(float3) (gridDim.x is usually 1).
 */
__global__ void reduceit_arrays (float3* in, float3* out, int n_arrays, int n_elements)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    // The y axis of our threads/threadblocks indexes which of the n_arrays this sum relates to
    int omm_id = blockIdx.y * blockDim.y  + threadIdx.y;
    // This gives a memory offset to get to the right part of the input memory
    int thread_offset = omm_id * n_elements;
    // (if you needed to do array index checking, the data size is n_arrays * n_elements)

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements && omm_id < n_arrays;
         i += blockDim.x * gridDim.x) { // jumps the whole threadblock as threadblock may not span all data
        int tidx = thread_offset + i;
        // Convert to x/y (int tx = (tidx % n_elements); int ty = (tidx / n_elements);) then to real memory index
        int midx = (tidx % n_elements) * n_arrays + (tidx / n_elements);
        sum.x += in[midx].x;
        sum.y += in[midx].y;
        sum.z += in[midx].z;
    }

    sum = blockReduceSum (sum);
    __syncthreads();

    // This gets the correct output location in out. gridDim.x is "n_sums"
    if (threadIdx.x == 0 && omm_id < n_arrays) {
        // Number of sums is the number of 1D threadblocks that span n_elements. This is gridDim.x.
        size_t out_idx = omm_id * gridDim.x + blockIdx.x;
        if (out_idx < n_arrays * gridDim.x) { out[out_idx] = sum; }
    }
}

/*
 * Perform sums of n_pixels arrays, where each pixel array contains n_samples
 * float3-sized data points, arranged in the device memory pointed to by d_omm.
 *
 * The samples of one pixel (aka ommatidium) are dispersed across the memory. Sample 0
 * of pixel 0 is at memory location 0; sample 1 of pixel 0 is at memory location (0 +
 * n_samples) and so on. The kernel (and/or the choice of thread blocks) needs to
 * account for this.
 *
 * Note that I did't figure out that the samples of one pixel/ommatidium are dispersed
 * across the memory until later in development. It may be more efficient to re-design
 * the choice of threadblocks (using the x dimension to refer to array and y to refer to
 * sample) and avoid the need for tidx->midx translation in reduceit_arrays. However, as
 * this function is not a performance bottleneck, I'll leave it for now.
 *
 * The result is written to d_sums, which should be a region of device memory of size
 * n_arrays * sizeof(float3).
 *
 * This host function determines threadblock size/grid size and then launches the
 * reduceit_arrays kernel to compute the sum of each array.
 */
__host__ void summing_kernel (float3* d_omm, float3* d_sums, int n_pixels, int n_samples)
{
    if (d_omm == nullptr || d_sums == nullptr) { return; }
    int tx = std::min ((n_samples / warpthreads) * warpthreads + (n_samples % warpthreads) ? warpthreads : 0, threadsperblock);
    dim3 blockdim(tx, 1);
    dim3 griddim(1, n_pixels / blockdim.y + (n_pixels % blockdim.y ? 1 : 0));
    reduceit_arrays<<<griddim, blockdim>>>(d_omm, d_sums, n_pixels, n_samples);
    gpuAssert (cudaDeviceSynchronize(), __FILE__, __LINE__);
}
