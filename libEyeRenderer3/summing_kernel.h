#pragma once

#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

/*
 * A CUDA kernel to sum up an array of arrays using the warp-shuffle method
 */
__host__ void summing_kernel (float3* d_omm, float3* d_sums, int n_pixels, int n_samples_per_pixel);
