#include <cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <algorithm>

#define TILE_SIZE 32

using namespace cooperative_groups;

__device__ int reduce_sum(thread_group g, int* temp, int val)
{
    int idx = threadIdx.x / g.size();
    atomicAdd(&temp[idx], val);
    g.sync(); // wait for all threads to store
    if (g.thread_rank() == 0) {
        val = temp[idx];
    }
    g.sync(); // wait for all threads to load

    return val; // note: only thread 0 will return full sum
}

__device__ int thread_sum(int* input, int n)
{
    int sum = 0;

    sum = input[blockIdx.x * blockDim.x + threadIdx.x];

    return sum;
}

__global__ void sum_kernel_block(int* sum, int* input, int n)
{
    int my_sum = thread_sum(input, n);
    extern __shared__ int temp[];
    thread_group tile = tiled_partition(this_thread_block(), TILE_SIZE);
    int tile_sum = reduce_sum(tile, temp, my_sum);
    if (tile.thread_rank() == 0) atomicAdd(sum, tile_sum);
}

cudaError_t runCooperativeGroups() {
    printf("starting %s\n", __func__);

    cudaError_t cudaStatus;
    int n = 1 << 10;
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int sharedBytes = threadsPerBlock / TILE_SIZE * sizeof(int);

    int *sum, * data;
    int *h_sum;
    h_sum = (int*)malloc(sizeof(int));
    *h_sum = 1;
    cudaMalloc((void**)&sum, sizeof(int));
    cudaMallocManaged(&data, n * sizeof(int));
    std::fill_n(data, n, 1); // initialize data
    cudaMemset(sum, 0, sizeof(int));

    sum_kernel_block<<<blocksPerGrid, threadsPerBlock, sharedBytes>>>(sum, data, n);

    cudaDeviceSynchronize();
    cudaMemcpy(h_sum, sum, sizeof(int), cudaMemcpyDeviceToHost);

    cudaStatus = cudaGetLastError();
    return cudaStatus;
}