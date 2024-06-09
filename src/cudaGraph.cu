#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include "device_launch_parameters.h"

// Kernel function to increment each element in the array
__global__ void increment(int* a, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        a[idx] += 1;
    }
}

cudaError_t runGraphs() {
    printf("starting %s\n", __func__);

    cudaError_t cudaStatus;
    int n = 1000;
    int size = n * sizeof(int);
    int* d_a;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaStream_t stream;

    // Allocate memory on the device (GPU)
    cudaMalloc((void**)&d_a, size);
    cudaMemset(d_a, 0, size);

    // Define the number of threads per block and the number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Create a CUDA stream
    cudaStreamCreate(&stream);

    // Capture the graph
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    increment<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, n);
    cudaStreamEndCapture(stream, &graph);

    // Instantiate the graph
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);

    // Launch the graph multiple times
    int launchCount = 5;
    for (int i = 0; i < launchCount; ++i) {
        cudaGraphLaunch(instance, stream);
    }

    // Wait for the graph executions to complete
    cudaStreamSynchronize(stream);

    // Copy result back to the host to verify
    int* h_a = (int*)malloc(size);
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result after launching the graph %d times:\n", launchCount);
    for (int i = 0; i < 10; i++) { // Print the first 10 elements
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // Clean up
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
    cudaFree(d_a);
    free(h_a);
    cudaStatus = cudaGetLastError();
    return cudaStatus;
}