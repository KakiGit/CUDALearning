#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16
#define WIDTH 32

// Kernel function to multiply two matrices using shared memory
__global__ void matrixMulShared(int* a, int* b, int* c, int width) {
    __shared__ int tile_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ int tile_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int value = 0;

    for (int m = 0; m < width / TILE_WIDTH; ++m) {
        tile_a[ty][tx] = a[row * width + (m * TILE_WIDTH + tx)];
        tile_b[ty][tx] = b[(m * TILE_WIDTH + ty) * width + col];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            value += tile_a[ty][k] * tile_b[k][tx];
        }

        __syncthreads();
    }

    c[row * width + col] = value;
}

cudaError_t matrixMulWithCuda() {
    printf("starting %s\n", __func__);

    cudaError_t cudaStatus;

    int width = WIDTH;
    int size = width * width * sizeof(int);
    int* a, * b, * c;
    int* d_a, * d_b, * d_c;

    // Allocate memory on the host (CPU)
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Initialize matrices a and b
    for (int i = 0; i < width * width; i++) {
        a[i] = 1; // You can use any initialization
        b[i] = 2; // You can use any initialization
    }

    // Allocate memory on the device (GPU)
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy matrices a and b to the device (GPU)
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Define the number of threads per block and the number of blocks
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid(width / TILE_WIDTH, width / TILE_WIDTH);

    // Launch the kernel
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width);

    // Copy the result matrix c back to the host (CPU)
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result
    printf("Result:\n");
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            printf("%d ", c[i * width + j]);
        }
        printf("\n");
    }

    // Free the memory on the device (GPU)
    cudaFree(d_a);
    cudaFree(d_b);
    cudaStatus = cudaFree(d_c);

    // Free the memory on the host (CPU)
    free(a);
    free(b);
    free(c);

    return cudaStatus;
}