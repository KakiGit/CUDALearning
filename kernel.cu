
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>


cudaError_t addWithCuda();
cudaError_t matrixMulWithCuda();

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
   
    //cudaError_t cudaStatus = addWithCuda();
    cudaError_t cudaStatus = matrixMulWithCuda();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "task failed!");
        return 1;
    }

 

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
    return cudaStatus;
}

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