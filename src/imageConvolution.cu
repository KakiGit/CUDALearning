#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>

#define MASK_WIDTH 3
#define TILE_WIDTH 16

__global__ void convolutionKernel(float* input, float* output, float* mask, int width, int height) {
	__shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y * TILE_WIDTH + ty;
	int col_o = blockIdx.x * TILE_WIDTH + tx;
	// Input index is 1 pixel shift than output index
	// So output[0][0] -> tile[1][1] neighbors -> input[0][0]
	// There are some edge cases but this is good enough.
	int row_i = row_o - 1;
	int col_i = col_o - 1;

	if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width)) {
		tile[ty][tx] = input[row_i * width + col_i];
	}
	else {
		tile[ty][tx] = 0.0f;
	}

	__syncthreads();

	float output_value = 0.0f;
	for (int i = 0; i < MASK_WIDTH; ++i) {
		for (int j = 0; j < MASK_WIDTH; ++j) {
			output_value += tile[ty + i][tx + j] * mask[i * MASK_WIDTH + j];
		}
	}

	if (row_o < height && col_o < width) {
		output[row_o * width + col_o] = output_value > 0.0f ? output_value : 0.0f;
	}
}

void convolution(float* input, float* output, float* mask, int width, int height) {
	int size = width * height * sizeof(float);
	float* d_input, * d_output, * d_mask;

	cudaMalloc((void**)&d_input, size);
	cudaMalloc((void**)&d_output, size);
	cudaMalloc((void**)&d_mask, MASK_WIDTH * MASK_WIDTH * sizeof(float));

	cudaMemcpy(d_input, input, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_mask, mask, MASK_WIDTH * MASK_WIDTH * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
	convolutionKernel<<<dimGrid, dimBlock>>>(d_input, d_output, d_mask, width, height);

	cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_mask);
}

cudaError_t cudaImageConvolution() {
	printf("starting %s\n", __func__);

	cudaError_t cudaStatus;
	int width = 1024;
	int height = 1024;
	int size = width * height * sizeof(float);

	float* input = (float*)malloc(size);
	float* output = (float*)malloc(size);
	float unsharp_kernel[MASK_WIDTH * MASK_WIDTH] = {
		0, -1, 0,
		-1, 5, -1,
		0, -1, 0
	};

	// Initialize input image
	for (int i = 0; i < width * height; ++i) {
		input[i] = (float)(i % 256);
	}

	convolution(input, output, unsharp_kernel, width, height);

	// Output some values for verification
	for (int i = 0; i < 10; ++i) {
		printf("%f ", output[i]);
	}
	printf("\n");

	free(input);
	free(output);

	cudaStatus = cudaGetLastError();
	return cudaStatus;
}