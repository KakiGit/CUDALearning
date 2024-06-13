#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>

#define DEBUG false

#define IMAGE_SIZE 28
#define INPUT_SIZE IMAGE_SIZE * IMAGE_SIZE
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10
#define LAYER_SIZES {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE}
#define NUM_LAYERS

#define LEARNING_RATE 3
#define EPOCHS 20
#define NUM_OF_SAMPLES 100



__device__ float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

__device__ float sigmoidPrime(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

__global__ void forwardPass(
	float* d_preActivation, float* d_activation,
	float* d_weights, float* d_biases) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;


	int hidden_offset_a = INPUT_SIZE;

	int output_offset_w = INPUT_SIZE * HIDDEN_SIZE;
	int output_offset_a = INPUT_SIZE + HIDDEN_SIZE;

	// Init d_preActivation
	if (idx < HIDDEN_SIZE + OUTPUT_SIZE) {
		d_preActivation[idx] = 0.0f;
	}
	__syncthreads();
	// Hidden layer preActivation
	if (idx < INPUT_SIZE) {
		for (int i = 0; i < HIDDEN_SIZE; i++) {
			atomicAdd(
				&d_preActivation[i],
				d_activation[idx] * d_weights[idx * HIDDEN_SIZE + i] + d_biases[i]
			);
			if (DEBUG) printf(
				"INPUT idx %d d_activation %.3f weight %.3f bias %.3f \n",
				idx, d_activation[idx], d_weights[idx * HIDDEN_SIZE + i], d_biases[hidden_offset_a + i]
			);
		}
	}
	__syncthreads();
	// Hidden layer activation
	// Output layer preActivation
	if (idx < HIDDEN_SIZE) {
		d_activation[hidden_offset_a + idx] = sigmoid(d_preActivation[idx]);
		if (DEBUG) printf(
			"HIDDEN idx %d d_preActivation %.3f d_activation %.3f\n",
			idx, d_preActivation[idx], d_activation[hidden_offset_a + idx]
		);
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			atomicAdd(
				&d_preActivation[HIDDEN_SIZE + i],
				d_activation[hidden_offset_a + idx] * d_weights[output_offset_w + idx * OUTPUT_SIZE + i] + d_biases[HIDDEN_SIZE + i]
			);
			if (DEBUG) printf("HIDDEN idx %d OUTPUT idx %d weight %.3f biases %.3f\n",
				hidden_offset_a + idx, output_offset_a + i, d_weights[output_offset_w + idx * OUTPUT_SIZE + i], d_biases[output_offset_a + i]
			);
		}
	}
	__syncthreads();
	// Output layer activation
	if (idx < OUTPUT_SIZE) {
		d_activation[output_offset_a + idx] = sigmoid(d_preActivation[HIDDEN_SIZE + idx]);
		if (DEBUG) printf("OUTPUT idx %d d_preActivation %.3f d_activation %.3f\n",
			idx, d_preActivation[HIDDEN_SIZE + idx], d_activation[output_offset_a + idx]
		);
	}

}

__global__ void backpropagate(
	float* d_target, float* d_activation, float* d_preActivation,
	float* d_weights, float* d_biases, float* d_deltas, float* d_nabla_w, float* d_nabla_b, float * learningRate) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	int hidden_offset_a = INPUT_SIZE;

	int output_offset_w = INPUT_SIZE * HIDDEN_SIZE;
	int output_offset_a = INPUT_SIZE + HIDDEN_SIZE;

	// Init d_preActivation
	if (idx < HIDDEN_SIZE + OUTPUT_SIZE) {
		d_preActivation[idx] = 0.0;
		if (DEBUG) printf("idx %d d_preActivation %.3f\n", idx, d_preActivation[idx]);
	}
	__syncthreads();
	// Hidden layer preActivation
	if (idx < INPUT_SIZE) {
		for (int i = 0; i < HIDDEN_SIZE; i++) {
			atomicAdd(
				&d_preActivation[i],
				d_activation[idx] * d_weights[idx * HIDDEN_SIZE + i] + d_biases[i]
			);
		}
	}
	__syncthreads();
	// Hidden layer activation
	// Output layer preActivation
	if (idx < HIDDEN_SIZE) {
		d_activation[hidden_offset_a + idx] = sigmoid(d_preActivation[idx]);
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			atomicAdd(
				&d_preActivation[HIDDEN_SIZE + i],
				d_activation[hidden_offset_a + idx] * d_weights[output_offset_w + idx * OUTPUT_SIZE + i] + d_biases[HIDDEN_SIZE + i]
			);
		}
	}
	__syncthreads();
	// Output layer activation
	if (idx < OUTPUT_SIZE) {
		d_activation[output_offset_a + idx] = sigmoid(d_preActivation[HIDDEN_SIZE + idx]);
		if (DEBUG) printf("OUTPUT idx %d d_preActivation %.3f d_activation %.3f\n", output_offset_a + idx, d_preActivation[output_offset_a + idx], d_activation[output_offset_a + idx]);
	}
	__syncthreads();

	// backward pass
	if (idx < OUTPUT_SIZE) {
		d_deltas[HIDDEN_SIZE + idx] = (d_activation[output_offset_a + idx] - d_target[idx]);
		// d_deltas[HIDDEN_SIZE + idx] = (d_activation[output_offset_a + idx] - d_target[idx]) * sigmoidPrime(d_preActivation[HIDDEN_SIZE + idx]);
		d_nabla_b[HIDDEN_SIZE + idx] = d_deltas[HIDDEN_SIZE + idx];
		for (int i = 0; i < HIDDEN_SIZE; i++) {
			d_nabla_w[output_offset_w + i * OUTPUT_SIZE + idx] = d_deltas[output_offset_a + idx] * d_activation[hidden_offset_a + i];
			atomicAdd(&d_deltas[i], d_deltas[HIDDEN_SIZE + idx] * d_weights[output_offset_w + i * OUTPUT_SIZE + idx]);
		}
		if (DEBUG) printf("OUTPUT deltas idx %d deltas %.3f\n", output_offset_a + idx, d_deltas[output_offset_a + idx]);
	}
	__syncthreads();
	if (idx < HIDDEN_SIZE) {
		d_deltas[idx] = d_deltas[idx] * sigmoidPrime(d_preActivation[idx]);
		d_nabla_b[idx] = d_deltas[idx];
		if (DEBUG) printf("d_deltas idx %d d_nabla_b %.3f d_deltas %.3f\n", hidden_offset_a + idx, d_nabla_b[hidden_offset_a + idx], d_deltas[hidden_offset_a + idx]);
		for (int i = 0; i < INPUT_SIZE; i++) {
			d_nabla_w[i * HIDDEN_SIZE + idx] = d_deltas[idx] * d_activation[i];
		}
	}
	__syncthreads();
	if (idx < INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE) {
		d_weights[idx] = d_weights[idx] - *learningRate * d_nabla_w[idx] / NUM_OF_SAMPLES;
		if (DEBUG) printf("d_weights idx %d d_nabla_w %.3f d_weights %.3f\n", idx, d_nabla_w[idx], d_weights[idx]);
	}
	__syncthreads();
	if (idx < HIDDEN_SIZE + OUTPUT_SIZE) {
		d_biases[idx] = d_biases[idx] - *learningRate * d_nabla_b[idx] / NUM_OF_SAMPLES;
		if (DEBUG) printf("d_biases idx %d d_nabla_b %.3f d_biases %.3f\n", idx, d_nabla_b[idx], d_biases[idx]);
	}
}

void trainNeuralNetwork(float* trainData, float* trainLabels, float* weights, float* biases, float learningRate) {
	float* d_target, * d_preActivation, * d_activation;
	float* d_weights, * d_biases, * d_deltas, * d_nabla_w, * d_nabla_b, * d_learningRate;

	int numOfWeights = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE);
	int activationSize = (INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(float);
	int preActivationSize = (HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(float);
	int weightSizes = numOfWeights * sizeof(float);

	int inputSize = INPUT_SIZE * sizeof(float);
	int outputSize = OUTPUT_SIZE * sizeof(float);

	cudaMalloc((void**)&d_target, outputSize);
	cudaMalloc((void**)&d_preActivation, preActivationSize);
	cudaMalloc((void**)&d_activation, activationSize);
	cudaMalloc((void**)&d_weights, weightSizes);
	cudaMalloc((void**)&d_biases, preActivationSize);
	cudaMalloc((void**)&d_deltas, preActivationSize);
	cudaMalloc((void**)&d_nabla_w, weightSizes);
	cudaMalloc((void**)&d_nabla_b, preActivationSize);
	cudaMalloc((void**)&d_learningRate, sizeof(float));
	cudaMemcpy(d_weights, weights, weightSizes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_biases, biases, preActivationSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_learningRate, &learningRate, sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = (numOfWeights + threadsPerBlock) / threadsPerBlock;

	for (int i = 0; i < NUM_OF_SAMPLES; i++) {
		cudaMemcpy(d_activation, &trainData[i * INPUT_SIZE], inputSize, cudaMemcpyHostToDevice);
		cudaMemcpy(d_target, &trainLabels[i * OUTPUT_SIZE], outputSize, cudaMemcpyHostToDevice);

		backpropagate<<<blocksPerGrid, threadsPerBlock>>>(
			d_target, d_activation, d_preActivation,
			d_weights, d_biases, d_deltas, d_nabla_w, d_nabla_b, d_learningRate
			);
	}

	cudaMemcpy(weights, d_weights, weightSizes, cudaMemcpyDeviceToHost);
	cudaMemcpy(biases, d_biases, preActivationSize, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	cudaFree(d_target);
	cudaFree(d_preActivation);
	cudaFree(d_activation);
	cudaFree(d_weights);
	cudaFree(d_biases);
	cudaFree(d_deltas);
	cudaFree(d_nabla_w);
	cudaFree(d_nabla_b);
}

int evaluateNeuralNetwork(float* trainData, float* trainLabels, float* weights, float* biases, int numOfSamples) {
	float* d_preActivation, * d_activation, * h_output;
	float* d_weights, * d_biases, * d_deltas, * d_nabla_w, * d_nabla_b;

	int numOfWeights = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE);
	int activationSize = (INPUT_SIZE + HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(float);
	int preActivationSize = (HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(float);
	int weightSizes = numOfWeights * sizeof(float);

	int inputSize = INPUT_SIZE * sizeof(float);
	int outputSize = OUTPUT_SIZE * sizeof(float);

	h_output = (float*)malloc(OUTPUT_SIZE * sizeof(float));

	cudaMalloc(&d_preActivation, preActivationSize);
	cudaMalloc(&d_activation, activationSize);
	cudaMalloc(&d_weights, weightSizes);
	cudaMalloc(&d_biases, preActivationSize);

	cudaMemcpy(d_weights, weights, weightSizes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_biases, biases, preActivationSize, cudaMemcpyHostToDevice);

	int threadsPerBlock = 1024;
	int blocksPerGrid = (numOfWeights + threadsPerBlock) / threadsPerBlock;
	int correctCount = 0;
	for (int i = 0; i < numOfSamples; i++) {
		cudaMemcpy(d_activation, &trainData[i * INPUT_SIZE], inputSize, cudaMemcpyHostToDevice);

		forwardPass<<<blocksPerGrid, threadsPerBlock>>>(
			d_preActivation, d_activation,
			d_weights, d_biases
			);
		cudaMemcpy(h_output, &d_activation[INPUT_SIZE + HIDDEN_SIZE], outputSize, cudaMemcpyDeviceToHost);

		bool same = true;
		if (DEBUG) {
			printf("target:\t", correctCount, numOfSamples);
			for (int j = 0; j < OUTPUT_SIZE; j++) {
				printf("%d ", static_cast<int>(trainLabels[i * OUTPUT_SIZE + j]));
			}
			printf("\n");
			printf("output:\t", correctCount, numOfSamples);
			for (int j = 0; j < OUTPUT_SIZE; j++) {
				printf("%d ", static_cast<int>(h_output[j]));
			}
			printf("\n");
		}
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			if (static_cast<int>(trainLabels[i * OUTPUT_SIZE + j]) != static_cast<int>(h_output[j])) same &= false;
		}
		if (same) correctCount += 1;
	}
	cudaDeviceSynchronize();
	printf("%d/%d correct\n", correctCount, numOfSamples);
	cudaFree(d_preActivation);
	cudaFree(d_activation);
	cudaFree(d_weights);
	cudaFree(d_biases);
	return correctCount;
}

void readImages(float* trainData, int offset, int num_images) {
	unsigned char* pixels = (unsigned char*)malloc(INPUT_SIZE * num_images * sizeof(unsigned char));
	std::ifstream imagesFile("resources/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
	imagesFile.seekg(16 + offset * INPUT_SIZE * sizeof(char));
	imagesFile.read((char*)pixels, INPUT_SIZE * num_images * sizeof(char));
	imagesFile.close();
	for (int i = 0; i < INPUT_SIZE * num_images; i++) {
		trainData[i] = static_cast<float>(pixels[i]) / 255;
	}
	free(pixels);
}

void readLabels(float* trainLabels, int offset, int num_labels) {
	unsigned char* labels = (unsigned char*)malloc(num_labels * sizeof(unsigned char));
	std::ifstream labelFile("resources/t10k-labels-idx1-ubyte", std::ios::in | std::ios::binary);
	labelFile.seekg(8 + offset * sizeof(char));
	labelFile.read((char*)labels, num_labels * sizeof(char));
	labelFile.close();
	for (int i = 0; i < num_labels; i++) {
		trainLabels[i * OUTPUT_SIZE + static_cast<int>(labels[i])] = 1.0f;
	}
	free(labels);
}

cudaError_t cudaTrainNeuralNetwork() {
	printf("starting %s\n", __func__);
	cudaError_t cudaStatus;


	int numOfWeights = (INPUT_SIZE * HIDDEN_SIZE) + (HIDDEN_SIZE * OUTPUT_SIZE);

	float* weights = (float*)malloc(numOfWeights * sizeof(float));
	float* biases = (float*)malloc((HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(float));
	float* ori_weights = (float*)malloc(numOfWeights * sizeof(float));
	float* ori_biases = (float*)malloc((HIDDEN_SIZE + OUTPUT_SIZE) * sizeof(float));


	// Initialize weights and biases
	for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE; ++i) {
		weights[i] = (float)rand() / RAND_MAX * 2 - 1;
		ori_weights[i] = weights[i];
	}
	for (int i = 0; i < HIDDEN_SIZE + OUTPUT_SIZE; ++i) {
		biases[i] = (float)rand() / RAND_MAX * 2 - 1;
		ori_biases[i] = biases[i];
	}
	int max_count = 0;
	for (int epoch = 0; epoch < EPOCHS; ++epoch) {
		float* trainData = (float*)malloc(INPUT_SIZE * NUM_OF_SAMPLES * sizeof(float));
		float* trainLabels = (float*)malloc(OUTPUT_SIZE * NUM_OF_SAMPLES * sizeof(float));
		std::fill_n(trainLabels, OUTPUT_SIZE * NUM_OF_SAMPLES, 0);
		int weightChangeCount = 0, biasChangeCount = 0;
		// Load images and labels
		int rand_idx = static_cast<int>(10000 * (float)rand() / RAND_MAX - NUM_OF_SAMPLES - 1);
		readImages(trainData, rand_idx, NUM_OF_SAMPLES);
		readLabels(trainLabels, rand_idx, NUM_OF_SAMPLES);
		trainNeuralNetwork(trainData, trainLabels, weights, biases, LEARNING_RATE);
		free(trainData);
		free(trainLabels);
		for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE; ++i) {
			if (ori_weights[i] != weights[i]) {
				weightChangeCount += 1;
			}
		}
		for (int i = 0; i < HIDDEN_SIZE + OUTPUT_SIZE; ++i) {
			if (ori_biases[i] != biases[i]) {
				biasChangeCount += 1;
			}
		}
		printf("Weight %d changed. Bias %d changed\n", weightChangeCount, biasChangeCount);
		float* evaluateData = (float*)malloc(INPUT_SIZE * 10000 * sizeof(float));
		float* evaluateLabels = (float*)malloc(OUTPUT_SIZE * 10000 * sizeof(float));
		std::fill_n(evaluateLabels, OUTPUT_SIZE * 10000, 0);
		readImages(evaluateData, 0, 10000);
		readLabels(evaluateLabels, 0, 10000);
		int count = evaluateNeuralNetwork(evaluateData, evaluateLabels, weights, biases, 10000);
		if (count > max_count) max_count = count;
		printf("Epoch %d done. Max %d / 10000\n", epoch + 1, max_count);
		free(evaluateData);
		free(evaluateLabels);
	}


	free(weights);
	free(biases);
	free(ori_weights);
	free(ori_biases);
	return cudaGetLastError();
}
