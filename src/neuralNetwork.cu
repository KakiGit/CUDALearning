#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>

#define IMAGE_SIZE 28
#define INPUT_SIZE IMAGE_SIZE * IMAGE_SIZE
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.001
#define EPOCHS 10
#define NUM_OF_SAMPLES 256

__device__ float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

__device__ float sigmoidPrime(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

__global__ void forwardPass(float* input, float* weightsInputHidden, float* hiddenLayer, float* weightsHiddenOutput, float* outputLayer) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < HIDDEN_SIZE) {
		float sum = 0.0;
		for (int i = 0; i < INPUT_SIZE; ++i) {
			sum += input[i] * weightsInputHidden[i * HIDDEN_SIZE + idx];
		}
		hiddenLayer[idx] = sigmoid(sum);
	}

	__syncthreads();

	if (idx < OUTPUT_SIZE) {
		float sum = 0.0;
		for (int i = 0; i < HIDDEN_SIZE; ++i) {
			sum += hiddenLayer[i] * weightsHiddenOutput[i * OUTPUT_SIZE + idx];
		}
		outputLayer[idx] = sigmoid(sum);
	}
}

__global__ void backpropagate(float* input, float* hiddenLayer, float* outputLayer, float* target, float* weightsInputHidden, float* weightsHiddenOutput, float* hiddenLayerDeltas, float* outputLayerDeltas) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < OUTPUT_SIZE) {
		// WIP: might still have some issues
		float delta = (outputLayer[idx] - target[idx]) * sigmoidPrime(outputLayer[idx]);
		outputLayerDeltas[idx] = delta;

		for (int i = 0; i < HIDDEN_SIZE; ++i) {
			atomicAdd(&weightsHiddenOutput[i * OUTPUT_SIZE + idx], -LEARNING_RATE * delta * hiddenLayer[i]);
		}
	}

	__syncthreads();

	if (idx < HIDDEN_SIZE) {
		float deltaSum = 0.0;
		// WIP: might still have some issues
		for (int i = 0; i < OUTPUT_SIZE; ++i) {
			deltaSum += outputLayerDeltas[i] * weightsHiddenOutput[idx * OUTPUT_SIZE + i];
		}
		float delta = deltaSum * sigmoidPrime(hiddenLayer[idx]);
		hiddenLayerDeltas[idx] = delta;

		for (int i = 0; i < INPUT_SIZE; ++i) {
			atomicAdd(&weightsInputHidden[i * HIDDEN_SIZE + idx], -LEARNING_RATE * delta * input[i]);
		}
	}
}

void trainNeuralNetwork(float* trainData, float* trainLabels, float* weightsInputHidden, float* weightsHiddenOutput) {
	float* d_input, * d_hidden, * d_output, * d_weightsInputHidden, * d_weightsHiddenOutput;
	float* d_target, * d_error, * d_hiddenLayerDeltas, * d_outputLayerDeltas;
	int inputSize = INPUT_SIZE * sizeof(float);
	int hiddenSize = HIDDEN_SIZE * sizeof(float);
	int outputSize = OUTPUT_SIZE * sizeof(float);

	cudaMalloc(&d_input, inputSize);
	cudaMalloc(&d_hidden, hiddenSize);
	cudaMalloc(&d_output, outputSize);
	cudaMalloc(&d_weightsInputHidden, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
	cudaMalloc(&d_weightsHiddenOutput, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
	cudaMalloc(&d_target, outputSize);
	cudaMalloc(&d_error, outputSize);
	cudaMalloc(&d_hiddenLayerDeltas, hiddenSize);
	cudaMalloc(&d_outputLayerDeltas, outputSize);

	cudaMemcpy(d_weightsInputHidden, weightsInputHidden, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weightsHiddenOutput, weightsHiddenOutput, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = 1;
	for (int epoch = 0; epoch < EPOCHS; ++epoch) {
		for (int i = 0; i < NUM_OF_SAMPLES; i++) {
			cudaMemcpy(d_input, &trainData[i * INPUT_SIZE], inputSize, cudaMemcpyHostToDevice);
			cudaMemcpy(d_target, &trainLabels[i * OUTPUT_SIZE], outputSize, cudaMemcpyHostToDevice);

			int threadsPerBlock = 256;
			int blocksPerGrid = 1;
			forwardPass<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weightsInputHidden, d_hidden, d_weightsHiddenOutput, d_output);
			backpropagate<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_hidden, d_output, d_target, d_weightsInputHidden, d_weightsHiddenOutput, d_hiddenLayerDeltas, d_outputLayerDeltas);
		}
		printf("Epoch %d done\n", epoch + 1);
	}
	cudaMemcpy(weightsInputHidden, d_weightsInputHidden, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(weightsHiddenOutput, d_weightsHiddenOutput, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_hidden);
	cudaFree(d_output);
	cudaFree(d_weightsInputHidden);
	cudaFree(d_weightsHiddenOutput);
	cudaFree(d_target);
	cudaFree(d_error);
	cudaFree(d_hiddenLayerDeltas);
	cudaFree(d_outputLayerDeltas);
}

void verifyNeuralNetwork(float* trainData, float* trainLabels, float* weightsInputHidden, float* weightsHiddenOutput, int numOfSamples) {
	float* d_input, * d_hidden, * d_output, *h_output, * d_weightsInputHidden, * d_weightsHiddenOutput;
	float* d_target, * d_error, * d_hiddenLayerDeltas, * d_outputLayerDeltas;
	int inputSize = INPUT_SIZE * sizeof(float);
	int hiddenSize = HIDDEN_SIZE * sizeof(float);
	int outputSize = OUTPUT_SIZE * sizeof(float);

	h_output = (float*)malloc(outputSize);
	cudaMalloc(&d_input, inputSize);
	cudaMalloc(&d_hidden, hiddenSize);
	cudaMalloc(&d_output, outputSize);
	cudaMalloc(&d_weightsInputHidden, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
	cudaMalloc(&d_weightsHiddenOutput, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));

	cudaMemcpy(d_weightsInputHidden, weightsInputHidden, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weightsHiddenOutput, weightsHiddenOutput, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

	int threadsPerBlock = 256;
	int blocksPerGrid = 1;
	for (int i = 0; i < numOfSamples; i++) {
		cudaMemcpy(d_input, &trainData[i * INPUT_SIZE], inputSize, cudaMemcpyHostToDevice);

		int threadsPerBlock = 256;
		int blocksPerGrid = 1;
		forwardPass<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_weightsInputHidden, d_hidden, d_weightsHiddenOutput, d_output);

		cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
		printf("Verify result: ");
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			printf("%.3f ", h_output[j]);
		}
		printf("\n");
		printf("Real value: ");
		for (int j = 0; j < OUTPUT_SIZE; j++) {
			printf("%.3f ", trainLabels[i * OUTPUT_SIZE + j]);
		}
		printf("\n");
	}

	cudaFree(d_input);
	cudaFree(d_hidden);
	cudaFree(d_output);
	cudaFree(d_weightsInputHidden);
	cudaFree(d_weightsHiddenOutput);
	free(h_output);
}

void readImages(float* trainData, int offset, int num_images) {
	unsigned char* pixels = (unsigned char*)malloc(INPUT_SIZE * num_images * sizeof(unsigned char));
	std::ifstream imagesFile("resources/t10k-images-idx3-ubyte", std::ios::in | std::ios::binary);
	imagesFile.seekg(16 + offset * INPUT_SIZE * sizeof(char));
	imagesFile.read((char*)pixels, INPUT_SIZE * num_images * sizeof(char));
	imagesFile.close();
	for (int i = 0; i < INPUT_SIZE * num_images; i++) {
		trainData[i] = static_cast<float>(pixels[i]);
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
	float* trainData = (float*)malloc(INPUT_SIZE * NUM_OF_SAMPLES * sizeof(float));
	float* trainLabels = (float*)malloc(OUTPUT_SIZE * NUM_OF_SAMPLES * sizeof(float));
	std::fill_n(trainLabels, OUTPUT_SIZE * NUM_OF_SAMPLES, 0);

	float* weightsInputHidden = (float*)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
	float* weightsHiddenOutput = (float*)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));

	// Load images and labels
	readImages(trainData, 0, NUM_OF_SAMPLES);
	readLabels(trainLabels, 0, NUM_OF_SAMPLES);

	// Initialize weights
	for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; ++i) {
		weightsInputHidden[i] = (float)rand() / RAND_MAX;
	}
	for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; ++i) {
		weightsHiddenOutput[i] = (float)rand() / RAND_MAX;
	}

	trainNeuralNetwork(trainData, trainLabels, weightsInputHidden, weightsHiddenOutput);

	std::fill_n(trainLabels, OUTPUT_SIZE * NUM_OF_SAMPLES, 0);
	readImages(trainData, NUM_OF_SAMPLES, 2);
	readLabels(trainLabels, NUM_OF_SAMPLES, 2);
	verifyNeuralNetwork(trainData, trainLabels, weightsInputHidden, weightsHiddenOutput, 2);
	free(trainData);
	free(trainLabels);
	free(weightsInputHidden);
	free(weightsHiddenOutput);

	return cudaGetLastError();
}
