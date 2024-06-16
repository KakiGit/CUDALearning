#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <neuralNetwork.h>
#include <random>

#define DEBUG false

#define IMAGE_HEIGHT 28
#define IMAGE_WIDTH 28
#define IMAGE_SIZE 28
#define INPUT_SIZE IMAGE_SIZE * IMAGE_SIZE
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10
#define LAYER_SIZES {INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE}
#define NUM_LAYERS

#define LEARNING_RATE 3
#define EPOCHS 20
#define NUM_OF_SAMPLES 100

#define TEST_IMG_PATH "resources/t10k-images-idx3-ubyte"
#define TEST_LABEL_PATH "resources/t10k-labels-idx1-ubyte"
#define TRAIN_IMG_PATH "resources/train-images-idx3-ubyte"
#define TRAIN_LABEL_PATH "resources/train-labels-idx1-ubyte"


__global__ void cudaDot(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = 0.0;
	for (int i = 0; i < m1Shape->second; i++) {
		resValues[idy * resShape->second + idx] += m1Values[idy * m1Shape->second + i] * m2Values[i * m2Shape->second + idx];
	}
}

__global__ void cudaMul(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx] * m2Values[idy * resShape->second + idx];
}

__global__ void cudaAdd(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx] + m2Values[idy * resShape->second + idx];
	printf("idx %d idy %d %.3f = %.3f + %.3f\n", idx, idy, resValues[idy * resShape->second + idx], m1Values[idy * resShape->second + idx], m2Values[idy * resShape->second + idx]);
}

__global__ void cudaSub(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx] - m2Values[idy * resShape->second + idx];
}

__global__ void cudaEqual(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx];
}

// m2 not used
__global__ void cudaTranspose(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;
	resValues[idy * resShape->second + idx] = m1Values[idx * m1Shape->second + idy];
}

__global__ void cudaSigmoid(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = sigmoid(m1Values[idy * resShape->second + idx]);
}

__global__ void cudaSigmoidPrime(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x;
	int idy = threadIdx.y;

	resValues[idy * resShape->second + idx] = sigmoidPrime(m1Values[idy * m1Shape->second + idx]);
}

Matrix::Matrix(float* values, _2DShape shape) : m_shape(shape) {
	int size = m_shape.first * m_shape.second * sizeof(float);
	m_values = (float*)malloc(size);
	std::memcpy(m_values, values, size);
}
Matrix::Matrix(const Matrix& rhs) : m_shape(rhs.m_shape) {
	int size = m_shape.first * m_shape.second * sizeof(float);
	m_values = (float*)malloc(size);
	std::memcpy(m_values, rhs.m_values, size);
}
Matrix::Matrix(_2DShape shape) : m_shape(shape) {
	m_values = (float*)malloc(m_shape.first * m_shape.second * sizeof(float));
	std::fill_n(m_values, m_shape.first * m_shape.second, 0.0f);
}

Matrix::~Matrix() {
	free(m_values);
}
float* Matrix::getValues() { return m_values; }
_2DShape Matrix::getShape() const { return m_shape; }

Matrix Matrix::launchCudaMatrixCalculation(
	const Matrix& m1, const Matrix& m2, _2DShape resShape,
	void (*cudaFunc)(
		float*, float*, float*,
		_2DShape*, _2DShape*, _2DShape*)) {
	Matrix res(resShape);
	float* d_m1Values, * d_m2Values, * d_resValues;
	_2DShape* d_m1Shape, * d_m2Shape, * d_resShape;
	dim3 threadPerBlock(res.m_shape.second, res.m_shape.first);
	int blocksPerGrid = 1;
	cudaMalloc((void**)&d_m1Values, m1.m_shape.first * m1.m_shape.second * sizeof(float));
	cudaMalloc((void**)&d_m2Values, m2.m_shape.first * m2.m_shape.second * sizeof(float));
	cudaMalloc((void**)&d_resValues, res.m_shape.first * res.m_shape.second * sizeof(float));
	cudaMalloc((void**)&d_m1Shape, sizeof(_2DShape));
	cudaMalloc((void**)&d_m2Shape, sizeof(_2DShape));
	cudaMalloc((void**)&d_resShape, sizeof(_2DShape));
	cudaMemcpy(d_m1Values, m1.m_values, m1.m_shape.first * m1.m_shape.second * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2Values, m2.m_values, m2.m_shape.first * m2.m_shape.second * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m1Shape, &m1.m_shape, sizeof(_2DShape), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2Shape, &m2.m_shape, sizeof(_2DShape), cudaMemcpyHostToDevice);
	cudaMemcpy(d_resShape, &res.m_shape, sizeof(_2DShape), cudaMemcpyHostToDevice);
	cudaFunc << <blocksPerGrid, threadPerBlock >> > (
		d_m1Values, d_m2Values, d_resValues,
		d_m1Shape, d_m2Shape, d_resShape);
	cudaMemcpy(res.m_values, d_resValues, res.m_shape.first * res.m_shape.second * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_m1Values);
	cudaFree(d_m2Values);
	cudaFree(d_resValues);
	cudaFree(d_m1Shape);
	cudaFree(d_m2Shape);
	cudaFree(d_resShape);
	return Matrix(res);
}

Matrix Matrix::dot(Matrix* m1, Matrix* m2) {
	assert(m1->m_shape.second == m2->m_shape.first);
	return launchCudaMatrixCalculation(*m1, *m2, _2DShape{ m1->m_shape.first , m2->m_shape.second }, cudaDot);
}

Matrix Matrix::dot(Matrix* m) {
	return dot(this, m);
}

Matrix Matrix::transpose() {
	Matrix _empty(_2DShape{ 0,0 });
	return launchCudaMatrixCalculation(*this, _empty, _2DShape{ this->m_shape.second , this->m_shape.first }, cudaTranspose);
}


Matrix Matrix::operator * (const Matrix& m) {
	assert(this->m_shape.first == m.m_shape.first && this->m_shape.second == m.m_shape.second);
	return launchCudaMatrixCalculation(*this, m, _2DShape{ this->m_shape.first , this->m_shape.second }, cudaMul);
}

Matrix operator * (const Matrix& m1, const Matrix& m2) {
	return m1 * m2;
}

Matrix operator * (const float& scalar, const Matrix& m2) {
	_2DShape m2Shape = m2.getShape();
	float* nums = (float*)malloc(m2Shape.first * m2Shape.second * sizeof(float));
	std::fill_n(nums, m2Shape.first * m2Shape.second, scalar);
	Matrix m(nums, m2Shape);
	free(nums);
	return m * m2;
}

Matrix Matrix::operator + (const Matrix& m) {
	assert(this->m_shape.first == m.m_shape.first && this->m_shape.second == m.m_shape.second);
	printf("%d * %d + %d * %d\n", this->getShape().first, this->getShape().second, m.getShape().first, m.getShape().second);
	return launchCudaMatrixCalculation(*this, m, _2DShape{ this->m_shape.first , this->m_shape.second }, cudaAdd);
}

Matrix operator + (const Matrix& m1, const Matrix& m2) {
	return m1 + m2;
}

Matrix operator + (const float& scalar, const Matrix& m2) {
	_2DShape m2Shape = m2.getShape();
	float* nums = (float*)malloc(m2Shape.first * m2Shape.second * sizeof(float));
	std::fill_n(nums, m2Shape.first * m2Shape.second, scalar);
	Matrix m(nums, m2Shape);
	free(nums);
	return m + m2;
}

Matrix Matrix::operator - (const Matrix& m) {
	assert(this->m_shape.first == m.m_shape.first && this->m_shape.second == m.m_shape.second);
	return launchCudaMatrixCalculation(*this, m, _2DShape{ this->m_shape.first , this->m_shape.second }, cudaSub);
}

Matrix operator - (const Matrix& m1, const Matrix& m2) {
	return m1 - m2;
}

Matrix operator - (const float& scalar, const Matrix& m2) {
	_2DShape m2Shape = m2.getShape();
	float* nums = (float*)malloc(m2Shape.first * m2Shape.second * sizeof(float));
	std::fill_n(nums, m2Shape.first * m2Shape.second, scalar);
	Matrix m(nums, m2Shape);
	free(nums);
	return m - m2;
}


Matrix& Matrix::operator = (const Matrix& m) {
	int size = m_shape.first * m_shape.second * sizeof(float);
	m_shape = m.m_shape;
	std::memcpy(m_values, m.m_values, size);
	return *this;
}

Matrix sigmoid(Matrix* m1) {
	Matrix _empty(_2DShape{ 0,0 });
	return Matrix::launchCudaMatrixCalculation(*m1, _empty, _2DShape{ m1->getShape().first , m1->getShape().second }, cudaSigmoid);
}

Matrix sigmoidPrime(Matrix* m1) {
	Matrix _empty(_2DShape{ 0,0 });
	return Matrix::launchCudaMatrixCalculation(*m1, _empty, _2DShape{ m1->getShape().first , m1->getShape().second }, cudaSigmoidPrime);
}

Data::Data(
	float* input, float* labels,
	int numOfImages) : m_numOfImages(numOfImages) {
	for (int i = 0; i < numOfImages; i++) {
		m_data.emplace_back(
			std::make_pair(
				new Matrix(&input[i * INPUT_SIZE], _2DShape{IMAGE_HEIGHT * IMAGE_WIDTH, 1}),
				new Matrix(&labels[i * OUTPUT_SIZE], _2DShape{OUTPUT_SIZE, 1})
			));
	}
}

Data::Data(std::vector<DATA_ITEM> data) : m_numOfImages(data.size()) {
	std::for_each(
		data.begin(), data.end(),
		[&](DATA_ITEM item) {
			m_data.emplace_back(std::make_pair(new Matrix(*item.first), new Matrix(*item.second)));
		});
}

Data::Data(const Data& rhs) : m_numOfImages(rhs.m_numOfImages) {
	std::for_each(
		rhs.m_data.begin(), rhs.m_data.end(),
		[&](DATA_ITEM item) {
			m_data.emplace_back(std::make_pair(new Matrix(*item.first), new Matrix(*item.second)));
		});
}

Data::~Data() {
	auto deleteMatrix = [](std::pair<Matrix*, Matrix*> item) { delete item.first; delete item.second; };
	std::for_each(m_data.begin(), m_data.end(), deleteMatrix);
	m_data.clear();
}

std::vector<Data*> Data::getBatches(int batchSize) {
	std::vector<Data*> res;
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(m_data.begin(), m_data.end(), g);
	for (int i = 0; i < m_data.size(); i += batchSize) {
		auto endIt = m_data.end();
		if (i + batchSize <= m_data.size()) {
			endIt = m_data.begin() + i + batchSize;
		}
		std::vector<DATA_ITEM> idata = std::vector<DATA_ITEM>(m_data.begin() + i, endIt);
		res.emplace_back(new Data(idata));
	}
	return res;
}

std::vector<DATA_ITEM> Data::getAll() {
	return m_data;
}

int Data::getNumOfImages() { return m_numOfImages; }

NeuralNetwork::NeuralNetwork(std::vector<int> shapes) : m_shapes(shapes) {
	assert(m_shapes.size() >= 3);
	printf("Creating network with %d layers\n", shapes.size());
	auto v1 = std::vector<int>(m_shapes.begin(), m_shapes.end() - 1);
	auto v2 = std::vector<int>(m_shapes.begin() + 1, m_shapes.end());
	for (int i = 0; i < v1.size(); i++) {
		float* weights = (float*)malloc(v2[i] * v1[i] * sizeof(float));
		float* biases = (float*)malloc(v2[i] * sizeof(float));
		for (int j = 0; j < v2[i] * v1[i]; j++) {
			weights[j] = (float)rand() / RAND_MAX * 2 - 1;
		}
		for (int j = 0; j < v2[i]; j++) {
			biases[j] = (float)rand() / RAND_MAX * 2 - 1;
		}
		m_weights.emplace_back(new Matrix(weights, _2DShape{ v2[i], v1[i] }));
		m_biases.emplace_back(new Matrix(biases, _2DShape{ v2[i], 1 }));
		free(weights);
		free(biases);
	}
}
NeuralNetwork::~NeuralNetwork() {
	printf("Deleting network\n");
	auto deleteMatrix = [](auto item) { delete item; };
	std::for_each(m_weights.begin(), m_weights.end(), deleteMatrix);
	std::for_each(m_biases.begin(), m_biases.end(), deleteMatrix);
	m_weights.clear();
	m_biases.clear();
}

void NeuralNetwork::stochasticGradientDescent(Data* trainData, Data* testData, int epochs, int batchSize, float eta) {
	int nTestData = testData->getNumOfImages();
	int n = trainData->getNumOfImages();
	for (int epoch = 0; epoch < epochs; epoch++) {
		std::vector<Data*> batches = trainData->getBatches(batchSize);
		int idx = 0;
		for (auto batch : batches) {
			train(batch, eta);
			idx += 1;
		}
		evaluate(testData);
		printf("Epoch %d done\n", epoch);
	}
}

void NeuralNetwork::train(Data* data, float eta) {
	std::vector<Matrix*> nabla_b, nabla_w;
	std::for_each(m_biases.begin(), m_biases.end(), [&](Matrix* item) { nabla_b.emplace_back(new Matrix(item->getShape())); });
	std::for_each(m_weights.begin(), m_weights.end(), [&](Matrix* item) { nabla_w.emplace_back(new Matrix(item->getShape())); });
	for (DATA_ITEM item : data->getAll()) {
		// back propagation
		std::pair<std::vector<Matrix*>, std::vector<Matrix*>> delta = backProb(item);
		for (int i = 0; i < delta.first.size(); i++) {
			*nabla_b[i] = *nabla_b[i] + *delta.first[i];
		}
		for (int i = 0; i < delta.second.size(); i++) {
			*nabla_w[i] = *nabla_w[i] + *delta.second[i];
		}
		for (auto item : delta.first) {
			delete item;
		}
		for (auto item : delta.second) {
			delete item;
		}
	}
	for (int i = 0; i < nabla_b.size(); i++) {
		*m_biases[i] = *m_biases[i] - eta / data->getNumOfImages() * *nabla_b[i];
	}
	for (int i = 0; i < nabla_w.size(); i++) {
		*m_weights[i] = *m_weights[i] - eta / data->getNumOfImages() * *nabla_w[i];
	}
	for (auto item : nabla_b) {
		delete item;
	}
	for (auto item : nabla_w) {
		delete item;
	}
}

void NeuralNetwork::evaluate(Data* data) {
	int correctCount = 0;
	for (DATA_ITEM item : data->getAll()) {
		std::vector<Matrix*> activations, preActivations;
		activations.emplace_back(new Matrix(*item.first));
		for (int i = 0; i < m_weights.size(); i++) {
			Matrix z = m_weights[i]->dot(activations[i]) + *m_biases[i];
			activations.emplace_back(new Matrix(sigmoid(&z)));
		}
		bool same = true;
		_2DShape outputShape = item.second->getShape();
		for (int i = 0; i < outputShape.first * outputShape.second; i++) {
			same &= (static_cast<int>(activations.back()->getValues()[i]) == static_cast<int>(item.second->getValues()[i]));
		}
		if (same) correctCount += 1;
		auto deleteMatrix = [](auto item) { delete item; };
		std::for_each(activations.begin(), activations.end(), deleteMatrix);
		std::for_each(preActivations.begin(), preActivations.end(), deleteMatrix);
	}
	printf("correct %d / %d\n", correctCount, data->getNumOfImages());
}

std::pair<std::vector<Matrix*>, std::vector<Matrix*>> NeuralNetwork::backProb(DATA_ITEM data) {
	std::vector<Matrix*> nabla_b, nabla_w, activations, preActivations;
	std::for_each(m_biases.begin(), m_biases.end(), [&](Matrix* item) { nabla_b.emplace_back(new Matrix(item->getShape())); });
	std::for_each(m_weights.begin(), m_weights.end(), [&](Matrix* item) { nabla_w.emplace_back(new Matrix(item->getShape())); });
	activations.emplace_back(new Matrix(*data.first));
	for (int i = 0; i < m_weights.size(); i++) {
		Matrix z = m_weights[i]->dot(activations[i]) + *m_biases[i];
		preActivations.emplace_back(new Matrix(z));
		activations.emplace_back(new Matrix(sigmoid(preActivations[i])));
	}
	Matrix delta = (*activations.back() - *data.second) * sigmoidPrime(preActivations.back());
	*nabla_b.back() = delta;
	*nabla_w.back() = delta.dot(&activations[m_shapes.size() - 1 - 1]->transpose());
	for (int i = 2; i < m_shapes.size(); i++) {
		Matrix sp = sigmoidPrime(preActivations[preActivations.size() - i]);
		delta = m_weights[m_weights.size() - i + 1]->transpose().dot(&delta) * sp;
		*nabla_b[nabla_b.size() - i] = delta;
		*nabla_w[nabla_w.size() - i] = delta.dot(&activations[activations.size() - i - 1]->transpose());
	}
	auto deleteMatrix = [](auto item) { delete item; };
	std::for_each(activations.begin(), activations.end(), deleteMatrix);
	std::for_each(preActivations.begin(), preActivations.end(), deleteMatrix);
	return std::make_pair(nabla_b, nabla_w);
}

__device__ float sigmoid(float x) {
	return 1.0 / (1.0 + exp(-x));
}

__device__ float sigmoidPrime(float x) {
	return sigmoid(x) * (1 - sigmoid(x));
}

void readImages(float* trainData, int offset, int num_images, std::string path) {
	unsigned char* pixels = (unsigned char*)malloc(INPUT_SIZE * num_images * sizeof(unsigned char));
	std::ifstream imagesFile(path, std::ios::in | std::ios::binary);
	imagesFile.seekg(16 + offset * INPUT_SIZE * sizeof(char));
	imagesFile.read((char*)pixels, INPUT_SIZE * num_images * sizeof(char));
	imagesFile.close();
	for (int i = 0; i < INPUT_SIZE * num_images; i++) {
		trainData[i] = static_cast<float>(pixels[i]) / 255;
	}
	free(pixels);
}

void readLabels(float* trainLabels, int offset, int num_labels, std::string path) {
	unsigned char* labels = (unsigned char*)malloc(num_labels * sizeof(unsigned char));
	std::ifstream labelFile(path, std::ios::in | std::ios::binary);
	labelFile.seekg(8 + offset * sizeof(char));
	labelFile.read((char*)labels, num_labels * sizeof(char));
	labelFile.close();
	for (int i = 0; i < num_labels; i++) {
		trainLabels[i * OUTPUT_SIZE + static_cast<int>(labels[i])] = 1.0f;
	}
	free(labels);
}

std::pair<Data*, Data*> readData() {
	Data* trainData, * testData;
	int numOfTrainImages = 1000;
	int numOfTestImages = 1000;
	float* f_trainData = (float*)malloc(INPUT_SIZE * numOfTrainImages * sizeof(float));
	float* f_trainLabel = (float*)malloc(OUTPUT_SIZE * numOfTrainImages * sizeof(float));
	std::fill_n(f_trainLabel, OUTPUT_SIZE * numOfTrainImages, 0.0f);
	float* f_testData = (float*)malloc(INPUT_SIZE * numOfTestImages * sizeof(float));
	float* f_testLabel = (float*)malloc(OUTPUT_SIZE * numOfTestImages * sizeof(float));
	std::fill_n(f_testLabel, OUTPUT_SIZE * numOfTrainImages, 0.0f);
	readImages(f_trainData, 0, numOfTrainImages, TRAIN_IMG_PATH);
	readLabels(f_trainLabel, 0, numOfTrainImages, TRAIN_LABEL_PATH);
	readImages(f_testData, 0, numOfTestImages, TEST_IMG_PATH);
	readLabels(f_testLabel, 0, numOfTestImages, TEST_LABEL_PATH);
	return std::make_pair(
		new Data(f_trainData, f_trainLabel, numOfTrainImages),
		new Data(f_testData, f_testLabel, numOfTestImages)
	);
}

cudaError_t cudaTrainNeuralNetwork() {
	printf("starting %s\n", __func__);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	// 1024 1024 64 1024 1024
	printf("%d %d %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2], prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor);
	cudaError_t cudaStatus;

	Matrix m1(_2DShape{ 3,3 }), m2(_2DShape{ 3,3 });
	std::vector<float> v1(
		{ 3,2,3,
		  4,5,6,
		  7,8,9});

	std::vector<float> v2(
		{ 2,1,3,
		  8,2,4,
		  9,3,5});

	for (auto i = 0; i < v1.size(); i++) {
		m1.getValues()[i] = v1[i];
	}
	for (auto i = 0; i < v2.size(); i++) {
		m2.getValues()[i] = v2[i];
	}
	std::vector<Matrix> res;
	res.emplace_back(m1.dot(&m2));
	res.emplace_back(m1 * m2);
	res.emplace_back(m1 + m2);
	res.emplace_back(m1 - m2);
	res.emplace_back(m1.transpose());
	for (auto& item : res) {
		auto resShape = item.getShape();
		for (int i = 0; i < resShape.first; i++) {
			for (int j = 0; j < resShape.second; j++) {
				printf("%.3f\t", item.getValues()[i * resShape.second + j]);
			}
			printf("\n");
		}
		printf("-----\n");
	}
	std::pair<Data*, Data*> data = readData();
	NeuralNetwork network(std::vector<int>({ INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE }));
	network.stochasticGradientDescent(data.first, data.second, 10, 30, 3.0);
	delete data.first;
	delete data.second;

	return cudaGetLastError();
}
