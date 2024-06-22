#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <neuralNetwork.h>
#include <random>
#include <string>
#include <sstream>

#define DEBUG false

#define IMAGE_HEIGHT 28
#define IMAGE_WIDTH 28
#define IMAGE_SIZE 28
#define INPUT_SIZE IMAGE_SIZE * IMAGE_SIZE
#define HIDDEN_SIZE 16
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

int maxThreadsDimx, maxThreadsDimy, maxThreadsPerBlock;

__global__ void cudaDot(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = 0.0;
		for (int i = 0; i < m1Shape->second; i++) {
			resValues[idy * resShape->second + idx] += m1Values[idy * m1Shape->second + i] * m2Values[i * m2Shape->second + idx];
		}
	}
}

__global__ void cudaMul(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx] * m2Values[idy * resShape->second + idx];
	}
}

__global__ void cudaAdd(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx] + m2Values[idy * resShape->second + idx];
	}
}

__global__ void cudaSub(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx] - m2Values[idy * resShape->second + idx];
	}
}

__global__ void cudaEqual(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = m1Values[idy * resShape->second + idx];
	}
}

// m2 not used
__global__ void cudaTranspose(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = m1Values[idx * m1Shape->second + idy];
	}
}

__global__ void cudaSigmoid(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = sigmoid(m1Values[idy * resShape->second + idx]);
	}
}

__global__ void cudaSigmoidPrime(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx < resShape->second && idy < resShape->first) {
		resValues[idy * resShape->second + idx] = sigmoidPrime(m1Values[idy * m1Shape->second + idx]);
	}
}

Matrix::Matrix(float* values, _2DShape shape) : m_shape(shape) {
	int size = m_shape.first * m_shape.second * sizeof(float);
	cudaMalloc((void**)&m_values, size);
	cudaMemcpy(m_values, values, size, cudaMemcpyHostToDevice);
}
Matrix::Matrix(const Matrix& rhs) : m_shape(rhs.m_shape) {
	int size = m_shape.first * m_shape.second * sizeof(float);
	cudaMalloc((void**)&m_values, size);
	cudaMemcpy(m_values, rhs.m_values, size, cudaMemcpyDeviceToDevice);
}
Matrix::Matrix(_2DShape shape) : m_shape(shape) {
	int size = m_shape.first * m_shape.second * sizeof(float);
	cudaMalloc((void**)&m_values, size);
	float* zeros = (float*)malloc(size);
	std::fill_n(zeros, m_shape.first * m_shape.second, 0.0f);
	cudaMemcpy(m_values, zeros, size, cudaMemcpyHostToDevice);
	free(zeros);
}

Matrix::~Matrix() {
	cudaFree(m_values);
}
float* Matrix::getValues() { return m_values; }
_2DShape Matrix::getShape() const { return m_shape; }

std::pair<dim3, dim3> findOptimalDims(_2DShape shape) {
	if (shape.first * shape.second <= maxThreadsPerBlock) {
		return std::make_pair(dim3(shape.second, shape.first), dim3(1, 1));
	}
	int bestTx = 1, bestTy = 1;
	int minBlocks = ceil(static_cast<float>(shape.second) / bestTy) * ceil(static_cast<float>(shape.first) / bestTx);
	int bx, by;

	for (int tx = 1; tx <= maxThreadsPerBlock; tx++) {
		int ty = std::min(maxThreadsPerBlock, maxThreadsPerBlock / tx);
		bx = ceil(static_cast<float>(shape.second) / tx);
		by = ceil(static_cast<float>(shape.first) / ty);
		int totalBlocks = bx * by;
		if (totalBlocks < minBlocks) {
			bestTx = tx;
			bestTy = ty;
			minBlocks = totalBlocks;
		}
	}
	bx = ceil(static_cast<float>(shape.second) / bestTx);
	by = ceil(static_cast<float>(shape.first) / bestTy);

	return std::make_pair(dim3(bestTx, bestTy), dim3(bx, by));
}

Matrix Matrix::launchCudaMatrixCalculation(
	const Matrix& m1, const Matrix& m2, _2DShape resShape,
	void (*cudaFunc)(
		float*, float*, float*,
		_2DShape*, _2DShape*, _2DShape*)) {

	_2DShape* d_m1Shape, * d_m2Shape, * d_resShape;
	Matrix res(resShape);

	auto dims = findOptimalDims(resShape);
	dim3 threadPerBlock = dims.first;
	dim3 blocksPerGrid = dims.second;
	cudaMalloc(&d_m1Shape, sizeof(_2DShape));
	cudaMalloc(&d_m2Shape, sizeof(_2DShape));
	cudaMalloc(&d_resShape, sizeof(_2DShape));
	cudaMemcpy(d_m1Shape, &m1.m_shape, sizeof(_2DShape), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m2Shape, &m2.m_shape, sizeof(_2DShape), cudaMemcpyHostToDevice);
	cudaMemcpy(d_resShape, &res.m_shape, sizeof(_2DShape), cudaMemcpyHostToDevice);
	cudaFunc << <blocksPerGrid, threadPerBlock >> > (
		m1.m_values, m2.m_values, res.m_values,
		d_m1Shape, d_m2Shape, d_resShape);
	cudaFree(d_m1Shape);
	cudaFree(d_m2Shape);
	cudaFree(d_resShape);
	return res;
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

int Matrix::argMax() {
	int size = m_shape.first * m_shape.second * sizeof(float);
	int mIdx = 0;
	float mValue = 0;
	float* values = (float*)malloc(size);
	cudaMemcpy(values, m_values, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m_shape.first * m_shape.second; i++) {
		if (values[i] > mValue) {
			mValue = values[i];
			mIdx = i;
		}
	}
	return mIdx;
}

Matrix Matrix::operator * (const Matrix& m) {
	assert(this->m_shape.first == m.m_shape.first && this->m_shape.second == m.m_shape.second);
	return launchCudaMatrixCalculation(*this, m, _2DShape{ this->m_shape.first , this->m_shape.second }, cudaMul);
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
	return launchCudaMatrixCalculation(*this, m, _2DShape{ this->m_shape.first , this->m_shape.second }, cudaAdd);
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
    cudaMemcpy(m_values, m.m_values, size, cudaMemcpyDeviceToDevice);
	return *this;
}

bool Matrix::isEqual(const Matrix& m) const {
	bool same = true;
	int size = m_shape.first * m_shape.second * sizeof(float);
	float* src_values = (float*)malloc(size);
	float* dst_values = (float*)malloc(size);
	cudaMemcpy(src_values, m_values, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(dst_values, m.m_values, size, cudaMemcpyDeviceToHost);
	same &= m.m_shape == this->m_shape;
	for (int i = 0; i < m.m_shape.first * m.m_shape.second; i++) {
		same &= src_values[i] == dst_values[i];
	}
	free(src_values);
	free(dst_values);
	return same;
}

bool operator == (const Matrix& m1, const Matrix& m2) {
	return m1.isEqual(m2);
}

void Matrix::print() {
	int size = m_shape.first * m_shape.second * sizeof(float);
	float* values = (float*)malloc(size);
	cudaMemcpy(values, m_values, size, cudaMemcpyDeviceToHost);
	for (int i = 0; i < m_shape.first; i++) {
		for (int j = 0; j < m_shape.second; j++) {
			printf("%.3f\t", values[i * m_shape.second + j]);
		}
		printf("\n");
	}
	printf("-----\n");
	free(values);
}

std::string Matrix::toString() const {
	std::ostringstream oss;
	oss.precision(3);
	int size = m_shape.first * m_shape.second * sizeof(float);
	float* values = (float*)malloc(size);
	cudaMemcpy(values, m_values, size, cudaMemcpyDeviceToHost);
	oss << "\n";
	for (int i = 0; i < m_shape.first; i++) {
		for (int j = 0; j < m_shape.second; j++) {
			oss << values[i * m_shape.second + j] << "\t";
		}
		oss << "\n";
	}
	oss;
	free(values);
	return oss.str();
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
				new Matrix(&input[i * INPUT_SIZE], _2DShape{ IMAGE_HEIGHT * IMAGE_WIDTH, 1 }),
				new Matrix(&labels[i * OUTPUT_SIZE], _2DShape{ OUTPUT_SIZE, 1 })
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
	std::random_device rd{};
	std::mt19937 gen{ rd() };
	std::normal_distribution<float> d;
	auto v1 = std::vector<int>(m_shapes.begin(), m_shapes.end() - 1);
	auto v2 = std::vector<int>(m_shapes.begin() + 1, m_shapes.end());
	for (int i = 0; i < v1.size(); i++) {
		float* weights = (float*)malloc(v2[i] * v1[i] * sizeof(float));
		float* biases = (float*)malloc(v2[i] * sizeof(float));
		for (int j = 0; j < v2[i] * v1[i]; j++) {
			weights[j] = d(gen);
		}
		for (int j = 0; j < v2[i]; j++) {
			biases[j] = d(gen);
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
		for (auto batch : batches) {
			train(batch, eta);
			printf(".");
		}
		printf("\n");
		evaluate(testData);
		std::for_each(batches.begin(), batches.end(), [](auto item) {delete item; });
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
		std::vector<Matrix*> activations;
		activations.emplace_back(new Matrix(*item.first));
		for (int i = 0; i < m_weights.size(); i++) {
			Matrix z = m_weights[i]->dot(activations[i]) + *m_biases[i];
			activations.emplace_back(new Matrix(sigmoid(&z)));
		}
		float* values = (float*)malloc(OUTPUT_SIZE * sizeof(float));
		cudaMemcpy(values, item.second->getValues(), OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
		if (values[activations.back()->argMax()] == 1.0) correctCount += 1;
		free(values);
		auto deleteMatrix = [](auto item) { delete item; };
		std::for_each(activations.begin(), activations.end(), deleteMatrix);
	}
	printf("correct %d / %d\n", correctCount, data->getNumOfImages());
}

std::pair<std::vector<Matrix*>, std::vector<Matrix*>> NeuralNetwork::backProb(DATA_ITEM data) {
	std::vector<Matrix*> nabla_b, nabla_w, activations, preActivations;
	std::for_each(m_biases.begin(), m_biases.end(), [&](Matrix* item) { nabla_b.emplace_back(new Matrix(item->getShape())); });
	std::for_each(m_weights.begin(), m_weights.end(), [&](Matrix* item) { nabla_w.emplace_back(new Matrix(item->getShape())); });
	activations.emplace_back(new Matrix(*data.first));
	for (int i = 0; i < m_weights.size(); i++) {
		Matrix wz = m_weights[i]->dot(activations[i]);
		Matrix z = wz + *m_biases[i];
		preActivations.emplace_back(new Matrix(z));
		activations.emplace_back(new Matrix(sigmoid(preActivations[i])));
	}
	Matrix delta = (*activations.back() - *data.second) * sigmoidPrime(preActivations.back());
	*nabla_b.back() = delta;
	*nabla_w.back() = delta.dot(&activations[activations.size() - 2]->transpose());
	for (int i = 2; i < m_shapes.size(); i++) {
		Matrix sp = sigmoidPrime(preActivations[preActivations.size() - i]);
		Matrix delta = m_weights[m_weights.size() - i + 1]->transpose().dot(nabla_b[nabla_b.size() - i + 1]) * sp;
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
	int numOfTrainImages = 60000;
	int numOfTestImages = 10000;
	float* f_trainData = (float*)malloc(INPUT_SIZE * numOfTrainImages * sizeof(float));
	float* f_trainLabel = (float*)malloc(OUTPUT_SIZE * numOfTrainImages * sizeof(float));
	std::fill_n(f_trainLabel, OUTPUT_SIZE * numOfTrainImages, 0.0f);
	float* f_testData = (float*)malloc(INPUT_SIZE * numOfTestImages * sizeof(float));
	float* f_testLabel = (float*)malloc(OUTPUT_SIZE * numOfTestImages * sizeof(float));
	std::fill_n(f_testLabel, OUTPUT_SIZE * numOfTestImages, 0.0f);
	readImages(f_trainData, 0, numOfTrainImages, TRAIN_IMG_PATH);
	readLabels(f_trainLabel, 0, numOfTrainImages, TRAIN_LABEL_PATH);
	readImages(f_testData, 0, numOfTestImages, TEST_IMG_PATH);
	readLabels(f_testLabel, 0, numOfTestImages, TEST_LABEL_PATH);
	auto res = std::make_pair(
		new Data(f_trainData, f_trainLabel, numOfTrainImages),
		new Data(f_testData, f_testLabel, numOfTestImages)
	);
	free(f_trainData);
	free(f_trainLabel);
	free(f_testData);
	free(f_testLabel);
	return res;
}

cudaError_t cudaTrainNeuralNetwork() {
	printf("starting %s\n", __func__);
	// My GPU properties:
	// maxThreadsDim.x 1024 maxThreadsDim.y 1024 maxThreadsDim.z 64
	// maxThreadsPerBlock 1024 maxThreadsPerMultiProcessor 1024 maxBlocksPerMultiProcessor 16
	// multiProcessorCount 34
	// 34 * 16 * 1024 = 557056
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("maxThreadsDim.x %d maxThreadsDim.y %d maxThreadsDim.z %d\n maxThreadsPerBlock %d maxThreadsPerMultiProcessor %d maxBlocksPerMultiProcessor %d multiProcessorCount %d\n",
		prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
		prop.maxThreadsPerBlock, prop.maxThreadsPerMultiProcessor,
		prop.maxBlocksPerMultiProcessor, prop.multiProcessorCount);
	maxThreadsDimx = prop.maxThreadsDim[0];
	maxThreadsDimy = prop.maxThreadsDim[1];
	maxThreadsPerBlock = prop.maxThreadsPerBlock;
	cudaError_t cudaStatus;

	std::pair<Data*, Data*> data = readData();
	NeuralNetwork network(std::vector<int>({ INPUT_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE }));
	network.stochasticGradientDescent(data.first, data.second, 10, 30, 3.0);
	delete data.first;
	delete data.second;
	return cudaGetLastError();
}
