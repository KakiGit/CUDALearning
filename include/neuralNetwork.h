#pragma once
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define _2DShape std::pair<int, int>
#define DATA_ITEM std::pair<Matrix*, Matrix*>

class Matrix {
public:
	Matrix(float* values, _2DShape shape);
	Matrix(const Matrix& rhs);
	Matrix(_2DShape shape);
	~Matrix();
	float* getValues();
	_2DShape getShape() const;
	static Matrix launchCudaMatrixCalculation(
		const Matrix& m1,const Matrix& m2, _2DShape resShape,
		void (*cudaFunc)(
			float*, float*, float*,
			_2DShape*, _2DShape*, _2DShape*));
	static Matrix dot(Matrix* m1, Matrix* m2);
	Matrix dot(Matrix* m1);
	Matrix transpose();
	int argMax();
	Matrix operator * (const Matrix& m);
	Matrix operator + (const Matrix& m);
	Matrix operator - (const Matrix& m);
	Matrix& operator = (const Matrix& m);
	bool isEqual (const Matrix& m) const;
	void print();
	std::string toString() const;

private:
	_2DShape m_shape;
	float* m_values;
};

Matrix operator * (const float& scalar, const Matrix& m2);
Matrix operator + (const float& scalar, const Matrix& m2);
Matrix operator + (const float& scalar, const Matrix& m2);
bool operator == (const Matrix& m1, const Matrix& m2);

class Data {
public:
	Data(float* input, float* labels,
		int numOfImages);
	Data(std::vector<DATA_ITEM> data);
	Data(const Data& rhs);
	~Data();
	std::vector<Data*> getBatches(int batchSize);
	std::vector<DATA_ITEM> getAll();
	int getNumOfImages();
private:
	std::vector<DATA_ITEM> m_data;
	int m_numOfImages;
};

class NeuralNetwork {
public:
	NeuralNetwork(std::vector<int> shapes);
	~NeuralNetwork();
	void stochasticGradientDescent(Data* trainData, Data* testData, int epochs, int batchSize, float eta);
	void train(Data* data, float eta);
	std::pair<std::vector<Matrix*>, std::vector<Matrix*>> backProb(DATA_ITEM data);
	void evaluate(Data* data);

private:
	std::vector<int> m_shapes;
	std::vector<Matrix*> m_weights, m_biases;
};

__device__ float sigmoid(float x);

__device__ float sigmoidPrime(float x);

__global__ void cudaDot(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaMul(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaAdd(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaSub(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaEqual(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaTranspose(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaSigmoid(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

__global__ void cudaSigmoidPrime(
	float* m1Values, float* m2Values, float* resValues,
	_2DShape* m1Shape, _2DShape* m2Shape, _2DShape* resShape);

Matrix sigmoid(Matrix* m1);
Matrix sigmoidPrime(Matrix* m1);

