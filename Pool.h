#pragma once

#include "NeuralMath.h"
#include "Layer.h"

enum class PoolingMethod;

class PoolingLayer;

class Pool
{
public:
	Pool(const PoolingMethod& _method, const int& kernelWidth, const int& kernelHeight, const PoolingLayer* _parent);

	int			  getWidth()  const;
	int			  getHeight() const;
	Matrix<float> getResult() const;

	~Pool();
private:
	PoolingMethod method;
	int width;
	int height;

	Matrix<float> result;

	PoolingLayer* parent;
};

