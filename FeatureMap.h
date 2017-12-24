#pragma once

#include "NeuralMath.h"
#include "Layer.h"

#include <random>

class ConvolutionLayer;

class FeatureMap
{
public:
	FeatureMap(const int& kernelWidth, const int& kernelHeight, ConvolutionLayer* _parent);

	void	init();

	Matrix<float>  getKernel() const;
	Matrix<float>  getResult() const;

	~FeatureMap();
private:
	float			bias;
	Matrix<float>	kernel;
	Matrix<float>	result;

	ConvolutionLayer* parent;
};

