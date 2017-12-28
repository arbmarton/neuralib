#pragma once

#include "NeuralMath.h"
#include "Layer.h"

#include "json.hpp"

#include <random>

class ConvolutionLayer;

//TODO: parent useless?

class FeatureMap
{
public:
	FeatureMap(const int& kernelWidth, const int& kernelHeight, ConvolutionLayer* _parent);
	FeatureMap(const nlohmann::json& other);

	void	init();

	Matrix<float>  getKernel() const;
	Matrix<float>  getResult() const;

	nlohmann::json toJSON()	   const;

	void print() const;

	~FeatureMap();
private:
	float			bias;
	Matrix<float>	kernel;
	Matrix<float>	result;

	ConvolutionLayer* parent;
};

