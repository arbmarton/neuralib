#pragma once

#include "NeuralMath.h"
#include "Layer.h"

#include "json.hpp"

#include <random>

class ConvolutionLayer;

// TODO: parent useless?
// TODO: get methods for the rotated kernel

class FeatureMap
{
public:
	FeatureMap(
		const int& kernelWidth,
		const int& kernelHeight,
		const int& resultWidth,
		const int& resultHeight,
		ConvolutionLayer* _parent);
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
	Matrix<float>	kernelRotated;
	Matrix<float>	result;  //maybe rename to activation?
	Matrix<float>	delta;
};

