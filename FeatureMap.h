#pragma once

#include "NeuralMath.h"
#include "Layer.h"

#include "json.hpp"

#include <random>

class LayerBase;
class ConvolutionLayer;

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

	Matrix<float>&  getKernel();		// returns a reference because of the way the convolving is done
	Matrix<float>&  getResult();		// returns a reference because of the way the convolving is done
	Matrix<float>	getDelta() const;
	Matrix<float>	getCostWeight() const;
	float			getBias() const;

	void applyBias();
	void addToBias(const float& val);
	void addToKernel(const Matrix<float>& mat);

	void calculateDelta(LayerBase* next, const int& curr);
	void calculateCostWeight(LayerBase* prev);

	nlohmann::json toJSON()	const;

	void print() const;

	~FeatureMap();
private:
	float			bias;
	Matrix<float>	kernel;
	Matrix<float>	kernelRotated;
	Matrix<float>	result;  // maybe rename to activation?
	Matrix<float>	delta;
	Matrix<float>   costWeight;
};

