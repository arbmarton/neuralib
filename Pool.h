#pragma once

#include "NeuralMath.h"
#include "Layer.h"

#include "json.hpp"

#include <vector>

enum class PoolingMethod;

class PoolingLayer;
class LayerBase;

class Pool
{
public:
	Pool(const PoolingMethod& _method, const int& kernelWidth, const int& kernelHeight, const PoolingLayer* _parent);
	Pool(const nlohmann::json& input);

	int			   getWidth()  const;
	int			   getHeight() const;
	Matrix<float>& getResult();		// returns a reference because of the way the pooling is done
	std::vector<std::pair<int, int>>* getErrorLocations();

	void calculateDelta(const LayerBase* next, const Matrix<float>& bigDelta, const int& curr);

	nlohmann::json toJSON()   const;

	void print() const;

	~Pool();
private:
	PoolingMethod method;
	int width;
	int height;

	Matrix<float> result;
	Matrix<float> delta;
	Matrix<float> costWeight;
	std::vector<std::pair<int, int>> errorLocations;	// where to send the error back in the previous layer, matrix coordinates
};

