#pragma once

#include "NeuralMath.h"
#include "Layer.h"

#include "json.hpp"

enum class PoolingMethod;

class PoolingLayer;

class Pool
{
public:
	Pool(const PoolingMethod& _method, const int& kernelWidth, const int& kernelHeight, const PoolingLayer* _parent);
	Pool(const nlohmann::json& input);

	int			  getWidth()  const;
	int			  getHeight() const;
	Matrix<float> getResult() const;

	nlohmann::json toJSON()   const;

	~Pool();
private:
	PoolingMethod method;
	int width;
	int height;

	Matrix<float> result;

	PoolingLayer* parent;
};

