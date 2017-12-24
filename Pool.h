#pragma once

#include "NeuralMath.h"

class Pool
{
public:
	Pool();

	void init() {};

	~Pool();
private:
	PoolingMethod method;

	Matrix<float> result;
};

