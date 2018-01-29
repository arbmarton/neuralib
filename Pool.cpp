#include "stdafx.h"
#include "Pool.h"


Pool::Pool(const PoolingMethod& _method, const int& kernelWidth, const int& kernelHeight, const PoolingLayer* _parent)
	: method(_method)
	, width(kernelWidth)
	, height(kernelHeight)
{
	ConvolutionLayer* prev = static_cast<ConvolutionLayer*>(_parent->getPreviousLayer());
	const int previousWidth = prev->getMaps()[0]->getResult().getCols();
	if (previousWidth % kernelWidth != 0)
	{
		throw NeuralException("\nKernelsize mismatch in Pooling layer.\n");
	}
	const int resultWidth = previousWidth / kernelWidth;

	result		= Matrix<float>(resultWidth, resultWidth);
	delta		= result;
	costWeight  = result;

	errorLocations.resize(result.getSize());
}

Pool::Pool(const nlohmann::json& input)
	: width(input["width"].get<int>())
	, height(input["height"].get<int>())
	, result(input["result"].get<nlohmann::json>())
	, delta(input["delta"].get<nlohmann::json>())
	, costWeight(input["costweight"].get<nlohmann::json>())
{
	method = jsonToPoolingMethod(input);

	errorLocations.resize(result.getSize());
}

int Pool::getWidth() const
{
	return width;
}

int Pool::getHeight() const
{
	return height;
}

Matrix<float>& Pool::getResult()
{
	return result;
}

Matrix<float> Pool::getDelta() const
{
	return delta;
}

std::vector<std::pair<int, int>>* Pool::getErrorLocations()
{
	return &errorLocations;
}

// curr says to this object which Pool is it in the vector
// we need to get the delta of this Pools from the delta of all the pools
void Pool::calculateDelta(const LayerBase* next, const Matrix<float>& bigDelta, const int& curr)
{
	for (int i = 0; i < delta.getRows(); ++i) {
		for (int j = 0; j < delta.getCols(); ++j) {
			delta(i, j) = bigDelta(curr*delta.getSize() + i*delta.getCols() + j, 0);
		}
	}
}

nlohmann::json Pool::toJSON() const
{
	nlohmann::json ret;

	ret["method"]		= poolingMethodToString(method);
	ret["width"]		= width;
	ret["height"]		= height;
	ret["result"]		= result.toJSON();
	ret["delta"]		= delta.toJSON();
	ret["costweight"]   = costWeight.toJSON();

	return ret;
}

void Pool::print() const
{
	std::cout << "Pool object:\n";
	std::cout << "width: " << width << ", height: " << height << '\n';
	result.print();
}

Pool::~Pool()
{
}
