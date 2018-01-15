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
{
	switch (str2int(input["method"].get<std::string>().c_str()))
	{
	case str2int("L2"):
		method = PoolingMethod::L2;
		break;

	case str2int("max"):
		method = PoolingMethod::max;
		break;

	default:
		throw NeuralException("\nUnknown poolingmethod encountered....\n");
		break;
	}
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

	ret["width"]  = width;
	ret["height"] = height;
	ret["result"] = result.toJSON();

	switch (method) {
	case PoolingMethod::L2:

		ret["method"] = "L2";

		break;

	case PoolingMethod::max:

		ret["method"] = "max";

		break;

	default:
		throw NeuralException("\nUnknown pooling method encountered...\n");
		break;
	}

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
