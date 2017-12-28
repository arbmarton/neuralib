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

	result = Matrix<float>(resultWidth, resultWidth);
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

Matrix<float> Pool::getResult() const
{
	return result;
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

Pool::~Pool()
{
}
