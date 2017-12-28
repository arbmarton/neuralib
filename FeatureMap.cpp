#include "stdafx.h"
#include "FeatureMap.h"


FeatureMap::FeatureMap(const int& kernelWidth, const int& kernelHeight, ConvolutionLayer* _parent)
	: kernel(Matrix<float>(kernelHeight, kernelWidth))
	, parent(_parent)
{
	const int prevActivations = _parent->getPreviousLayer()->getActivations().getRows();
	const int inputSide = sqrt(prevActivations);
	int resultSide = inputSide - 2 * floor(kernelWidth / 2);
	
	if (kernelWidth % 2 == 0) {
		resultSide += 1;
	}

	result = Matrix<float>(resultSide, resultSide);
}

FeatureMap::FeatureMap(const nlohmann::json& other)
	: bias(other["bias"].get<float>())
	, kernel(other["kernel"].get<nlohmann::json>())
	, result(other["result"].get<nlohmann::json>())
{
}

void FeatureMap::init()
{
	std::uniform_real_distribution<float> uni(-1, 1);

	bias = uni(Random::getMT());

	kernel.fillGauss(0, 1);
}

Matrix<float> FeatureMap::getKernel() const
{
	return kernel;
}

Matrix<float> FeatureMap::getResult() const
{
	return result;
}

nlohmann::json FeatureMap::toJSON()   const
{
	nlohmann::json ret;

	ret["bias"]   = bias;
	ret["kernel"] = kernel.toJSON();
	ret["result"] = result.toJSON();

	return ret;
}

void FeatureMap::print() const
{
	std::cout << "Printing FeatureMap:\n";
	std::cout << "bias: " << bias << '\n';
	std::cout << "kernel:\n";
	kernel.print();
	std::cout << "result:\n";
	result.print();
}

FeatureMap::~FeatureMap()
{
}
