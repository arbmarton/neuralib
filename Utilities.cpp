#include "stdafx.h"
#include "Utilities.h"


PoolingMethod jsonToPoolingMethod(const nlohmann::json& input)
{
	switch (str2int(input["method"].get<std::string>().c_str()))
	{
	case str2int("max"):
		return PoolingMethod::max;

	case str2int("L2"):
		return PoolingMethod::L2;

	case str2int("average"):
		return PoolingMethod::average;

	default:
		throw NeuralException("\nCant parse poolingmethod...\n");
		break;
	}
}

LayerType jsonToLayerType(const nlohmann::json& input)
{
	switch (str2int(input["layertype"].get<std::string>().c_str()))
	{
	case str2int("input"):
		return LayerType::Input;

	case str2int("general"):
		return LayerType::General;

	case str2int("output"):
		return LayerType::Output;

	case str2int("pooling"):
		return LayerType::Pooling;

	case str2int("convolutional"):
		return LayerType::Convolutional;

	default:
		throw NeuralException("\nUnknown layertype encountered...\n");
		break;
	}
}

std::string layerTypeToString(const LayerType& layertype)
{
	switch (layertype)
	{
	case LayerType::Convolutional:
		return std::string("convolutional");
		break;

	case LayerType::General:
		return std::string("general");
		break;

	case LayerType::Input:
		return std::string("input");
		break;

	case LayerType::Output:
		return std::string("output");
		break;

	case LayerType::Pooling:
		return std::string("pooling");
		break;

	default:
		throw NeuralException("undefined layertype");
		break;
	}
}

std::string neuronTypeToString(const NeuronType& neurontype)
{
	switch (neurontype)
	{
	case NeuronType::Input:
		return "input";
		break;

	case NeuronType::Output:
		return "output";
		break;

	case NeuronType::Sigmoid:
		return "sigmoid";
		break;

	case NeuronType::Softmax:
		return "softmax";
		break;

	default:
		throw NeuralException("undefined neurontype");
		break;
	}
}

std::string poolingMethodToString(const PoolingMethod& poolingmethod)
{
	switch (poolingmethod)
	{
	case PoolingMethod::average:
		return "average";

	case PoolingMethod::L2:
		return "L2";

	case PoolingMethod::max:
		return "max";

	default:
		throw NeuralException("unknown poolingmethod encountered");
		break;
	}
}

std::string costfunctionTypeToString(const CostFunction& costfunctiontype)
{
	switch (costfunctiontype)
	{
	case CostFunction::LeastSquares:
		return "leastsquares";

	case CostFunction::CrossEntropy:
		return "crossentropy";

	default:
		throw NeuralException("unknown costfunctiontype");
	}
}