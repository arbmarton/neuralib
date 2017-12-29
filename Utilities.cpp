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