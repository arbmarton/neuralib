#pragma once

#include "Layer.h"
#include "NeuralMath.h"

#include "json.hpp"

#include <string>

enum class PoolingMethod;
enum class LayerType;
enum class CostFunction;

PoolingMethod jsonToPoolingMethod(const nlohmann::json& input);
LayerType	  jsonToLayerType(const nlohmann::json& input);

std::string	  layerTypeToString(const LayerType& layertype);
std::string   neuronTypeToString(const NeuronType& neurontype);
std::string	  poolingMethodToString(const PoolingMethod& poolingmethod);
std::string   costfunctionTypeToString(const CostFunction& costfunctiontype);