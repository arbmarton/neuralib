#pragma once

#include "Layer.h"
#include "NeuralMath.h"

#include "json.hpp"

enum class PoolingMethod;
enum class LayerType;

PoolingMethod jsonToPoolingMethod(const nlohmann::json& input);
LayerType	  jsonToLayerType(const nlohmann::json& input);