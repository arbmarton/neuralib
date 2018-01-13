#pragma once

#include "Layer.h"
#include "Neuron.h"
#include "NeuralException.h"
#include "NeuralInputClass.h"
#include "NeuralMath.h"

#include "json.hpp"

#include <iostream>
#include <assert.h>
#include <thread>
#include <fstream>

//when constructing a net from json, the layers should be connected "manually"


class Net
{
public:
	Net(NeuralInputClass* inputMaker,
		const CostFunction& cost  = CostFunction::LeastSquares,
		const Regularization& reg = Regularization::None);
	Net(const nlohmann::json& input);

	void createNewLayer(
		const int& size,
		const NeuronType& neuronType,
		const LayerType&  layerType
	);
	void createNewLayer(
		const int& size,
		const LayerType& layertype,
		const int& width,
		const int& heights
	);

	LayerBase* getLayer(const int& layerNumber) const;
	LayerBase* getLayer(const LayerType& layerType) const;
	LayerBase* getLastLayer() const;

	void addInputClass(NeuralInputClass* inputclass);

	void testForward();
	void train(
		const int&        epochNumber,
		const int&        minibatchSizeParam,
		const float&      newEta,
		const float&	  regularizationParam = 0);
	void work();

	void calculateActivationInAllLayers() const;
	void calculateDeltaInAllLayers() const;
	void calculateDerivativesInAllLayers() const;  // invoke after delta calculation only
	void addUpWeightsAndBiases(
		std::vector<Matrix<float>>& weights,
		std::vector<Matrix<float>>& biases) const;
	void updateWeightsAndBiases(
		const std::vector<Matrix<float>>& weights,
		const std::vector<Matrix<float>>& biases,
		const float&					  multiplier,
		const float&					  regularizationParam,
		const int&						  trainingSetSize) const;

	void printLayer(const int& layerNumber) const;
	void printOutputLayer() const;
	void printLayerInfo() const;

	nlohmann::json toJSON() const;
	void saveAs(const char* filename) const;

	~Net();
private:
	int   epochs;
	int	  minibatchSize;
	float eta;
	float regularization;

	NeuralInputClass* inputClass;
	CostFunction	  costFunctionType;
	Regularization	  regularizationType;

	std::vector<LayerBase*> layers;

	void connectLayers();
};

// less typing when constructing a Net from a json
inline nlohmann::json getJson(const char* filename)
{
	std::ifstream i(filename);
	nlohmann::json readjson;
	i >> readjson;
	return readjson;
}