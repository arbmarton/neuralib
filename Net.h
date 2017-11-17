#pragma once

#include "Layer.h"
#include "Neuron.h"
#include "NeuralException.h"
#include "NeuralInputClass.h"

#include "json.hpp"

#include <iostream>
#include <assert.h>
#include <thread>

//when constructing a net from json, the layers should be connected manually


class Net
{
public:
	Net(const int&        epochNumber,
		const int&		  minibatchSizeParam,
		const float&	  newEta,
		NeuralInputClass* inputMaker);

	Net(const nlohmann::json& input);

	void createNewLayer(
		const int& size,
		const NeuronType& neuronType,
		const LayerType&  layerType
	);

	Layer* getLayer(const int& layerNumber) const;
	Layer* getLayer(const LayerType& layerType) const;
	Layer* getLastLayer() const;

	void addInputClass(NeuralInputClass* inputclass);

	void calculate();
	void calculateActivationInAllLayers() const;
	void calculateDeltaInAllLayers() const;
	void calculateDerivativesInAllLayers() const;  // invoke after delta calculation only
	void addUpWeightsAndBiases(
		std::vector<Matrix<float>>& weights,
		std::vector<Matrix<float>>& biases) const;
	void updateWeightsAndBiases(
		const std::vector<Matrix<float>>& weights,
		const std::vector<Matrix<float>>& biases,
		const float& multiplier) const;

	void printLayer(const int& layerNumber) const;
	void printOutputLayer() const;
	void printLayerInfo() const;

	nlohmann::json toJSON() const;

	~Net();
private:
	int   epochs;
	int	  minibatchSize;
	float eta;

	NeuralInputClass* inputClass;

	std::vector<Layer*> layers;

	void connectLayers();

	unsigned int threads;
};

