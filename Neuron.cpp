#include "stdafx.h"
#include "Neuron.h"



Neuron::Neuron()
	: result(INF)		// initialize with bogus value for easy error checking
	, bias(0.0f)
{
}

float Neuron::getResult() const
{
	return result;
}

void Neuron::setResult(const float& newResult)
{
	result = newResult;
}

/*std::vector<std::pair<Neuron*, float>> Neuron::getConnections() const
{
	return connections;
}*/

/*void Neuron::setParents(const std::vector<Neuron*>& parents)
{
	if (connections.size() == 0) {
		connections.resize(parents.size());

		for (int i = 0; i < connections.size(); ++i) {
			connections[i] = std::make_pair(parents[i], 0.0f);
		}
	}
	else {
		if (connections.size() == parents.size()) {

			for (int i = 0; i < connections.size(); ++i) {
				connections[i].first = parents[i];
			}

		}
		else {
			connections.erase(connections.begin(), connections.end());
			connections.resize(parents.size());

			for (int i = 0; i < connections.size(); ++i) {
				connections[i] = std::make_pair(parents[i], 0.0f);
			}
		}
	}
}*/

/*void Neuron::setWeights(const std::vector<float>& weights)
{
	if (connections.size() == 0) {
		connections.resize(weights.size());

		for (int i = 0; i < connections.size(); ++i) {
			connections[i] = std::make_pair(nullptr, weights[i]);
		}
	}
	else {
		if (connections.size() == weights.size()) {
			
			for (int i = 0; i < connections.size(); ++i) {
				connections[i].second = weights[i];
			}
		}
		else {
			connections.erase(connections.begin(), connections.end());
			connections.resize(weights.size());

			for (int i = 0; i < connections.size(); ++i) {
				connections[i] = std::make_pair(nullptr, weights[i]);
			}
		}
	}
}*/

/*void Neuron::setConnections(const std::vector<std::pair<Neuron*, float>>& newConnections)
{
	connections = newConnections;
}*/

/*void Neuron::initializeWeights()
{
	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0, 1);

	for (auto& pair : connections) {
		pair.second = dist(rng);
	}
}*/

Neuron::~Neuron()
{
}



Sigmoid::Sigmoid()
{
	
}

void Sigmoid::calculateResult()
{
	/*float accum = bias;
	for (int i = 0; i < connections.size(); ++i) {
		accum += connections[i].first->getResult() * connections[i].second;  // dot product
	}
	result = 1 / (1 + exp(-1*accum));*/
}



Sigmoid::~Sigmoid()
{

}


InputNeuron::InputNeuron()
{

}

void InputNeuron::setResult(const float& newResult)
{
	result = newResult;
}

InputNeuron::~InputNeuron()
{

}