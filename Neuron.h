#pragma once

#include "NeuralException.h"

#include <vector>
#include <math.h>
#include <functional>
#include <random>


#define INF std::numeric_limits<float>::infinity() // less typing, more FUN


enum class NeuronType
{
	Input,
	Sigmoid,
	Softmax,
	/* ... */
	Output
};



class Neuron
{
public:
	Neuron();

	virtual void calculateResult() = 0;

	virtual float getResult() const;
	virtual void setResult(const float& newResult);

	//virtual std::vector<std::pair<Neuron*, float>> getConnections() const;

	//virtual void setParents    (const std::vector<Neuron*>& parents);
	//virtual void setWeights	   (const std::vector<float>& weights);
	//virtual void setConnections(const std::vector<std::pair<Neuron*, float>>& newConnections);

	//virtual void initializeWeights();

	virtual ~Neuron() = 0;
protected:
	/*std::vector<
		std::pair<Neuron*, float>>  connections; //the neurons in the previous layer*/
	float							result;
	float							bias;
};


class Sigmoid : public Neuron
{
public:
	Sigmoid();

	void calculateResult();

	~Sigmoid();
private:

};


class InputNeuron : public Neuron
{
public:
	InputNeuron();

	void setResult(const float& newResult);

	~InputNeuron();
private:
	/*std::vector<std::pair<Neuron*, float>> getConnections() const override  // input neurons should not use this, therefore hidden
			{ return std::vector<std::pair<Neuron*, float>>(); } */

	void calculateResult() {}
};


class OutputNeuron : public Neuron
{

};

