#pragma once

#include "Neuron.h"
#include "NeuralException.h"
#include "Matrix.h"
#include "NeuralMath.h"
#include "FeatureMap.h"

#include "json.hpp"

#include <vector>
#include <random>


enum class LayerType
{
	Input,
	General,
	Convolutional,
	Output
};


////////////////////////////////////////////////////////////
///// LAYERBASE 
////////////////////////////////////////////////////////////


class LayerBase {
public:
	LayerBase(const int& newSize, LayerBase* _prev = nullptr, LayerBase* _next = nullptr)
		: size(newSize)
		, prev(_prev)
		, next(_next)
	{
	}

	virtual int getSize() const = 0;

	virtual LayerBase* getPreviousLayer() const;
	virtual LayerBase* getNextLayer() const;
	virtual void setPreviousLayer(LayerBase* layer);
	virtual void setNextLayer(LayerBase* layer);

	virtual void printLayerInfo() const = 0;

	virtual nlohmann::json toJSON() const = 0;

	virtual ~LayerBase() {};

protected:
	int			size;
	LayerBase*	prev;
	LayerBase*	next;
};


////////////////////////////////////////////////////////////
///// LAYER
////////////////////////////////////////////////////////////


class Layer : public LayerBase
{
public:
	Layer(
		const int& newSize,
		const NeuronType& newNeuronType,
		const LayerType& newLayerType,
		LayerBase* _prev    = nullptr,
		LayerBase* _next    = nullptr
	);

	Layer(const nlohmann::json& input);

	virtual int getSize() const override;
	virtual std::vector<Neuron*> getNeurons() const;
	//virtual Neuron& getNeuron(const int& neuronNumber) const;  // cant return copy: abstract class  // maybe a pointer should be used? 

	//virtual LayerBase* getPreviousLayer() const override;
	//virtual LayerBase* getNextLayer() const override;

	//virtual void setPreviousLayer(LayerBase* layer) override;
	//virtual void setNextLayer(LayerBase* layer) override;

	virtual Matrix<float> getActivations() const;
	virtual Matrix<float> getBias()		   const;
	virtual Matrix<float> getZed()		   const;
	virtual Matrix<float> getDelta()	   const;
	virtual Matrix<float> getCostBias()	   const;  // returns delta
	virtual Matrix<float> getCostWeight()  const;

	virtual void calculateActivation();
	virtual void calculateDelta();
	virtual void calculateCostWeight();
	virtual void update(
		const Regularization& regMethod,
		const Matrix<float>&  weightUpdate,
		const Matrix<float>&  biasUpdate,
		const float&		  multiplier,
		const float&		  regularization,
		const int&			  trainingSetSize
	);

	virtual void printLayerInfo() const override;

	virtual nlohmann::json toJSON() const override;

	virtual ~Layer();

protected:
	//int						size;
	NeuronType				neurontype;
	LayerType				layertype;

	//Layer*					prev;
	//Layer*					next;

	std::vector<Neuron*>	neurons;

	Matrix<float>			activations;
	Matrix<float>			weights;
	Matrix<float>			biases;
	Matrix<float>			zed;
	Matrix<float>			delta;
	Matrix<float>			costWeight;  // the weight matrix delta basically

	virtual void initWeights();
	virtual void initBiases();
};


////////////////////////////////////////////////////////////
///// CONVOLUTIONLAYER 
////////////////////////////////////////////////////////////


class ConvolutionLayer : public LayerBase {
public:
	ConvolutionLayer(
		const int& newSize,
		LayerBase* _prev = nullptr,
		LayerBase* _next = nullptr
	);

	int getSize() const override;

	nlohmann::json toJSON() const override { return nlohmann::json(); };
	void printLayerInfo() const override {};
private:
	std::vector<FeatureMap*> featureMaps;
};


////////////////////////////////////////////////////////////
///// INPUTLAYER 
////////////////////////////////////////////////////////////


class InputLayer : public Layer
{
public:
	InputLayer(
		const int& newSize,
		const NeuronType& newNeuronType,
		const std::function<void(std::vector<Neuron*>&)>& func
	);

	InputLayer(const nlohmann::json& input);

	void setInputFunction(const std::function<void(std::vector<Neuron*>&)>& func);

	void calculateActivation() override;
	void calculateDelta()      override {};
	void calculateCostWeight() override {};
	void update(
		const Regularization& regMethod,
		const Matrix<float>&  weightUpdate,
		const Matrix<float>&  biasUpdate,
		const float&		  multiplier,
		const float&		  regularization,
		const int&			  trainingSetSize) override {};

	void printLayerInfo() const override;

	~InputLayer();
private:
	std::function<void(std::vector<Neuron*>&)> inputFunction;
};


////////////////////////////////////////////////////////////
///// OUTPUTLAYER
////////////////////////////////////////////////////////////


//update json
class OutputLayer : public Layer
{
public:
	OutputLayer(
		const int& newSize,
		const NeuronType& newNeuronType,
		const CostFunction& costType,
		const std::function<void(std::vector<float>&)>& func,
		LayerBase* previous = nullptr,
		LayerBase* _next    = nullptr
	);

	OutputLayer(const nlohmann::json& input);

	void setIdealOutput(const std::function<void(std::vector<float>&)>& func);
	Matrix<float> getIdealOutput() const;

	void calculateDelta() override;

	void resetCounters() { correct = 0; notCorrect = 0; }
	float getRatio() const { return float(correct) / float(notCorrect + correct); }

	void printLayerInfo() const override;

	nlohmann::json toJSON() const override;

	~OutputLayer();
private:
	CostFunction costFunctionType;
	std::function<void(std::vector<float>&)> idealOutputFunction;

	int correct;
	int notCorrect;

	void initBiases() override;
};

