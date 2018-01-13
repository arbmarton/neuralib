#pragma once

#include "Neuron.h"
#include "NeuralException.h"
#include "Matrix.h"
#include "NeuralMath.h"
#include "FeatureMap.h"
#include "Pool.h"
#include "Utilities.h"

#include "json.hpp"

#include <vector>
#include <random>

// TODO: complete the feedforward cycle and test it

// TODO: errorlocations in the pooling layer
// TODO: thourough JSON test, io with all the functions
// TODO: implement softmax layer
// TODO: feature map output: H-kernelheight+1? correct?
// TODO: maybe fullconvolution should be void?

class FeatureMap;
class Pool;
class Neuron;
enum class CostFunction;
enum class Regularization;
enum class PoolingMethod;
enum class LayerType;
enum class NeuronType;

enum class LayerType
{
	Input,
	General,
	Convolutional,
	Pooling,
	Output
};




////////////////////////////////////////////////////////////
///// LAYERBASE 
////////////////////////////////////////////////////////////

// best place for activations?
class LayerBase {
public:
	LayerBase(const LayerType& type, const int& newSize, LayerBase* _prev = nullptr, LayerBase* _next = nullptr);

	virtual int getSize() const = 0;

	virtual LayerBase* getPreviousLayer() const;
	virtual LayerBase* getNextLayer() const;
	virtual void setPreviousLayer(LayerBase* layer);
	virtual void setNextLayer(LayerBase* layer);

	virtual Matrix<float> getActivations() const = 0;

	virtual void calculateActivation() = 0;
	virtual void calculateDelta() = 0;

	virtual void printLayerInfo() const = 0;
	virtual void printLayer() const = 0;

	virtual nlohmann::json toJSON() const = 0;

	virtual ~LayerBase() {};

protected:
	LayerType		layertype;
	int				size;

	LayerBase*		prev;
	LayerBase*		next;

//	Matrix<float>	activations;

	virtual void init() = 0;
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
		const LayerType&  newLayerType,
		LayerBase* _prev    = nullptr,
		LayerBase* _next    = nullptr
	);

	Layer(const nlohmann::json& input);

	virtual int getSize() const override;
	virtual std::vector<Neuron*> getNeurons() const;
	//virtual Neuron& getNeuron(const int& neuronNumber) const;  // cant return copy: abstract class  // maybe a pointer should be used? 

	virtual Matrix<float> getActivations() const override;
	virtual Matrix<float> getBias()		   const;
	virtual Matrix<float> getZed()		   const;
	virtual Matrix<float> getDelta()	   const;
	virtual Matrix<float> getCostBias()	   const;  // returns delta
	virtual Matrix<float> getCostWeight()  const;

	virtual void calculateActivation() override;
	virtual void calculateDelta() override;
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
	virtual void printLayer() const override;

	virtual nlohmann::json toJSON() const override;

	virtual ~Layer();

protected:
	NeuronType				neurontype;
//	LayerType				layertype;

	std::vector<Neuron*>	neurons;

	Matrix<float>			activations;
	Matrix<float>			weights;
	Matrix<float>			biases;
	Matrix<float>			zed;
	Matrix<float>			delta;
	Matrix<float>			costWeight;  // the weight matrix delta basically

	virtual void init() override;
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
		const int& width,
		const int& height,
		LayerBase* _prev,
		LayerBase* _next = nullptr
	);
	ConvolutionLayer(const nlohmann::json& input);

	virtual void init() override;

	virtual int getSize() const override;
	int getMapRows() const;
	std::vector<FeatureMap*>& getMaps();

	virtual void calculateActivation() override;
	virtual void calculateDelta() override;

	virtual Matrix<float> getActivations() const override { return Matrix<float>(); }

	nlohmann::json toJSON() const override;

	void printLayerInfo() const override;
	void printLayer() const override;

	virtual ~ConvolutionLayer();
private:
	Matrix<float> input;

	std::vector<FeatureMap*> featureMaps;
	int kernelWidth;
	int kernelHeight;
	int resultWidth;
	int resultHeight;
};

////////////////////////////////////////////////////////////
///// PoolingLayer
////////////////////////////////////////////////////////////


class PoolingLayer : public LayerBase
{
public:
	PoolingLayer(
		const int& newSize,
		const PoolingMethod& _method,
		const int& poolWidth,
		const int& poolHeight,
		ConvolutionLayer*	 _prev = nullptr,
		LayerBase*			 _next = nullptr
	);
	PoolingLayer(const nlohmann::json& input);

	virtual void init() override;

	virtual int getSize() const override;
	int getPoolRows() const;
	std::vector<Pool*>& getPools();

	virtual void calculateActivation() override;
	virtual void calculateDelta() override;

	virtual Matrix<float> getActivations() const override;

	nlohmann::json toJSON() const override;

	void printLayerInfo() const override;
	void printLayer() const override;

	virtual ~PoolingLayer();
private:
	PoolingMethod method;
	int width;
	int height;

	std::vector<Pool*> pools;
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

	void calculateActivation() override;
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

