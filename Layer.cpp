#include "stdafx.h"
#include "Layer.h"

#include <iostream>


////////////////////////////////////////////////////////////
///// LAYERBASE 
////////////////////////////////////////////////////////////


LayerBase::LayerBase(const LayerType& type, const int& newSize, LayerBase* _prev, LayerBase* _next)
	: layertype(type)
	, size(newSize)
//	, activations(Matrix<float>(newSize, 1))
	, prev(_prev)
	, next(_next)
{

}


LayerBase* LayerBase::getPreviousLayer() const
{
	return prev;
}

LayerBase* LayerBase::getNextLayer() const
{
	return next;
}

void LayerBase::setPreviousLayer(LayerBase* layer)
{
	prev = layer;
}

void LayerBase::setNextLayer(LayerBase* layer)
{
	next = layer;
}

Matrix<float> Layer::getActivations() const
{
	return activations;
}


////////////////////////////////////////////////////////////
///// LAYER
////////////////////////////////////////////////////////////


Layer::Layer(
	const int&        newSize,
	const NeuronType& newNeuronType,
	const LayerType&  newLayerType,
	LayerBase* _prev,
	LayerBase* _next)
	: LayerBase(newLayerType, newSize, _prev, _next)
	, neurontype(newNeuronType)
//	, layertype(newLayerType)
//	, weights(Matrix<float>(newSize, _prev ? static_cast<Layer*>(_prev)->activations.getRows() : newSize))
	, activations(Matrix<float>(newSize, 1))
	, biases(Matrix<float>(newSize, 1))
	, zed(Matrix<float>(newSize, 1))
	, delta(Matrix<float>(newSize, 1))
//	, costWeight(Matrix<float>(newSize, _prev ? static_cast<Layer*>(_prev)->activations.getRows() : newSize))
{
	neurons.resize(newSize);

	switch (newNeuronType)
	{

	case NeuronType::Sigmoid:
		for (int i = 0; i < newSize; ++i) {
			neurons[i] = new Sigmoid();
		}
		break;

	default:
		std::fill(neurons.begin(), neurons.end(), nullptr);
		break;
	}

	if (newLayerType != LayerType::Input) {
		init();
	}

	if (_prev) {
		auto prevLayer = dynamic_cast<PoolingLayer*>(_prev);
		if (prevLayer) {
			weights = Matrix<float>(newSize, prevLayer->getPoolRows());
			costWeight = Matrix<float>(newSize, prevLayer->getPoolRows());
			return;
		}

		auto prevLayer2 = dynamic_cast<ConvolutionLayer*>(_prev);
		if (prevLayer2) {
			weights = Matrix<float>(newSize, prevLayer2->getMapRows());
			costWeight = Matrix<float>(newSize, prevLayer2->getMapRows());
			return;
		}

		weights = Matrix<float>(newSize, static_cast<Layer*>(_prev)->activations.getRows());
		costWeight = Matrix<float>(newSize, static_cast<Layer*>(_prev)->activations.getRows());
	}
	else {
		weights = Matrix<float>(newSize, newSize);
		costWeight = Matrix<float>(newSize, newSize);
	}
}

Layer::Layer(const nlohmann::json& input)
	: LayerBase(jsonToLayerType(input), input["size"].get<int>())
	, weights(input["weights"].get<nlohmann::json>())
//	, activations(input["activations"].get<nlohmann::json>())
	, biases(input["biases"].get<nlohmann::json>())
	, zed(input["zed"].get<nlohmann::json>())
	, delta(input["delta"].get<nlohmann::json>())
	, costWeight(input["costweight"].get<nlohmann::json>())
{
	neurons.resize(size);

	switch (str2int(input["neurontype"].get<std::string>().c_str()))
	{
	case str2int("input"):
		neurontype = NeuronType::Input;

		for (int i = 0; i < size; ++i) {
			neurons[i] = new InputNeuron();
		}
		break;

	case str2int("sigmoid"):
		neurontype = NeuronType::Sigmoid;

		for (int i = 0; i < size; ++i) {
			neurons[i] = new Sigmoid();
		}
		break;

	case str2int("output"):
		neurontype = NeuronType::Output;

		for (int i = 0; i < size; ++i) {
			neurons[i] = new Sigmoid();
		}
		break;

	default:
		throw NeuralException("Cannot parse neurontype from json file...");
		//std::fill(neurons.begin(), neurons.end(), nullptr);
		break;
	}
}

int Layer::getSize() const
{
	return size;
}

std::vector<Neuron*> Layer::getNeurons() const
{
	return neurons;
}

Matrix<float> Layer::getBias() const
{
	return biases;
}

Matrix<float> Layer::getZed() const
{
	return zed;
}

Matrix<float> Layer::getDelta() const
{
	return delta;
}

Matrix<float> Layer::getCostBias() const
{
	return delta;
}

Matrix<float> Layer::getCostWeight() const
{
	return costWeight;
}

void Layer::calculateActivation()
{
	Matrix<float> temp = weights * (dynamic_cast<Layer*>(prev)->activations);
	//temp += biases;

	zed = temp + biases;

	for (int i = 0; i < activations.getRows(); ++i) {
		activations(i, 0) = sigmoid(-zed(i, 0));

		//neurons[i]->setResult(activations(i, 0));
	}
}

void Layer::calculateDelta()
{
	auto nextPtr = dynamic_cast<Layer*>(next);
	delta = hadamardProduct(
		transpose(nextPtr->weights)*nextPtr->delta,
		sigmoidDerivative(zed)
	);
}

void Layer::calculateCostWeight()
{
	auto prevPtr = dynamic_cast<Layer*>(prev);
	costWeight = delta * transpose(prevPtr->activations);
}

void Layer::update(
	const Regularization& regMethod,
	const Matrix<float>&  weightUpdate,
	const Matrix<float>&  biasUpdate,
	const float&		  multiplier,
	const float&		  regularization,
	const int&			  trainingSetSize)
{
	switch (regMethod)
	{
	case Regularization::L1:

		weights -= multiplier * weightUpdate - (regularization / trainingSetSize)*weights.signum();
		biases -= multiplier * biasUpdate;

		break;

	case Regularization::L2:

		weights -= multiplier * weightUpdate - (regularization / trainingSetSize)*weights;
		biases -= multiplier * biasUpdate;

		break;

	default:  // Regularization::None

		weights -= multiplier * weightUpdate;
		biases -= multiplier * biasUpdate;

		break;
	}
	
}

void Layer::printLayerInfo() const
{
	std::string type("General");
	std::string _neurontype;
	switch (neurontype)
	{

	case NeuronType::Sigmoid:
		_neurontype = "Sigmoid";
		break;

	default:
		_neurontype = "undefined neurontype";
		break;
	}
	std::string size = std::to_string(neurons.size());

	std::cout << "Layertype: " + type + "\nNeurontype: " + _neurontype + "\nLayersize: " + size << "\n\n";
}

void Layer::init()
{
	initWeights();
	initBiases();
}

void Layer::initWeights()
{
	//weights.fillGauss(0, 1);
	weights.fillGaussNormalized(0, 1, prev->getSize());
}

void Layer::initBiases()
{
	//biases.fillGauss(0, 1);
	biases.fillGaussNormalized(0, 1, prev->getSize());
}

nlohmann::json Layer::toJSON() const
{
	nlohmann::json ret;

	ret["size"] = size;

	switch (layertype)
	{
	case LayerType::Input:
		ret["layertype"] = "input";
		break;

	case LayerType::General:
		ret["layertype"] = "general";
		break;

	case LayerType::Output:
		ret["layertype"] = "output";
		break;

	default:
		ret["layertype"] = "unknown";
		break;
	}

	switch (neurontype)
	{
	case NeuronType::Input:
		ret["neurontype"] = "input";
		break;

	case NeuronType::Sigmoid:
		ret["neurontype"] = "sigmoid";
		break;

	case NeuronType::Output:
		ret["neurontype"] = "output";
		break;

	default:
		ret["neurontype"] = "unknown";
		break;
	}
	
	//?? neurons?

	ret["activations"] = activations.toJSON();
	ret["weights"]	   = weights.toJSON();
	ret["biases"]      = biases.toJSON();
	ret["zed"]		   = zed.toJSON();
	ret["delta"]	   = delta.toJSON();
	ret["costweight"]  = costWeight.toJSON();

	return ret;
}

Layer::~Layer()
{
	for (int i = 0; i < size; ++i) {
		delete neurons[i];
	}
}


////////////////////////////////////////////////////////////
///// CONVOLUTIONLAYER 
////////////////////////////////////////////////////////////


ConvolutionLayer::ConvolutionLayer(
	const int& newSize,
	const int& width,
	const int& height,
	LayerBase* _prev,
	LayerBase* _next)
	: LayerBase(LayerType::Convolutional, newSize, _prev, _next)
	, kernelWidth(width)
	, kernelHeight(height)
{
	featureMaps.resize(newSize);
	init();
}

ConvolutionLayer::ConvolutionLayer(const nlohmann::json& input)
	: LayerBase(LayerType::Convolutional, input["size"].get<int>())
	, kernelWidth(input["kernelWidth"].get<int>())
	, kernelHeight(input["kernelHeight"].get<int>())
	, resultWidth(input["resultWidth"].get<int>())
	, resultHeight(input["resultHeight"].get<int>())
{
	featureMaps.resize(input["size"].get<int>());
	std::vector<nlohmann::json> maps = input["maps"].get<std::vector<nlohmann::json>>();

	for (int i = 0; i < featureMaps.size(); ++i) {
		featureMaps[i] = new FeatureMap(maps[i]);
	}
}

void ConvolutionLayer::init()
{
	for (FeatureMap*& feat : featureMaps) {
		feat = new FeatureMap(kernelWidth, kernelHeight, this);
		feat->init();
	}
	/*for (int i = 0; i < featureMaps.size(); ++i) {
		featureMaps[i] = new FeatureMap(kernelWidth, kernelHeight, this);
		featureMaps[i]->init();
	}*/
}

int ConvolutionLayer::getSize() const
{
	return featureMaps.size();
}

int ConvolutionLayer::getMapRows() const
{
	return featureMaps.size() * featureMaps[0]->getResult().getRows();
}

std::vector<FeatureMap*>& ConvolutionLayer::getMaps()
{
	return featureMaps;
}

void ConvolutionLayer::calculateActivation()
{
	for (FeatureMap* feat : featureMaps) {
		convolve(prev->getActivations(), *feat);
	}
}

nlohmann::json ConvolutionLayer::toJSON() const
{
	nlohmann::json ret;

	ret["kernelWidth"]  = kernelWidth;
	ret["kernelHeight"] = kernelHeight;
	ret["resultWidth"]  = resultWidth;
	ret["resultHeight"] = resultHeight;

	std::vector<nlohmann::json> maps;
	maps.resize(featureMaps.size());

	for (int i = 0; i < maps.size(); ++i) {
		maps[i] = featureMaps[i]->toJSON();
	}

	ret["maps"] = maps;
	ret["size"] = size;

	return ret;
}

void ConvolutionLayer::printLayerInfo() const
{
	std::cout << "Printing ConvolutionLayer:\n";
	std::cout << "kernelwidth: " << kernelWidth << ", kernelheight: " << kernelHeight << '\n';
	std::cout << "resultwidth: " << resultWidth << ", resutheight: " << resultHeight << '\n';
	for (FeatureMap* map : featureMaps) {
		map->print();
	}
}

ConvolutionLayer::~ConvolutionLayer()
{
	for (int i = 0; i < size; ++i) {
		delete featureMaps[i];
	}
}

////////////////////////////////////////////////////////////
///// POOLINGLAYER
////////////////////////////////////////////////////////////


PoolingLayer::PoolingLayer(
	const int& newSize,
	const PoolingMethod& _method,
	const int& poolWidth,
	const int& poolHeight,
	ConvolutionLayer*   _prev,
	LayerBase*			_next
)
	: LayerBase(LayerType::Pooling, newSize, _prev, _next)
	, method(_method)
	, width(poolWidth)
	, height(poolHeight)
{
	pools.resize(newSize);
	init();
}

PoolingLayer::PoolingLayer(const nlohmann::json& input)
	: LayerBase(LayerType::Input, input["size"].get<int>())
	, method(jsonToPoolingMethod(input))
	, width(input["width"].get<int>())
	, height(input["height"].get<int>())
{
	pools.resize(input["size"].get<int>());
	std::vector<nlohmann::json> poolsInput = input["pools"].get<std::vector<nlohmann::json>>();

	for (int i = 0; i < pools.size(); ++i) {
		pools[i] = new Pool(poolsInput[i]);
	}
}

void PoolingLayer::init()
{
	for (Pool*& pool : pools) {
		pool = new Pool(method, width, height, this);
	}
	/*for (int i = 0; i < pools.size(); ++i) {
		pools[i] = new Pool(method, width, height, this);
	}*/
}

int PoolingLayer::getSize() const
{
	return pools.size();
}

int PoolingLayer::getPoolRows() const
{
	return pools.size() * pools[0]->getResult().getRows();
}

std::vector<Pool*>& PoolingLayer::getPools()
{
	return pools;
}

void PoolingLayer::calculateActivation()
{
	ConvolutionLayer* previous = static_cast<ConvolutionLayer*>(prev);

	for (int i = 0; i < pools.size(); ++i) {
		/*createPool(
			method, width, height,
			previous->getMaps()[i], 1, 1,
			pools[i].getResult, 1, 1);*/
	}
}

nlohmann::json PoolingLayer::toJSON() const
{
	nlohmann::json ret;

	switch (method)
	{
	case PoolingMethod::max:

		ret["method"] = "max";

		break;
	case PoolingMethod::L2:

		ret["method"] = "L2";

		break;
	default:
		throw NeuralException("\nUnknown poolingmethod encountered...\n");
		break;
	}

	ret["width"]  = width;
	ret["height"] = height;

	std::vector<nlohmann::json> poolsjson;
	poolsjson.resize(pools.size());
	for (int i = 0; i < pools.size(); ++i) {
		poolsjson[i] = pools[i]->toJSON();
	}

	ret["pools"] = poolsjson;

	return ret;
}

void PoolingLayer::printLayerInfo() const
{
	std::cout << "Printing PoolingLayer info:\n";
	std::cout << "width: " << width << ", height: " << height << '\n';
	for (Pool* pool : pools) {
		pool->print();
	}
}

PoolingLayer::~PoolingLayer()
{
	for (int i = 0; i < pools.size(); ++i) {
		delete pools[i];
	}
}


////////////////////////////////////////////////////////////
///// INPUTLAYER 
////////////////////////////////////////////////////////////


InputLayer::InputLayer(
	const int& newSize,
	const NeuronType& newNeuronType,
	const std::function<void(std::vector<Neuron*>&)>& func)
	: Layer(newSize, newNeuronType, LayerType::Input)
	, inputFunction(func)
{
	neurons.resize(newSize);
	for (int i = 0; i < neurons.size(); ++i) {
		neurons[i] = new InputNeuron();
	}
}

InputLayer::InputLayer(const nlohmann::json& input)
	: Layer(input)
{

}

void InputLayer::setInputFunction(const std::function<void(std::vector<Neuron*>&)>& func)
{
	inputFunction = func;
}

void InputLayer::calculateActivation()
{
	inputFunction(neurons);

	for (int i = 0; i < activations.getRows(); ++i) {
		activations(i, 0) = neurons[i]->getResult();
	}
}

void InputLayer::printLayerInfo() const
{
	std::string type("Input");
	std::string _neurontype("Input");
	std::string size = std::to_string(neurons.size());

	std::cout << "Layertype: " + type + "\nNeurontype: " + _neurontype + "\nLayersize: " +  size << "\n\n";
}

InputLayer::~InputLayer()
{

}


////////////////////////////////////////////////////////////
///// OUTPUTLAYER
////////////////////////////////////////////////////////////


OutputLayer::OutputLayer(
	const int& newSize,
	const NeuronType& newNeuronType,
	const CostFunction& costType,
	const std::function<void(std::vector<float>&)>& func,
	LayerBase* previous,
	LayerBase* _next)
	: Layer(newSize, newNeuronType, LayerType::Output, previous, _next)
	, idealOutputFunction(func)
	, costFunctionType(costType)
	, correct(0)
	, notCorrect(0)
{
	neurons.resize(newSize);
	for (int i = 0; i < neurons.size(); ++i) {
		neurons[i] = new Sigmoid();
	}

	init();
}

OutputLayer::OutputLayer(const nlohmann::json& input)
	: Layer(input)
{
	switch (str2int(input["costfunction"].get<std::string>().c_str()))
	{
	case str2int("leastsquares"):
		costFunctionType = CostFunction::LeastSquares;
		break;

	case str2int("crossentropy"):
		costFunctionType = CostFunction::CrossEntropy;
		break;

	default:
		throw NeuralException("Cant parse costfunction type...");
		break;
	}
}

void OutputLayer::setIdealOutput(const std::function<void(std::vector<float>&)>& func)
{
	idealOutputFunction = func;
}

// TODO: modify the output function to accept a Matrix<T> -> less intructions
Matrix<float> OutputLayer::getIdealOutput() const
{
	std::vector<float> vec;
	idealOutputFunction(vec);
	return Matrix<float>(vec);
}


void OutputLayer::calculateDelta()
{
	int index = getIdealOutput().largestIndex();
	if (index == activations.largestIndex())
		correct++;
	else
		notCorrect++;

	//Matrix<float> difference;
	switch (costFunctionType)
	{
	case CostFunction::LeastSquares:
		//difference = activations - getIdealOutput();

		delta = hadamardProduct(
			sigmoidDerivative(zed),
			activations - getIdealOutput()
		);
		break;

	case CostFunction::CrossEntropy:
		delta = activations - getIdealOutput();
		break;

	default:
		throw NeuralException("Invalid costfunction type...");
		break;
	}

}

void OutputLayer::initBiases()
{
	biases = Matrix<float>(biases.getRows(), biases.getCols());
}

void OutputLayer::printLayerInfo() const
{
	std::string type("Output");
	std::string _neurontype;
	switch (neurontype)
	{

	case NeuronType::Sigmoid:
		_neurontype = "Sigmoid";
		break;

	default:
		_neurontype = "undefined neurontype";
		break;
	}
	std::string size = std::to_string(neurons.size());

	std::cout << "Layertype: " + type + "\nNeurontype: " + _neurontype + "\nLayersize: " + size << "\n\n";
}

nlohmann::json OutputLayer::toJSON() const
{
	nlohmann::json ret;

	ret["size"] = size;

	switch (layertype)
	{
	case LayerType::Input:
		ret["layertype"] = "input";
		break;

	case LayerType::General:
		ret["layertype"] = "general";
		break;

	case LayerType::Output:
		ret["layertype"] = "output";
		break;

	default:
		ret["layertype"] = "unknown";
		break;
	}

	switch (neurontype)
	{
	case NeuronType::Input:
		ret["neurontype"] = "input";
		break;

	case NeuronType::Sigmoid:
		ret["neurontype"] = "sigmoid";
		break;

	case NeuronType::Output:
		ret["neurontype"] = "output";
		break;

	default:
		ret["neurontype"] = "unknown";
		break;
	}

	switch (costFunctionType)
	{
	case CostFunction::LeastSquares:
		ret["costfunction"] = "leastsquares";
		break;

	case CostFunction::CrossEntropy:
		ret["costfunction"] = "crossentropy";
		break;

	default:
		ret["costfunction"] = "unknown";
		break;
	}

	//?? neurons?

	ret["activations"] = activations.toJSON();
	ret["weights"]     = weights.toJSON();
	ret["biases"]	   = biases.toJSON();
	ret["zed"]		   = zed.toJSON();
	ret["delta"]	   = delta.toJSON();
	ret["costweight"]  = costWeight.toJSON();

	return ret;
}

OutputLayer::~OutputLayer()
{

}
