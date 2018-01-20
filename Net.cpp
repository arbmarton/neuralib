#include "stdafx.h"
#include "Net.h"


Net::Net(NeuralInputClass* input, const CostFunction& cost, const Regularization& reg)
	: inputClass(input)
	, costFunctionType(cost)
	, regularizationType(reg)
	, epochs(0)
	, minibatchSize(0)
	, eta(0)
	, regularization(0)
{
	input->init();
}

// when constructing a net from json, the inputclass should be added manually
Net::Net(const nlohmann::json& input)
	: epochs(input["epochs"].get<int>())
	, minibatchSize(input["minibatchsize"].get<int>())
	, eta(input["eta"].get<float>())
	, regularization(input["regularization"].get<float>())
	, inputClass(nullptr)
{
	std::vector<nlohmann::json> layersJson = input["layers"].get<std::vector<nlohmann::json>>();
	layers.resize(layersJson.size());

	for (int i = 0; i < layersJson.size(); ++i) {
		/*switch (str2int(layersJson[i]["layertype"].get<std::string>().c_str()))
		{
		case str2int("input"):
			layers[i] = new InputLayer(layersJson[i]);
			break;

		case str2int("general"):
			layers[i] = new Layer(layersJson[i]);
			break;

		case str2int("output"):
			layers[i] = new OutputLayer(layersJson[i]);
			break;

		default:
			throw NeuralException("cannot parse layertype...");
			break;
		}*/

		switch (jsonToLayerType(layersJson[i]))
		{
		case LayerType::Input:
			layers[i] = new InputLayer(layersJson[i]);
			break;

		case LayerType::General:
			layers[i] = new Layer(layersJson[i]);
			break;

		case LayerType::Output:
			layers[i] = new OutputLayer(layersJson[i]);
			break;

		case LayerType::Convolutional:
			layers[i] = new ConvolutionLayer(layersJson[i]);
			break;

		case LayerType::Pooling:
			layers[i] = new PoolingLayer(layersJson[i]);
			break;

		default:
			throw NeuralException("\ncant parse layertype....\n");
			break;
		}
	}

	costFunctionType   = CostFunction::CrossEntropy;
	regularizationType = Regularization::None;

	connectLayers();
}

void Net::createNewLayer(const int& size, const NeuronType& neuronType, const LayerType& layerType)
{
	switch (layerType)
	{
	case LayerType::Input:

		layers.push_back(new InputLayer(size, neuronType, inputClass->getInputFunction()));

		break;

	case LayerType::General:

		if (layers.size() > 0) {
			layers.push_back(
				new Layer(size, neuronType, layerType, layers[layers.size() - 1])
			);
		}
		else {
			layers.push_back(new Layer(size, neuronType, layerType));
		}

		break;

	case LayerType::Output:

		layers.push_back(
			new OutputLayer(
				size,
				neuronType,
				costFunctionType,
				inputClass->getOutputFunction(),
				layers[layers.size() - 1]
			)
		);

		break;

	default:
		throw NeuralException("\nadding new layer failed\n");
		break;
	}

	connectLayers();
}

void Net::createNewLayer(
	const int& size,
	const LayerType& layertype,
	const int& width,
	const int& height,
	const PoolingMethod& pooling)
{
	switch (layertype)
	{
	case LayerType::Convolutional:

		layers.push_back(
			new ConvolutionLayer(size, width, height, layers[layers.size() - 1])
		);

		break;


	case LayerType::Pooling:

		layers.push_back(
			new PoolingLayer(size, pooling, width, height, static_cast<ConvolutionLayer*>(layers[layers.size() - 1]))
		);

		break;

	default:
		throw NeuralException("\nadding new layer failed\n");
		break;
	}
}

LayerBase* Net::getLayer(const int& layerNumber) const
{
	if (layerNumber > (layers.size() - 1)) {
		throw NeuralException("invalid layer access index...\n");
	}

	return layers[layerNumber];
}


LayerBase* Net::getLayer(const LayerType& layerType) const
{
	auto startIter = layers.begin();
	auto endIter   = layers.end();
	switch (layerType)
	{
	case LayerType::Input:

		while (startIter != layers.end()) {
			if (dynamic_cast<InputLayer*>(*startIter)) return *startIter;
			startIter++;
		}

		throw NeuralException("Theres no inputlayer in the network\n");

		break;

	case LayerType::Output:
		endIter--;
		while (endIter != layers.begin()) {
			if (dynamic_cast<OutputLayer*>(*endIter)) return *endIter;
			endIter--;
		}

		throw NeuralException("Theres no outputlayer in the network\n");

		break;

	default:
		throw NeuralException("What kind of a layer do you want?\n");
		break;
	}
}

LayerBase* Net::getLastLayer() const
{
	return layers[layers.size() - 1];
}

void Net::addInputClass(NeuralInputClass* inputclass)
{
	inputClass = inputclass;

	static_cast<InputLayer*>(getLayer(LayerType::Input))->setInputFunction(inputclass->getInputFunction());
	static_cast<OutputLayer*>(getLayer(LayerType::Output))->setIdealOutput(inputclass->getOutputFunction());
}

// might be inefficient? 
void Net::connectLayers()
{
	if (layers.size() == 1) return;

	layers[0]->setNextLayer(layers[1]);
	layers[layers.size() - 1]->setPreviousLayer(layers[layers.size() - 2]);

	for (int i = 1; i < layers.size() - 1; ++i) {
		layers[i]->setPreviousLayer(layers[i - 1]);
		layers[i]->setNextLayer(layers[i + 1]);
	}
}

void Net::testForward()
{
	inputClass->shuffle();
	inputClass->resetCounter();
	dynamic_cast<OutputLayer*>(getLastLayer())->resetCounters();

	calculateActivationInAllLayers();
	calculateDeltaInAllLayers();
	calculateDerivativesInAllLayers();

	printOutputLayer();
}

void Net::train(
	const int&        epochNumber,
	const int&        minibatchSizeParam,
	const float&      newEta,
	const float&	  regularizationParam)
{
	// save these to a member variable so the json writer can access it later
	epochs         = epochNumber;
	minibatchSize  = minibatchSizeParam;
	eta            = newEta;
	regularization = regularizationParam;

	std::cout << "Starting calculations...\n";
	printLayerInfo();

	int minibatchNumber = inputClass->getTotalSize() / minibatchSize;

	std::cout << "Network parameters:\nEpoch count: " << epochs 
		<< "\nMinibatch size: " << minibatchSize << "\nEta: " << eta << "\n\n";

	for (int epochCounter = 0; epochCounter < epochs; ++epochCounter) {

		inputClass->shuffle();      // shuffle the images
		inputClass->resetCounter(); // reset the curr counter in the imageholder
		static_cast<OutputLayer*>(getLastLayer())->resetCounters();  // reset the hit counter in the outputlayer

		for (int minibatch = 0; minibatch < minibatchNumber; ++minibatch) {

			std::vector<Matrix<float>> weightUpdater;
			std::vector<Matrix<float>> biasUpdater;
			weightUpdater.resize(layers.size());
			biasUpdater.resize(layers.size());

			for (int i = 0; i < layers.size(); ++i) {
				if (dynamic_cast<InputLayer*>(layers[i])) continue;

				weightUpdater[i] = Matrix<float>(layers[i]->getSize(), layers[i - 1]->getSize());
				biasUpdater[i]   = Matrix<float>(layers[i]->getSize(), 1);
			}


			for (int minibatchCounter = 0; minibatchCounter < minibatchSize; ++minibatchCounter) {

				calculateActivationInAllLayers();

				calculateDeltaInAllLayers();
				calculateDerivativesInAllLayers();

				addUpWeightsAndBiases(weightUpdater, biasUpdater);

			}
			

			updateWeightsAndBiases(weightUpdater, biasUpdater, eta / float(minibatchSize),
				regularization, inputClass->getTotalSize());
			//printOutputLayer();
			
		}
		
		std::cout << "After epoch number " << epochCounter + 1 
			<< ", the ratio is: "<< static_cast<OutputLayer*>(getLastLayer())->getRatio() << '\n';
		//std::cout << layers[1]->getCostWeight();
		//printOutputLayer();
	}
}

void Net::work()
{

}

void Net::calculateActivationInAllLayers() const
{
	for (LayerBase* layer : layers) {
		layer->calculateActivation();
	}
}

void Net::calculateDeltaInAllLayers() const
{
	for (int i = layers.size() - 1; i >= 0; --i) {  // inputlayer wont do anything, we can include it
		//static_cast<Layer*>(layers[i])->calculateDelta();
		layers[i]->calculateDelta();
	}
}

void Net::calculateDerivativesInAllLayers() const
{
	//for (int i = layers.size() - 1; i >= 0; --i) {  // inputlayer wont do anything, we can include it
	//	static_cast<Layer*>(layers[i])->calculateCostWeight();
	//}

	for (int i = layers.size() - 1; i >= 0; --i) {
		layers[i]->calculateCostWeight();
	}
}

void Net::addUpWeightsAndBiases(
	std::vector<Matrix<float>>& weights,
	std::vector<Matrix<float>>& biases) const
{
	for (int i = 0; i < layers.size(); ++i) {
		if (dynamic_cast<InputLayer*>(layers[i])) continue;

		weights[i] += static_cast<Layer*>(layers[i])->getCostWeight();
		biases[i]  += static_cast<Layer*>(layers[i])->getCostBias();
	}
}

void Net::updateWeightsAndBiases(
	const std::vector<Matrix<float>>& weights,
	const std::vector<Matrix<float>>& biases,
	const float& multiplier,
	const float& regularizationParam,
	const int&   trainingSetSize) const
{
	for (int i = 0; i < layers.size(); ++i) {
		if (dynamic_cast<InputLayer*> (layers[i])) continue;
		if (dynamic_cast<OutputLayer*>(layers[i])) continue;

		static_cast<Layer*>(
			layers[i])->update(regularizationType, weights[i], biases[i], multiplier, regularizationParam, trainingSetSize
		);
	}
}

void Net::printLayer(const int& layerNumber) const
{
	//Layer* out = static_cast<Layer*>(getLayer(layerNumber));

	std::cout << "Printing layer number: " << layerNumber << "...\n";
	/*for (Neuron* neuron : out->getNeurons()) {
		std::cout << neuron->getResult() << '\n';
	}*/
	getLayer(layerNumber)->printLayer();

	std::cout << "\n\n";
}

void Net::printOutputLayer() const
{
	Layer* out = static_cast<Layer*>(getLayer(LayerType::Output));

	std::cout << "Printing output layer...\n";
	//for (Neuron* neuron : out->getNeurons()) {
		//std::cout << neuron->getResult() << '\n';
		
	//}
	out->getActivations().print();

	std::cout << "\n\n";
}

void Net::printLayerInfo() const
{
	int count = 0;
	for (LayerBase* layer : layers) {
		std::cout << "LAYER NUMBER " << count << ":\n";
		layer->printLayerInfo();
		count++;
	}
}

nlohmann::json Net::toJSON() const
{
	nlohmann::json ret;

	ret["epochs"]		  = epochs;
	ret["minibatchsize"]  = minibatchSize;
	ret["eta"]			  = eta;
	ret["regularization"] = regularization;

	std::vector<nlohmann::json> jsonLayers;
	jsonLayers.resize(layers.size());

	for (int i = 0; i < layers.size(); ++i) {
		jsonLayers[i] = layers[i]->toJSON();
	}

	ret["layers"] = jsonLayers;

	return ret;
}

void Net::saveAs(const char* filename) const
{
	std::ofstream o(filename);
	o << toJSON();
	o.close();
}

Net::~Net()
{
	for (int i = 0; i < layers.size(); ++i) {
		delete layers[i];
	}
}
