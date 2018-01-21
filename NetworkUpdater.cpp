#include "stdafx.h"
#include "NetworkUpdater.h"

NetworkUpdater::NetworkUpdater(Net* _parent)
	: parent(_parent)
{
	init();
}

void NetworkUpdater::init()
{
	updates.resize(parent->getLayerCount());
	for (int i = 0; i < updates.size(); ++i) {
		if (dynamic_cast<InputLayer*>(parent->getLayer(i)) 
			|| dynamic_cast<PoolingLayer*>(parent->getLayer(i)) 
			|| dynamic_cast<OutputLayer*>(parent->getLayer(i)))
		{
			updates[i] = nullptr;
		}
		else if (dynamic_cast<Layer*>(parent->getLayer(i))) {
			updates[i] = new LayerUpdater;
			updates[i]->init(parent->getLayer(i));
		}
		else {
			updates[i] = new ConvolutionUpdater;
			updates[i]->init(parent->getLayer(i));
		}
	}
}

void NetworkUpdater::reset()
{
	for (int i = 0; i < updates.size(); ++i) {
		if (updates[i]) {
			updates[i]->reset();
		}
	}
}

void NetworkUpdater::addUpWeightsAndBiases()
{
	for (int i = 0; i < parent->getLayerCount(); ++i) {
		auto currBase = parent->getLayer(i);
		if (dynamic_cast<InputLayer*>(currBase) || dynamic_cast<OutputLayer*>(currBase) || dynamic_cast<PoolingLayer*>(currBase)) {
			continue;
		}
		else if (dynamic_cast<Layer*>(currBase)) {
			auto currLayer  = static_cast<Layer*>(currBase);
			auto currUpdate = static_cast<LayerUpdater*>(updates[i]);

			currUpdate->addToFirst(currLayer->getCostBias());
			currUpdate->addToSecond(currLayer->getCostWeight());
		}
		else {
			auto currConv = static_cast<ConvolutionLayer*>(currBase);
			auto currUpdate = static_cast<ConvolutionUpdater*>(updates[i]);

			for (int j = 0; j < currUpdate->getUpdatePair().size(); ++j) {
				currUpdate->addToFirst(currConv->getMaps()[j]->getDelta(), j);
				currUpdate->addToSecond(currConv->getMaps()[j]->getCostWeight(), j);
			}
		}
	}
}
//
//std::vector<NetworkUpdater::Updater*>& NetworkUpdater::getUpdates()
//{
//	return updates;
//}

Matrix<float> NetworkUpdater::getMatrixLayerUpdater(const int& updater, const bool bias) const
{
	if (bias) {
		return static_cast<LayerUpdater*>(updates[updater])->getUpdatePair().first;
	}
	else {
		return static_cast<LayerUpdater*>(updates[updater])->getUpdatePair().second;
	}
}

std::vector<std::pair<Matrix<float>, Matrix<float>>>& NetworkUpdater::getVectorUpdaterConv(const int& pos)
{
	return static_cast<ConvolutionUpdater*>(updates[pos])->getUpdatePair();
}


NetworkUpdater::~NetworkUpdater()
{
	for (int i = 0; i < updates.size(); ++i) {
		delete updates[i];
	}
}


NetworkUpdater::Updater::Updater()
{

}

NetworkUpdater::Updater::~Updater()
{	
}



NetworkUpdater::LayerUpdater::LayerUpdater()
	: Updater()
{

}

void NetworkUpdater::LayerUpdater::init(LayerBase* layer)
{
	auto layerPtr = dynamic_cast<Layer*>(layer);

	updatePair.first = Matrix<float>(layerPtr->getBias().getRows(), layerPtr->getBias().getCols());
	//updatePair.first.fillValue(0.0f);

	updatePair.second = Matrix<float>(layerPtr->getCostWeight().getRows(), layerPtr->getCostWeight().getCols());
	//updatePair.second.fillValue(0.0f);
}

void NetworkUpdater::LayerUpdater::reset()
{
	updatePair.first.fillValue(0.0f);
	updatePair.second.fillValue(0.0f);
}

std::pair<Matrix<float>, Matrix<float>> NetworkUpdater::LayerUpdater::getUpdatePair() const
{
	return updatePair;
}

void NetworkUpdater::LayerUpdater::addToFirst(const Matrix<float>& mat)
{
	updatePair.first += mat;
}

void NetworkUpdater::LayerUpdater::addToSecond(const Matrix<float>& mat)
{
	updatePair.second += mat;
}

NetworkUpdater::LayerUpdater::~LayerUpdater()
{
}




NetworkUpdater::ConvolutionUpdater::ConvolutionUpdater()
	: Updater()
{

}

void NetworkUpdater::ConvolutionUpdater::init(LayerBase* layer)
{
	auto layerPtr = dynamic_cast<ConvolutionLayer*>(layer);

	updatePair.resize(layerPtr->getSize());
	for (int i = 0; i < updatePair.size(); ++i) {
		updatePair[i].first = Matrix<float>(
			layerPtr->getMaps()[i]->getDelta().getRows(),
			layerPtr->getMaps()[i]->getDelta().getCols()
		);

		updatePair[i].second = Matrix<float>(
			layerPtr->getMaps()[i]->getCostWeight().getRows(),
			layerPtr->getMaps()[i]->getCostWeight().getCols()
		);
	}
}

void NetworkUpdater::ConvolutionUpdater::reset()
{
	for (int i = 0; i < updatePair.size(); ++i) {
		updatePair[i].first.fillValue(0.0f);
		updatePair[i].second.fillValue(0.0f);
	}
}

std::vector<std::pair<Matrix<float>, Matrix<float>>>& NetworkUpdater::ConvolutionUpdater::getUpdatePair()
{
	return updatePair;
}

void NetworkUpdater::ConvolutionUpdater::addToFirst(const Matrix<float>& mat, const int& pos)
{
	updatePair[pos].first += mat;
}

void NetworkUpdater::ConvolutionUpdater::addToSecond(const Matrix<float>& mat, const int& pos)
{
	updatePair[pos].second += mat;
}

NetworkUpdater::ConvolutionUpdater::~ConvolutionUpdater()
{
}