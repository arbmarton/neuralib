#pragma once

#include "Matrix.h"
#include "Layer.h"
#include "Net.h"

class Net;

class NetworkUpdater
{
	class Updater {
	public:
		Updater();

		virtual void init(LayerBase* layer) = 0;
		virtual void reset() = 0;

		virtual ~Updater();
	};

	class LayerUpdater : public Updater {
	public:
		LayerUpdater();

		virtual void init(LayerBase* layer);
		virtual void reset() override;

		std::pair<Matrix<float>, Matrix<float>> getUpdatePair() const;

		void addToFirst(const Matrix<float>& mat);
		void addToSecond(const Matrix<float>& mat);

		virtual ~LayerUpdater();
	private:
		std::pair<Matrix<float>, Matrix<float>> updatePair;		// first is bias, second is costweight
	};

	class ConvolutionUpdater : public Updater {
	public:
		ConvolutionUpdater();

		virtual void init(LayerBase* layer);
		virtual void reset() override;

		std::vector<std::pair<Matrix<float>, Matrix<float>>>& getUpdatePair();

		void addToFirst(const Matrix<float>& mat, const int& pos);
		void addToSecond(const Matrix<float>& mat, const int& pos);

		virtual ~ConvolutionUpdater();
	private:
		std::vector<std::pair<Matrix<float>, Matrix<float>>> updatePair;	// first is bias, second is costweight
	};


public:
	NetworkUpdater(Net* _parent);

	void init();
	void reset();
	void addUpWeightsAndBiases();

	//std::vector<Updater*>& getUpdates();
	Matrix<float> getMatrixLayerUpdater(const int& updater, const bool bias) const;
	std::vector<std::pair<Matrix<float>, Matrix<float>>>& getVectorUpdaterConv(const int& pos);

	~NetworkUpdater();
private:
	Net*				  parent;
	std::vector<Updater*> updates;
};

