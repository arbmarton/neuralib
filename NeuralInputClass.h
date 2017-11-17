#pragma once

#include "Neuron.h"
#include "NeuralException.h"
#include "MNISTImage.h"
#include "Matrix.h"

#include <algorithm>

class NeuralInputClass
{
public:
	NeuralInputClass();

	virtual void init() {};

	virtual void shuffle() {};
	virtual void resetCounter() {};

	virtual std::function<void(std::vector<Neuron*>&)> getInputFunction() = 0; // put the data in the neurons

	virtual std::function<void(std::vector<float>&)> getOutputFunction() = 0; // create a vector for the base of the cost function

	virtual ~NeuralInputClass();
private:

};

class MNISTInputClass : public NeuralInputClass
{
public:
	MNISTInputClass(const int& imagesToRead = 60000);
	MNISTInputClass(const MNISTInputClass& other);
	MNISTInputClass(MNISTInputClass&& other);

	MNISTInputClass& operator=(const MNISTInputClass& other);
	MNISTInputClass& operator=(MNISTInputClass&& other);

	void init() override;

	void shuffle() override { images->shuffle(); }
	void resetCounter() override { images->curr = 0; }

	int getTotalImages() const { return totalImages; }
	ImageHolder* getImageHolder() const { return images; }

	std::function<void(std::vector<Neuron*>&)> getInputFunction();    
	std::function<void(std::vector<float>&)>   getOutputFunction();

	~MNISTInputClass();
private:
	int			  totalImages;
	ImageHolder*  images;

	void imageToLayer(std::vector<Neuron*>& input);
};