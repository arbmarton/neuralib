#include "stdafx.h"
#include "NeuralInputClass.h"


NeuralInputClass::NeuralInputClass()
{
}


NeuralInputClass::~NeuralInputClass()
{
}

MNISTInputClass::MNISTInputClass(const int& imagesToRead)
	: totalImages(imagesToRead)
	, images(nullptr)
{

}

MNISTInputClass::MNISTInputClass(const MNISTInputClass& other)
	: totalImages(other.totalImages)
	, images(new ImageHolder(*other.images))
{

}

MNISTInputClass::MNISTInputClass(MNISTInputClass&& other)
	: totalImages(other.totalImages)
	, images(other.images)
{
	other.images = nullptr;
}

MNISTInputClass& MNISTInputClass::operator=(const MNISTInputClass& other)
{
	MNISTInputClass temp(other);
	*this = std::move(temp);
	return *this;
}

MNISTInputClass& MNISTInputClass::operator=(MNISTInputClass&& other)
{
	totalImages = other.totalImages;

	delete images;
	images = other.images;
	other.images = nullptr;

	return *this;
}

void MNISTInputClass::init()
{
	images = new ImageHolder(totalImages);
}

std::function<void(std::vector<Neuron*>&)> MNISTInputClass::getInputFunction() {
	std::function<void(std::vector<Neuron*>&)> func =
		[=](std::vector<Neuron*>& input) -> void	// capture by value or reference? which means what here?
		{
			this->imageToLayer(input);
		};
	return func;
}

std::function<void(std::vector<float>&)> MNISTInputClass::getOutputFunction() {
	std::function<void(std::vector<float>&)> func =
		[=](std::vector<float>& input) -> void
		{
			input.erase(input.begin(), input.end());
			input.resize(10);
			std::fill(input.begin(), input.end(), 0.0f);

			int val = getImageHolder()->labels[getImageHolder()->curr - 1]; // why do i need this offset?
			input[val] = 1.0f;
		};
	return func;
}

void MNISTInputClass::imageToLayer(std::vector<Neuron*>& input)
{
	if (!images) throw NeuralException("Image container was nullptr");

	MNISTImage* currImg = (images->MNISTimages)[images->curr];

	for (int i = 0; i < (currImg->m)*(currImg->n); ++i) {
		dynamic_cast<InputNeuron*>(input[i])->
			setResult(currImg->data[i] / 255.0f);
	}

	if (images->curr < images->MNISTimages.size()) {
		(images->curr)++;
	}
	else {
		throw NeuralException("ran out of images...");
	}
}

MNISTInputClass::~MNISTInputClass()
{
	delete images;
}
