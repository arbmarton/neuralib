#include "stdafx.h"
#include "MNISTImage.h"


int reverseInt(const int& i)
{
	unsigned char c1, c2, c3, c4;

	c1 =         i & 255;    // takes the last 8 bits of the number i basically, clever
	c2 = (i >>  8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

ImageHolder::ImageHolder(const int& imagesToRead)
	: curr(0)
{
	std::ifstream file("train-images.idx3-ubyte", std::ios::binary | std::ios::in);
	if (!file.is_open()) {
		std::cout << "cant open image file...";
		return;
	}

	std::vector<std::pair<MNISTImage*, short>> pairs;
	pairs.resize(imagesToRead);

	// https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c fuck datastreams honestly
	int magic_number = 0;
	int number_of_images = 0;
	int n_rows = 0;
	int n_cols = 0;

	file.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	file.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);

	file.read((char*)&n_rows, sizeof(n_rows));
	n_rows = reverseInt(n_rows);
	file.read((char*)&n_cols, sizeof(n_cols));
	n_cols = reverseInt(n_cols);

	//std::cout << magic_number << ' ' << number_of_images << ' ' << n_rows << ' ' << n_cols << '\n';

	number_of_images = imagesToRead;

	/*std::vector<MNISTImage*> images;
	images.resize(number_of_images);*/

	for (int i = 0; i < number_of_images; ++i) {
		//images[i] = new MNISTImage(n_rows, n_cols);
		pairs[i].first = new MNISTImage(n_rows, n_cols);

		for (int j = 0; j < n_rows; ++j) {
			for (int k = 0; k < n_cols; ++k) {
				unsigned char input;
				file.read((char*)&input, sizeof(input));
				//images[i]->data[j*n_cols + k] = input;
				pairs[i].first->data[j*n_cols + k] = input;

				//file.read((char*)images[i]->data[j*n_cols + k], sizeof(unsigned char));  // why is this not working?
			}
		}
	}

	//MNISTimages = images;

	std::ifstream fileLabels("train-labels.idx1-ubyte", std::ios::binary | std::ios::in);
	if (!file.is_open()) {
		std::cout << "cant open label file...";
		return;
	}

	fileLabels.read((char*)&magic_number, sizeof(magic_number));
	magic_number = reverseInt(magic_number);
	fileLabels.read((char*)&number_of_images, sizeof(number_of_images));
	number_of_images = reverseInt(number_of_images);

	number_of_images = imagesToRead;

	//std::vector<short> labelsTemp;
	//labelsTemp.resize(number_of_images);
	for (int i = 0; i < number_of_images; ++i) {
		unsigned char input;
		fileLabels.read((char*)&input, sizeof(input));
		//labelsTemp[i] = unsigned short(input);
		pairs[i].second = unsigned short(input);
	}

	//labels = labelsTemp;

	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(pairs), std::end(pairs), rng);

	MNISTimages.resize(imagesToRead);
	labels.resize(imagesToRead);

	for (int i = 0; i < imagesToRead; ++i) {
		MNISTimages[i] = pairs[i].first;
		labels[i] = pairs[i].second;
	}

	file.close();
}

ImageHolder::ImageHolder(const ImageHolder& other)
	: MNISTimages(other.MNISTimages)
	, labels(other.labels)
	, curr(other.curr)
{
	std::transform(
		other.MNISTimages.begin(),
		other.MNISTimages.end(),
		MNISTimages.begin(),
		[](MNISTImage* ptr) { 
			MNISTImage* heap = new MNISTImage;
			*heap = *ptr;
			return heap;
		}
	);

	std::transform(
		other.labels.begin(),
		other.labels.end(),
		labels.begin(),
		[](short val) { return val; }
	);
}

ImageHolder::ImageHolder(ImageHolder&& other)
	: MNISTimages(std::move(other.MNISTimages))
	, labels(std::move(other.labels))
	, curr(other.curr)
{

}

ImageHolder& ImageHolder::operator=(const ImageHolder& other)
{
	ImageHolder temp(other);
	*this = std::move(temp);
	return *this;
}

ImageHolder& ImageHolder::operator=(ImageHolder&& other)
{
	MNISTimages = std::move(other.MNISTimages);
	labels = std::move(other.labels);
	curr = other.curr;
	return *this;
}

void ImageHolder::shuffle()
{
	int imageCount = labels.size();

	std::vector<std::pair<MNISTImage*, short>> pairs;
	pairs.resize(imageCount);

	for (int i = 0; i < imageCount; ++i) {
		pairs[i].first = MNISTimages[i];
		pairs[i].second = labels[i];
	}

	auto rng = std::default_random_engine{};
	std::shuffle(std::begin(pairs), std::end(pairs), rng);

	for (int i = 0; i < imageCount; ++i) {
		MNISTimages[i] = pairs[i].first;
		labels[i] = pairs[i].second;
	}
}