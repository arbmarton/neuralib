#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

struct MNISTImage
{
	MNISTImage()
		: n(0)
		, m(0)
		, data(nullptr)
	{

	}

	MNISTImage(const int& newn, const int& newm)
		: n(newn)
		, m(newm)
		, data(new unsigned char[newn*newm])
	{

	}

	MNISTImage(const MNISTImage& other)
		: n(other.n)
		, m(other.m)
		, data(new unsigned char[other.n*other.m])
	{
		std::memcpy(data, other.data, sizeof(unsigned char)*other.n*other.m);
	}

	MNISTImage(MNISTImage&& other)
		: n(other.n)
		, m(other.m)
		, data(other.data)
	{
		other.data = nullptr;
	}

	MNISTImage& operator=(const MNISTImage& other)
	{
		MNISTImage temp(other);
		*this = std::move(temp);
		return *this;
	}

	MNISTImage& operator=(MNISTImage&& other)
	{
		n = other.n;
		m = other.m;

		delete[] data;
		data = other.data;
		other.data = nullptr;

		return *this;
	}

	void print() {
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				std::cout << data[i*m + j] << ' ';
			}
			std::cout << '\n';
		}
		std::cout << '\n';
	}

	~MNISTImage() {
		delete[] data;
	}

	unsigned char* data;
	int n;
	int m;
};

inline
std::ostream& operator<<(std::ostream& os, const MNISTImage& img)
{
	for (int i = 0; i < img.n; ++i) {
		for (int j = 0; j < img.m; ++j) {
			std::cout << img.data[i*img.m + j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
	return os;
}

int reverseInt(const int& i);

struct ImageHolder
{
	ImageHolder() {};
	ImageHolder(const int& imagesToRead = 60000);
	ImageHolder(const ImageHolder& other);
	ImageHolder(ImageHolder&& other);

	ImageHolder& operator=(const ImageHolder& other);
	ImageHolder& operator=(ImageHolder&& other);

	std::vector<MNISTImage*> MNISTimages;
	std::vector<short>       labels;
	int curr;

	void shuffle();

	~ImageHolder() {
		for (MNISTImage* image : MNISTimages) {
			delete image;
		}
	}
};