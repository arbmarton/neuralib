#pragma once

#include "Matrix.h"

#include <math.h>

enum class CostFunction {
	LeastSquares,
	CrossEntropy
};

template<class T>
float sigmoid(const T& input)
{
	return 1 / (1 + exp(-input));
}

template<class T>
Matrix<T> sigmoid(const Matrix<T>& input)
{
	Matrix<T> ret(input.getRows(), input.getCols());

	for (int i = 0; i < input.getRows(); ++i) {
		for (int j = 0; j < input.getCols(); ++j) {
			ret(i, j) = sigmoid(input(i, j));
		}
	}

	return ret;
}

template<class T>
float sigmoidDerivative(const T& input)
{
	// if its too small or too large dont bother, the threshold could be even lower
	/*if		(input >  50) return 0;
	else if (input < -50) return 0;

	float power = exp(input);

	return power / ((power + 1)*(power + 1));*/

	float sig = sigmoid(input);

	return sig*(1 - sig);
}

template<class T>
Matrix<T> sigmoidDerivative(const Matrix<T>& input)
{
	Matrix<T> ret(input.getRows(), input.getCols());

	for (int i = 0; i < input.getRows(); ++i) {
		for (int j = 0; j < input.getCols(); ++j) {
			ret(i, j) = sigmoidDerivative(input(i, j));
		}
	}

	return ret;
}

template<class T>
Matrix<T> crossEntropy(const Matrix<T>& input)
{
	
}

// basically a string to int hash function for switching
// https://stackoverflow.com/questions/16388510/evaluate-a-string-with-a-switch-in-c
inline constexpr unsigned int str2int(const char* str, int h = 0)
{
	return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

// singleton class for getting Mersenne-Twister objects
class Random {
public:
	static std::mt19937& getMT() {
		static std::mt19937 mt(getDevice()());
		return mt;
	}

private:
	Random() = delete;
	Random(const Random& other) = delete;
	Random(Random&& other) = delete;
	Random& operator=(const Random& other) = delete;
	Random& operator=(Random&& other) = delete;

	static std::random_device& getDevice() {
		static std::random_device rd;
		return rd;
	}
};