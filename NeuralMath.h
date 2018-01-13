#pragma once

#include "Layer.h"
#include "Matrix.h"
#include "FeatureMap.h"
#include "Pool.h"

#include <math.h>
#include <limits>

class FeatureMap;
class Pool;

enum class CostFunction 
{
	LeastSquares,
	CrossEntropy
};

enum class Regularization
{
	None,
	L1,
	L2
};

enum class PoolingMethod
{
	max,
	L2
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

/*template<class T>
float tanh(const T& input)
{
	const float arg1 = exp(input);
	const float arg2 = exp(-input);

	const float ret = (arg1 - arg2) / (arg1 + arg2);
	return (ret + 1) / 2;
}*/

template<class T>
Matrix<T> tanh(const Matrix<T>& input)
{
	Matrix<T> ret(input.getRows(), input.getCols());

	for (int i = 0; i < input.getRows(); ++i) {
		for (int j = 0; j < input.getCols(); ++j) {
			ret(i, j) = tanh(input(i, j));
		}
	}

	return ret;
}

template<class T>
float relu(const T& input)
{
	return std::max(T(0), input);
}

template<class T>
Matrix<T> relu(const Matrix<T>& input)
{
	Matrix<T> ret(input.getRows(), input.getCols());

	for (int i = 0; i < input.getRows(); ++i) {
		for (int j = 0; j < input.getCols(); ++j) {
			ret(i, j) = relu(input(i, j));
		}
	}

	return ret;
}

template<class T>
float sigmoidDerivative(const T& input)
{
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

// this should only get positive matrices
template<class T>
Matrix<T> softMax(const Matrix<T>& input)
{
	Matrix<T> ret(input.getRows(), input.getCols());

	T accum = T(0);
	for (int i = 0; i < input.getRows(); ++i) {
		for (int j = 0; j < input.getCols(); ++j) {
			accum += input(i, j);
		}
	}

	for (int i = 0; i < input.getRows(); ++i) {
		for (int j = 0; j < input.getCols(); ++j) {
			ret(i, j) = input(i, j) / accum;
		}
	}

	if (accum < 0) {
	//	std::cout << "\naccum was negative: " << accum << '\n';
	}

	return ret;
}

// do i need to prepare for inputX != inputY?
template<class T>
void validConvolution(
	const T* const input,
	const int& inputX,
	const int& inputY,
	const T* const kernel,
	const int& kernelX,
	const int& kernelY,
	T* result,
	const int& resultX,
	const int& resultY)
{
	//const int resultX = inputX - 2 * floor(kernelX / 2);
	//const int resultY = inputY - 2 * floor(kernelY);

	for (int xStart = 0; xStart < inputX - kernelX + 1; ++xStart) {
		for (int yStart = 0; yStart < inputY - kernelY + 1; ++yStart) {

			T accum = T(0);
			for (int i = 0; i < kernelX; ++i) {
				const int offset = xStart + i;
				for (int j = 0; j < kernelY; ++j) {
					accum += input[offset + (yStart + j)*inputX] * kernel[i + j*kernelX];
				}
			}
			
//			std::cout << xStart + yStart*resultX;
			result[xStart + yStart*resultX] = accum;
		}
	}
}

template<class T>
void validConvolution(const Matrix<T>& mat, const FeatureMap& feat)
{
	validConvolution(
		mat.getData(), mat.getCols(), mat.getRows(),
		feat.getKernel().getData(), feat.getKernel().getCols(), feat.getKernel().getRows(),
		feat.getResult().getData(), feat.getResult().getCols(), feat.getResult().getRows()
	);
}

// pad the input with zeroes and do a validconvolution
template<class T>
Matrix<T> fullConvolution(const Matrix<T>& kernel, const Matrix<T>& delta)
{
	Matrix<T> ret(delta.getRows() + kernel.getRows() - 1, delta.getCols() + kernel.getCols() - 1);
	Matrix<T> rotatedKernel = kernel.rotate180();
	Matrix<T> paddedInput(delta.getRows() + kernel.getRows() * 2 - 2, delta.getCols() + kernel.getCols() * 2 - 2);
	
	for (int i = 0; i < delta.getRows(); ++i) {
		for (int j = 0; j < delta.getCols(); ++j) {
			paddedInput(i + kernel.getRows() - 1, j + kernel.getCols() - 1) = delta(i, j);
		}
	}

	validConvolution(
		paddedInput.getData(), paddedInput.getCols(), paddedInput.getRows(),
		rotatedKernel.getData(), rotatedKernel.getCols(), rotatedKernel.getRows(),
		ret.getData(), ret.getCols(), ret.getRows()
	);

	return ret;
}

template<class T>
void createPool(
	const PoolingMethod& pooltype,
	const int& poolX,
	const int& poolY,
	const T* const input,
	const int& inputX,
	const int& inputY,
	T* result,
	const int& resultX,
	const int& resultY)
{
	for (int i = 0; i < resultX; ++i) {
		for (int j = 0; j < resultY; ++j) {

			T accum;

			switch (pooltype)
			{
			case PoolingMethod::max:

				accum = std::numeric_limits<T>::min();

				for (int k = i*poolX; k < (i + 1)*poolX; ++k) {
					for (int l = j*poolY; l < (j + 1)*poolY; ++l) {

						T temp = input[k + l*inputX];

						if (temp > accum) {
							accum = temp;
						}
					}
				}

				break;

			case PoolingMethod::L2:

				accum = 0;

				for (int k = i*poolX; k < (i + 1)*poolX; ++k) {
					for (int l = j*poolY; l < (j + 1)*poolY; ++l) {

						T temp = input[k + l*inputX];

						accum += temp*temp;
					}
				}

				accum = sqrt(accum);

				break;

			default:
				break;
			}

			result[i + resultX*j] = accum;
		}
	}
}

// if i inline it it wont compile, why?
template<int T = 0>
void createPool(const PoolingMethod& pooltype, const FeatureMap* const feat, const Pool* pool)
{
	createPool(
		pooltype, pool->getWidth(), pool->getHeight(),
		feat->getResult().getData(), feat->getResult().getCols(), feat->getResult().getRows(),
		pool->getResult().getData(), pool->getResult().getCols(), pool->getResult().getRows()
	);
}

// basically a string to int hash function for switching, credit:
// https://stackoverflow.com/questions/16388510/evaluate-a-string-with-a-switch-in-c
inline constexpr unsigned int str2int(const char* str, int h = 0)
{
	return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h];
}

// credit:
// https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
int sgn(const T& val)
{
	return (T(0) < val) - (val < T(0));
}

// singleton class for getting Mersenne-Twister objects
class Random {
public:
	static std::mt19937& getMT() {
		static std::mt19937 mt(getDevice()()); // The Parentheses of Trickery™
		return mt;
	}

private:
	Random()								= delete;
	Random(const Random& other)				= delete;
	Random(Random&& other)				    = delete;
	Random& operator=(const Random& other)  = delete;
	Random& operator=(Random&& other)		= delete;

	static std::random_device& getDevice() {
		static std::random_device rd;
		return rd;
	}
};