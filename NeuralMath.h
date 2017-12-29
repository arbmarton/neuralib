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

// do i need to prepare for inputX != inputY?
template<class T>
void convolve(
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
			
			std::cout << xStart + yStart*resultX;
			result[xStart + yStart*resultX] = accum;
		}
	}
}

template<class T>
void convolve(const Matrix<T>& mat, const FeatureMap& feat)
{
	convolve(
		mat.getData(), mat.getCols(), mat.getRows(),
		feat.getKernel().getData(), feat.getKernel().getCols(), feat.getKernel().getRows(),
		feat.getResult().getData(), feat.getResult().getCols(), feat.getResult().getRows()
	);
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

template<class T>
void createPool(const PoolingMethod& pooltype, const FeatureMap& feat, const Pool& pool)
{
	createPool(
		pooltype, pool.getWidth(), pool.getHeight(),
		feat.getResult(), feat.getResult().getCols(), feat.getResult().getRows(),
		pool.getResult(), pool.getResult().getCols(), pool.getResult().getRows()
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