#pragma once

#include "json.hpp"
#include "ThreadPool.h"

#include <memory>
#include <iostream>
#include <random>
#include <thread>
#include <iomanip>

//#define DEBUG_MODE

template<class T>
class Matrix
{
public:
	// default constructor needed for instantiating containers such as std::vector
	Matrix()  
		: n(0)
		, m(0)
		, data(nullptr)
	{

	}

	Matrix(const int& rows, const int& cols)
		: n(rows)
		, m(cols)
		, data(new T[rows*cols])
	{		
		#ifdef DEBUG_MODE
		if (rows < 0 || cols < 0)
			throw NeuralException("Matrix sizes cant be negative...");
		#endif // DEBUG_MODE

		memset(data, T(0), sizeof(T)*rows*cols);
	}

	Matrix(const std::vector<T>& inputVector)
		: n(inputVector.size())
		, m(1)
		, data(new T[inputVector.size()])
	{
		for (int i = 0; i < inputVector.size(); ++i) {
			data[i] = inputVector[i];
		}
	}

	Matrix(const nlohmann::json& input)
		: n(input["n"].get<int>())
		, m(input["m"].get<int>())
		, data(new T[input["n"].get<int>() * input["m"].get<int>()])
	{
		auto vec = input["data"].get<std::vector<float>>();

		for (int i = 0; i < vec.size(); ++i) {
			data[i] = vec[i];
		}
	}

	Matrix(const Matrix& other)
		: n(other.n)
		, m(other.m)
		, data(new T[other.n * other.m])
	{
		memcpy(data, other.data, sizeof(T)*other.n*other.m);
	}

	Matrix(Matrix&& other) noexcept
		: n(other.n)
		, m(other.m)
		, data(other.data)
	{
		other.data = nullptr;
	}

	Matrix& operator=(const Matrix& other) {
		Matrix temp(other);
		*this = std::move(temp);
		return *this;
	}

	Matrix& operator=(Matrix&& other) noexcept {
		delete[] data;

		n = other.n;
		m = other.m;

		data = other.data;
		other.data = nullptr;

		return *this;
	}

	int getRows() const { return n; }
	int getCols() const { return m; }
	int getSize() const { return n*m; }
	T*  getData() const { return data; }

	T& operator()(const int& i, const int& j) { 
		#ifdef DEBUG_MODE
		if (i < 0 || j < 0)
			throw NeuralException("Matrix index lower than zero...");
		if ((i + 1) > n || (j + 1) > m)
			throw NeuralException("Matrix index out of range...");
		#endif // DEBUG_MODE

		return data[i*m + j];
	}

	T operator()(const int& i, const int& j) const {
		#ifdef DEBUG_MODE
		if (i < 0 || j < 0)
			throw NeuralException("Matrix index lower than zero...");
		if ((i + 1) > n || (j + 1) > m)
			throw NeuralException("Matrix index out of range...");
		#endif // DEBUG_MODE

		return data[i*m + j];
	}

	Matrix& operator*=(const Matrix& rhs) {
		#ifdef DEBUG_MODE
		if (m != rhs.n)
			throw NeuralException("Matrix multiplication: incorrent matrix sizes...");
		#endif // DEBUG_MODE

		Matrix temp(n, rhs.m);

		// if the matrix is larger than a certain number use multithreading
		if (temp.getCols() * temp.getRows() > 10000) {

			std::vector<std::future<void>> fut;
			fut.resize(ThreadPoolWrapper::getThreads());
			for (int i = 0; i < ThreadPoolWrapper::getThreads(); ++i) {
				/*fut.emplace_back(
					ThreadPoolWrapper::getThreadPool().enqueue(
						getLambda(), this, &rhs, i, ThreadPoolWrapper::getThreads(), &temp
					)
				);*/

				fut[i] = ThreadPoolWrapper::getThreadPool().enqueue(
					getLambda(), this, &rhs, i, ThreadPoolWrapper::getThreads(), &temp
				);
			}

			// this is needed so the stack frame wont get destroyed while the calculations are still running
			for (int i = 0; i < fut.size(); ++i) {
				fut[i].get();
			}

		}
		else {
			for (int i = 0; i < temp.getRows(); ++i) {
				const int rowOffset = i*m;
				for (int j = 0; j < temp.getCols(); ++j) {
					T accum = T(0);

					for (int k = 0; k < m; ++k) {
						accum += data[/*i*m*/ rowOffset + k] * rhs(k, j);
					}

					temp(i, j) = accum;
				}
			}
		}
		
		*this = temp;
		return *this;
	}

	Matrix& operator+=(const Matrix& rhs) {
		#ifdef DEBUG_MODE
		if (n != rhs.n || m != rhs.m)
			throw NeuralException("Matrix dimension mismatch in addition...");
		#endif // DEBUG_MODE

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				data[i*m + j] += rhs(i, j);
			}
		}

		return *this;
	}

	Matrix& operator+=(const float& rhs) {
		for (int i = 0; i < n*m; ++i) {
			data[i] += rhs;
		}

		return *this;
	}

	Matrix& operator-=(const Matrix& rhs) {
		#ifdef DEBUG_MODE
		if (n != rhs.n || m != rhs.m)
			throw NeuralException("Matrix dimension mismatch in addition...");
		#endif // DEBUG_MODE

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				data[i*m + j] -= rhs(i, j);
			}
		}

		return *this;
	}

	Matrix& operator-=(const float& rhs) {
		for (int i = 0; i < n*m; ++i) {
			data[i] -= rhs;
		}

		return *this;
	}

	Matrix& operator*=(const float& rhs) {
		for (int i = 0; i < n*m; ++i) {
			data[i] *= rhs;
		}

		return *this;
	}

	void fillValue(const float& val) {
		for (int i = 0; i < n*m; ++i) {
			data[i] = val;
		}
	}

	void fillRand(const float& low, const float& high) {
		std::uniform_real_distribution<float> uni(low, high);

		for (int i = 0; i < n*m; ++i) {
			data[i] = uni(Random::getMT());
		}
	}

	void fillGauss(const float& mean, const float& sigma) {
		std::normal_distribution<float> uni(mean, sigma);

		for (int i = 0; i < n*m; ++i) {
			data[i] = uni(Random::getMT());
		}
	}

	void fillGaussNormalized(const float& mean, const float& sigma, const int& connections) {
		std::normal_distribution<float> uni(mean, sigma);

		for (int i = 0; i < n*m; ++i) {
			data[i] = uni(Random::getMT()) / sqrt(float(connections));
		}
	}

	T getSquaredDifference(const Matrix<T>& cmp) {
		#ifdef DEBUG_MODE
		if (n != cmp.n || m != cmp.m)
			throw NeuralException("Invalid matrix sizes in squared difference calculation!");
		#endif // DEBUG_MODE
		T accum = 0.0f;
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				accum += (data[i*m + j] - cmp(i, j)) * (data[i*m + j] - cmp(i, j));
			}
		}

		return accum;
	}

	Matrix<T> getSquaredDifferenceMatrix(const Matrix<T>& cmp) {
		#ifdef DEBUG_MODE
		if (n != cmp.n || m != cmp.m)
			throw NeuralException("Invalid matrix sizes in squared difference calculation!");
		#endif // DEBUG_MODE
		Matrix<T> ret(n, m);
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				ret(i,j) = (data[i*m + j] - cmp(i, j)) * (data[i*m + j] - cmp(i, j));
			}
		}

		return ret;
	}

	Matrix<T> getSquaredDifferenceMatrix(const std::vector<T>& cmp) {
		#ifdef DEBUG_MODE
		if (n != cmp.size())
			throw NeuralException("Invalid matrix sizes in squared difference (vector) calculation!");
		#endif // DEBUG_MODE

		Matrix<T> ret(n, 1);
		for (int i = 0; i < n; ++i) {
			ret(i, 0) = (data[i] - cmp[i]) * (data[i] - cmp[i]);
		}

		return ret;
	}

	int largestIndex() const {
		#ifdef DEBUG_MODE
		if (m != 1)
			throw NeuralException("Apply to column vectors only...");
		#endif // DEBUG_MODE

		int ret = 0;
		T largest = 0;
		for (int i = 0; i < n; ++i) {
			if (data[i] > largest) {
				largest = data[i];
				ret = i;
			}
				
		}

		return ret;
	}

	Matrix<T>& substOne() {
		for (int i = 0; i < n*m; ++i) {
			data[i] = 1 - data[i];
		}

		return *this;
	}

	Matrix<T> signum() const {
		Matrix<T> temp(*this);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				temp(i, j) = sgn(data[i*m + j]);
			}
		}

		return temp;
	}

	// this is used in convolutional backpropagation
	Matrix<T> rotate180() const {
		Matrix<T> temp(*this);

		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < m; ++j) {
				temp(i, j) = (*this)(n - i - 1, m - j - 1);
			}
		}

		return temp;
	}

	float averageOfElements() const {
		float ret = 0.0f;

		for (int i = 0; i < n*m; ++i) {
			ret += data[i];
		}

		ret /= float(n*m);

		return ret;
	}

	void print() const {
		std::cout << *this;
	}

	nlohmann::json toJSON() const {
		nlohmann::json ret;

		std::vector<T> jsonVector;
		jsonVector.resize(n*m);
		for (int i = 0; i < n*m; ++i) {
			jsonVector[i] = data[i];
		}

		ret["n"]	= n;
		ret["m"]	= m;
		ret["data"] = jsonVector;

		return ret;
	}

	~Matrix() noexcept {
		delete[] data;
	}
private:
	int n; // row
	int m; // col
	T* data;

	// used in threaded matrix multiplication
	std::function<void(
		const Matrix<T>* const left,
		const Matrix<T>* const right,
		const int start,
		const int increment,
		Matrix<T>* result)> getLambda()
	{
		auto lambda = [](
			const Matrix<T>* const left,
			const Matrix<T>* const right,
			const int start,
			const int increment,
			Matrix<T>* result
			) -> void
		{
			for (int i = start; i < result->getRows(); i += increment) {
				for (int j = 0; j < result->getCols(); ++j) {
					T accum = T(0);

					for (int k = 0; k < left->getCols(); ++k) {
						accum += (*left)(i, k) * (*right)(k, j);
					}

					(*result)(i, j) = accum;
				}
			}
		};
		return lambda;
	}
};

template<class T>
inline Matrix<T> operator*(Matrix<T> lhs, const Matrix<T>& rhs)
{
	lhs *= rhs;
	return lhs;
}

template<class T>
inline Matrix<T> operator+(Matrix<T> lhs, const Matrix<T>& rhs)
{
	lhs += rhs;
	return lhs;
}

template<class T>
inline Matrix<T> operator+(Matrix<T> lhs, const float& rhs)
{
	lhs += rhs;
	return lhs;
}

template<class T>
inline Matrix<T> operator-(Matrix<T> lhs, const Matrix<T>& rhs)
{
	lhs -= rhs;
	return lhs;
}

template<class T>
inline Matrix<T> operator-(Matrix<T> lhs, const float& rhs)
{
	lhs -= rhs;
	return lhs;
}

template<class T>
inline Matrix<T> operator*(Matrix<T> lhs, const float& rhs)
{
	lhs *= rhs;
	return lhs;
}

template<class T, class U>
inline Matrix<U> operator*(const T& lhs, const Matrix<U>& rhs)
{
	Matrix<T> ret(rhs.getRows(), rhs.getCols());

	for (int i = 0; i < rhs.getRows(); ++i) {
		for (int j = 0; j < rhs.getCols(); ++j) {
			ret(i, j) = rhs(i, j)*lhs;
		}
	}

	return ret;
}

template<class T>
inline Matrix<T> hadamardProduct(const Matrix<T>& left, const Matrix<T>& right)
{
	#ifdef DEBUG_MODE
	if (left.getRows() != right.getRows() || left.getCols() != right.getCols())
		throw NeuralException("Matrix dimension mismatch in hadamard product...");

	if (left.getCols() != 1)
		throw NeuralException("Hadamard product can only be applied to column vectors...");
	#endif // DEBUG_MODE

	Matrix<T> ret(left.getRows(), 1);
	for (int i = 0; i < left.getRows(); ++i) {
		ret(i, 0) = left(i, 0)*right(i, 0);
	}

	return ret;
}

template<class T>
inline Matrix<T> transpose(const Matrix<T>& rhs)
{
	Matrix<T> ret(rhs.getCols(), rhs.getRows());

	for (int i = 0; i < rhs.getRows(); ++i) {
		for (int j = 0; j < rhs.getCols(); ++j) {
			ret(j, i) = rhs(i, j);
		}
	}

	return ret;
}

template<class T>
inline std::ostream& operator<<(std::ostream& os, const Matrix<T>& mat)
{
	for (int i = 0; i < mat.getRows(); ++i) {
		for (int j = 0; j < mat.getCols(); ++j) {
			std::cout << std::setw(2) << mat(i, j) << ' ';
		}
		std::cout << '\n';
	}
	std::cout << '\n';
	return os;
}

template<class T>
inline bool operator==(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
	if (lhs.getCols() != rhs.getCols())
		return false;

	if (lhs.getRows() != rhs.getRows())
		return false;

	for (int i = 0; i < lhs.getRows(); ++i) {
		for (int j = 0; j < rhs.getCols(); ++j) {
			if (abs(lhs(i, j) - rhs(i, j)) > 0.000005f)   // TODO: make this not arbitrary
				return false;
		}
	}

	return true;
}

template<class T>
inline bool operator!=(const Matrix<T>& lhs, const Matrix<T>& rhs)
{
	return !(lhs == rhs);
}

