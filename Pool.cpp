#include "stdafx.h"
#include "Pool.h"


Pool::Pool(const PoolingMethod& _method, const int& kernelWidth, const int& kernelHeight, const PoolingLayer* _parent)
	: method(_method)
	, width(kernelWidth)
	, height(kernelHeight)
{
	ConvolutionLayer* prev = static_cast<ConvolutionLayer*>(_parent->getPreviousLayer());
	const int previousWidth = prev->getMaps()[0]->getResult().getCols();
	const int resultWidth = previousWidth / kernelWidth;

	result = Matrix<float>(resultWidth, resultWidth);
}

int Pool::getWidth() const
{
	return width;
}

int Pool::getHeight() const
{
	return height;
}

Matrix<float> Pool::getResult() const
{
	return result;
}

Pool::~Pool()
{
}
