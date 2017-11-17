#include "stdafx.h"
#include "Sigmoid.h"


Sigmoid::Sigmoid(const float& nthreshold, std::list<Sigmoid> nchildren)
	: threshold(nthreshold)
	, children(nchildren)
{
}

float Sigmoid::getThreshold() const
{
	return threshold;
}

void Sigmoid::setThreshold(const float& newValue)
{
	threshold = newValue;
}

std::list<Sigmoid> Sigmoid::getChildren() const
{
	return children;
}

void Sigmoid::setChildren(const std::list<Sigmoid>& newChildren)
{
	children = newChildren;
}

Sigmoid::~Sigmoid()
{
}
