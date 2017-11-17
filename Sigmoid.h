#pragma once

#include <list>
#include <math.h>

class Sigmoid
{
public:
	Sigmoid(const float& nthreshold, std::list<Sigmoid> nchildren = std::list<Sigmoid>());

	float getThreshold() const;
	void   setThreshold(const float& newValue);

	std::list<Sigmoid> getChildren() const;
	void setChildren(const std::list<Sigmoid>& newChildren);



	~Sigmoid();
private:
	float					threshold;
	std::list<Sigmoid>		children;
};

