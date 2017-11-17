#include "stdafx.h"
#include "NeuralException.h"


NeuralException::NeuralException(const std::string& errorString)
	:errorMsg(errorString)
{
}

std::string NeuralException::getErrorMessage() const
{
	return errorMsg;
}

NeuralException::~NeuralException()
{
}
