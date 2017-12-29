#pragma once

#include <exception>
#include <string>

class NeuralException : public std::exception
{
public:
	NeuralException(const std::string& errorString);

	std::string getErrorMessage() const;

	~NeuralException();
private:
	std::string errorMsg;
};