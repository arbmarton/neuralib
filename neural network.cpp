// neural network.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "Net.h"
#include "Layer.h"
#include "Neuron.h"
#include "NeuralException.h"
#include "MNISTImage.h"
#include "Matrix.h"
#include "NeuralInputClass.h"

#include "json.hpp"

#include <iostream>
#include <fstream>
#include <thread>

using json = nlohmann::json;

int main()
{
	MNISTInputClass input(1000);

	try {

		Net network(10, 10, 3.0f, &input);
		network.createNewLayer(784,  NeuronType::Input,   LayerType::Input );
		network.createNewLayer(30,   NeuronType::Sigmoid, LayerType::General);
		network.createNewLayer(10,   NeuronType::Sigmoid, LayerType::Output);

		network.calculate();

		json netJSON = network.toJSON();

		std::ofstream o("net.json");
		o << netJSON;
		o.close();

		json readfile;
		std::ifstream i("net.json");
		i >> readfile;

		Net cpy(readfile);
		cpy.addInputClass(&input);

		cpy.calculate();
	}
	catch (const NeuralException& exc) {
		std::cout << exc.getErrorMessage() << '\n';
	}
	catch (...) {
		std::cout << "unspecified exception caught!\n";
	}
	

	std::cout << "\n\nCALCULATIONS FINISHED...\n";
	char cc;
	std::cin >> cc;
    return 0;
}

