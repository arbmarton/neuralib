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

//using json = nlohmann::json;

#define json nlohmann::json

int main()
{
	MNISTInputClass input(1000);
	BitXORInputClass bit(1000);

	try {

		//Net network(&input);
		//network.createNewLayer(784,  NeuronType::Input,   LayerType::Input );
		//network.createNewLayer(30,   NeuronType::Sigmoid, LayerType::General);
		//network.createNewLayer(10,   NeuronType::Sigmoid, LayerType::Output);

		//network.train(10, 10, 3.0f);

		//network.saveAs("net.json");

		//Net cpy(getJson("net.json"));
		//cpy.addInputClass(&input);

		//cpy.train(10, 5, 2.0f);
		

		Net bitnet(&bit);
		bitnet.createNewLayer(2,  NeuronType::Input,   LayerType::Input);
		bitnet.createNewLayer(10,  NeuronType::Sigmoid, LayerType::General);
		bitnet.createNewLayer(2,  NeuronType::Sigmoid, LayerType::Output);

		bitnet.train(1000, 10, 0.1f);

		bitnet.saveAs("XOR.json");
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

