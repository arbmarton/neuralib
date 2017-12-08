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
#include "Timer.h"

#include "json.hpp"

#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <vector>
#include <map>

//using json = nlohmann::json;

#define json nlohmann::json


int main()
{
	MNISTInputClass input(60000);
	BitXORInputClass bit(1000);

	Timer timer;

	try {

		/*Matrix<float> mat(100, 1);
		mat.fillValue(1);
		Matrix<float> mat2(100, 100);
		mat2.fillValue(1);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		
		for (int i = 0; i < 10000; ++i) {
			Matrix<float> res = mat2*mat;
		}
		

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

		std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';*/

		Net network(&input, CostFunction::CrossEntropy);
		network.createNewLayer(784,  NeuronType::Input,   LayerType::Input );
		network.createNewLayer(30,   NeuronType::Sigmoid, LayerType::General);
		network.createNewLayer(10,   NeuronType::Sigmoid, LayerType::Output);

		//std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		timer.createTimePoint("before");
		network.train(10, 10, 5.0f);
		timer.createTimePoint("after");

		timer.printTimeDifferenceMs("before", "after");
		//std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		//std::cout << "time: " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << '\n';
		//network.saveAs("net.json");


		//Net cpy(getJson("net.json"));
		//cpy.addInputClass(&input);

		//cpy.train(10, 10, 1.0f);
		

		/*Net bitnet(&bit);
		bitnet.createNewLayer(2,  NeuronType::Input,   LayerType::Input);
		bitnet.createNewLayer(10, NeuronType::Sigmoid, LayerType::General);
		bitnet.createNewLayer(2,  NeuronType::Sigmoid, LayerType::Output);

		bitnet.train(1000, 10, 0.1f);*/

		//bitnet.saveAs("XOR.json");
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

