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
#include "NeuralMath.h"

#include "json.hpp"

#include <iostream>

//using json = nlohmann::json;

#define json nlohmann::json


int main()
{
	MNISTInputClass input(1000);
	BitXORInputClass bit(1000);

	Timer timer;

	try {
		//int i = 25;
		//int k = 5;

		//int modifier = 0;
		//if (k % 2 == 0) modifier = 1;

		/*Matrix<float> input(i, i);
		input.fillValue(1);

		Matrix<float> kernel(k, k);
		kernel.fillValue(1);

		Matrix<float> result(input.getCols() - 2 * floor(kernel.getCols() / 2) + modifier, input.getCols() - 2 * floor(kernel.getCols() / 2) + modifier);
		result.fillValue(0);

		convolve(
			input.getData(),  input.getRows(),  input.getCols(),
			kernel.getData(), kernel.getRows(), kernel.getCols(),
			result.getData(), result.getRows(), result.getCols());*/

		/*Matrix<float> delta(20, 20);
		Matrix<float> kernel(5, 5);
		delta.fillValue(1);
		kernel.fillValue(1);

		auto res = fullConvolution(kernel, delta);
		res.print();*/

		//result.print();

		Net conv(&input, CostFunction::CrossEntropy, Regularization::L2);
		conv.createNewLayer(784, NeuronType::Input, LayerType::Input);
		conv.createNewLayer(10,  LayerType::Convolutional, 5, 5);
		conv.createNewLayer(10,  LayerType::Pooling, 2, 2, PoolingMethod::average);
		conv.createNewLayer(100, NeuronType::Sigmoid, LayerType::General);
		conv.createNewLayer(10,  NeuronType::Sigmoid, LayerType::Output);

		conv.printLayerInfo();
		timer.createTimePoint("start");
		conv.testUpdater(5, 10, 3.0f, 0.001f);
		timer.createTimePoint("end");
		timer.printTimeDifferenceSec("start", "end");
		conv.saveAs("0128.json");

		Net cpy(getJson("0128.json"));
		cpy.addInputClass(&input);
		cpy.testUpdater(5, 10, 3.0f, 0.001f);


		//Net network(&input, CostFunction::CrossEntropy, Regularization::L1);
		//network.createNewLayer(784,  NeuronType::Input,   LayerType::Input );
		//network.createNewLayer(100,  NeuronType::Sigmoid, LayerType::General);
		//network.createNewLayer(10,   NeuronType::Sigmoid, LayerType::Output);

		//timer.createTimePoint("before");
		////network.testForward();
		//network.testUpdater(10, 10, 5.0f, 0.001f);
		////network.train(3, 10, 5.0f, 0.001f);
		//timer.createTimePoint("after");

		//timer.printTimeDifferenceSec("before", "after");
		//network.saveAs("0128.json");


		//Net cpy(getJson("0128.json"));
		//cpy.addInputClass(&input);
		//cpy.testUpdater(10, 10, 5.0f, 0.001f);
		//cpy.saveAs("0128.json");
		

		/*Net bitnet(&bit, CostFunction::CrossEntropy, Regularization::L2);
		bitnet.createNewLayer(2,  NeuronType::Input,   LayerType::Input);
		bitnet.createNewLayer(4, NeuronType::Sigmoid, LayerType::General);
		bitnet.createNewLayer(2,  NeuronType::Sigmoid, LayerType::Output);

		bitnet.train(1000, 10, 0.1f, 0.01f);

		bitnet.saveAs("XOR.json");*/
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


//int* arr = new int[25 * 25];
//for (int i = 0; i < 25 * 25; ++i) {
//	arr[i] = 2;
//}
//int* kernel = new int[5 * 5];
//for (int i = 0; i < 5 * 5; ++i) {
//	kernel[i] = 2;
//}
//int* result = new int[21 * 21];
//
//convolve(arr, 25, 25, kernel, 5, 5, result);
//
//for (int i = 0; i < 21; ++i) {
//	for (int j = 0; j < 21; ++j) {
//		std::cout << result[j + i * 21] << ' ';
//	}
//	std::cout << '\n';
//}
//
//delete[] arr;
//delete[] kernel;
//delete[] result;
//
//std::cout << "\n\nCALCULATIONS FINISHED...\n";
//char ccg;
//std::cin >> ccg;
//
//return 0;