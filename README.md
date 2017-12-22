# neuralib (work in progress)

This repository contains a project written in C++ which aims to help the user to create simple, but costumizable neural networks. The main design goal is to separate the logic of the neural network from the task its currently doing. (E.g.: if you go from training a network to recognize letters to recognizing faces you dont have to modifiy a single line in the code which describes the behaviour and logic of the network - you just have to write the functions that put the data in the network, and those which get it out.)

You can save the actual state of you network in JSON format to the disk. This enables you to train a network at a certain time and go about your business. When you need your trained network to do a task you just have to load it, and tell it to work its magic on the new data set.

Basic usage:
```c++

// create an InputClass object - in this case it reads 10000 MNIST images to memory
MNISTInputClass input(10000);

// create a Net object, which uses the MNIST images read earlier, and define the cost function
Net network(&input, CostFunction::CrossEntropy);

// add layers to the network
network.createNewLayer(784,  NeuronType::Input,   LayerType::Input );
network.createNewLayer(30,   NeuronType::Sigmoid, LayerType::General);
network.createNewLayer(10,   NeuronType::Sigmoid, LayerType::Output);

// train the network: 100 epochs, 10 minibatchsize, 5.0f eta
network.train(100, 10, 5.0f);

// save the result to the disk in json format
network.saveAs("net.json");



BitXORInputClass bit(1000);

Net bitnet(&bit);
bitnet.createNewLayer(2,  NeuronType::Input,   LayerType::Input);
bitnet.createNewLayer(10, NeuronType::Sigmoid, LayerType::General);
bitnet.createNewLayer(2,  NeuronType::Sigmoid, LayerType::Output);

bitnet.train(1000, 10, 0.1f);

bitnet.saveAs("XOR.json");

```

This project uses:
- https://nlohmann.github.io/json/
- https://github.com/progschj/ThreadPool

Current TODO list:
 - convolutional layers
 - implementing more math stuff for better computation
 - creating more example inputclasses besides the MNIST one
 - CPU based multithreading somehow
 - CUDA support
 - Qt based GUI (?)
