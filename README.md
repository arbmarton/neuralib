# neuralib (work in progress)

This repository contains a project which aims to help the user to create simple, but costumizable neural networks. The main design goal is to separate the logic of the neural network from the task its currently doing. (E.g.: if you go from training a network to recognize letters to recognizing faces you dont have to modifiy a single line in the code which describes the behaviour of the network - you just have to write the functions that put the data in the network, and those which get it out.)

You can save the actual state of you network in JSON format to the disk. This enables you to train a network at a certain time and go about your business. When you need your trained network to do a task you just have to load it, and tell it to work its magic on the new data set.
This project uses https://nlohmann.github.io/json/.
