# MNIST-NeuralNetwork
A simple neural network designed to train with MNIST csv datas to recognize handwritten numbers.
MNIST data files are not included in repository
download training dataset here: https://pjreddie.com/media/files/mnist_train.csv
download  testing dataset (to evaluate performance, it has to be a different dataset): https://pjreddie.com/media/files/mnist_test.csv

This neural network has 1 input layer, 1 hidden layer and a output layer
Input layer has 784 nodes as there are 28x28 pixels as input from the image
Hidden layer has 200 nodes, and output has 10 nodes which are the result of digit 0-9
It has an average performance of 97% accuracy
Input takes a 28x28 png image file, repository included two examples, switch the filename to test different files

Use python3 to run, 
required to install numpy and matplotlib
When run, it will train the neural network first, and run performance test to evaluate accuracy
then classify the result of the input image file
Training should take around 10 to 20 minutes depend on your machine
