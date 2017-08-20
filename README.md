# MNIST-NeuralNetwork
A simple neural network designed to train with MNIST csv datas to recognize handwritten numbers.
MNIST data files are not included in repository since the files are too big
Download training dataset here: https://pjreddie.com/media/files/mnist_train.csv
Download  testing dataset (to evaluate performance, it has to be a different dataset): https://pjreddie.com/media/files/mnist_test.csv
Put both files as the same directory as the python file

This neural network has 1 input layer, 1 hidden layer and a output layer
Input layer has 784 nodes as there are 28x28 pixels as input from the image
Hidden layer has 200 nodes, and output has 10 nodes which are the result of digit 0-9
It has an average performance of 97% accuracy
Input takes a 28x28 png image file, repository included two examples, switch the filename to test different files

Use python3 to run handwritten_numbers_nn.py
required to install numpy and matplotlib
Or install jupyter to open NN.ipynb in interactive python
When run, it will train the neural network first, and run performance test to evaluate accuracy
then classify the result of the input image file
Training should take around 10 to 20 minutes depend on your machine
