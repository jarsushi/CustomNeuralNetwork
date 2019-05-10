#CustomNeuralNetwork.py

import numpy as np

#Activation function: f(x) = 1/(1+e^(-x))
def sigmoid(x):
	return 1/(1+np.exp(-x))


class Neuron:
	def __init__(self, weights, bias):
		self.weights = weights
		self.bias = bias

	def feed_forward(self, inputs):
		total = np.dot(self.weights, inputs) + self.bias
		return sigmoid(total)


class NeuralNetwork:
	'''
	  A neural network with:
	    - 2 inputs
	    - a hidden layer with 2 neurons (h1, h2)
	    - an output layer with 1 neuron (o1)
	  Each neuron has the same weights and bias
  	'''
	def __init__(self, weights, bias):
		self.h1 = Neuron(weights, bias)
		self.h2 = Neuron(weights, bias)
		self.o1 = Neuron(weights, bias)

	def feed_forward(self, inputs):
		out_h1 = self.h1.feed_forward(inputs)
		out_h2 = self.h2.feed_forward(inputs)
		out_o1 = self.o1.feed_forward(np.array([out_h1, out_h2]))

		return out_o1

weights = np.array([0,1])
bias = 0

x = np.array([2,3])

network = NeuralNetwork(weights, bias)

ans = network.feed_forward(x)
print(ans)