#CustomNeuralNetwork.py

import numpy as np
import pandas as pd

#Activation function: f(x) = 1/(1+e^(-x))
def sigmoid(x):
	return 1/(1+np.exp(-x))


#Derivate of sigmoid function
def deriv_sigmoid(x):
  fx = sigmoid(x)
  return fx * (1 - fx)


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
	def __init__(self):
		# Weights
	    self.w1 = np.random.normal()
	    self.w2 = np.random.normal()
	    self.w3 = np.random.normal()
	    self.w4 = np.random.normal()
	    self.w5 = np.random.normal()
	    self.w6 = np.random.normal()

	    # Biases
	    self.b1 = np.random.normal()
	    self.b2 = np.random.normal()
	    self.b3 = np.random.normal()

	def feedforward(self, x):
		# x is a numpy array with 2 elements.
		h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
		h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
		o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
		return o1


	def train(self, data, all_y_trues):
	    '''
	    - data is a (n x 2) numpy array, n = # of samples in the dataset.
	    - all_y_trues is a numpy array with n elements.
	      Elements in all_y_trues correspond to those in data.
	    '''
	    learn_rate = 0.1
	    epochs = 1000 # number of times to loop through the entire dataset

	    for epoch in range(epochs):
	      for x, y_true in zip(data, all_y_trues):
	        # --- Do a feedforward (we'll need these values later)
	        # print(x)
	        # print(y_true)
	        # raise SystemExit
	        sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
	        h1 = sigmoid(sum_h1)

	        sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
	        h2 = sigmoid(sum_h2)

	        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
	        o1 = sigmoid(sum_o1)
	        y_pred = o1

	        # --- Calculate partial derivatives.
	        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
	        d_L_d_ypred = -2 * (y_true - y_pred)

	        # Neuron o1
	        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
	        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
	        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

	        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
	        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

	        # Neuron h1
	        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
	        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
	        d_h1_d_b1 = deriv_sigmoid(sum_h1)

	        # Neuron h2
	        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
	        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
	        d_h2_d_b2 = deriv_sigmoid(sum_h2)

	        # --- Update weights and biases
	        # Neuron h1
	        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
	        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
	        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

	        # Neuron h2
	        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
	        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
	        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

	        # Neuron o1
	        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
	        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
	        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

	      # --- Calculate total loss at the end of each epoch
	      if epoch % 10 == 0:
	        y_preds = np.apply_along_axis(self.feedforward, 1, data)
	        loss = mse_loss(all_y_trues, y_preds)
	        print("Epoch %d loss: %.3f" % (epoch, loss))



#Mean Squared Error function
def mse_loss(y_true, y_pred):
	# print(y_true)
	# print(y_pred)
	# raise SystemExit
	return ((y_true - y_pred) ** 2).mean()

# y_true = np.array([1, 0, 0, 1])
# y_pred = np.array([0, 0, 0, 0])

# print(mse_loss(y_true, y_pred)) # 0.5

# weights = np.array([0,1])
# bias = 0

# x = np.array([2,3])

# network = NeuralNetwork(weights, bias)

# ans = network.feed_forward(x)
# print(ans)


df = pd.read_csv('data/data.csv')
# print(df)

df.Weight = df.Weight-df.Weight.mean().astype(int)
df.Height = df.Height-df.Height.mean().astype(int)
# print(df)


inputData = df[["Weight", "Height"]]
# print(inputData)
targetData = df[["Gender"]]
# print(targetData)
# raise SystemExit
targetData.loc[targetData["Gender"] == 'F', 'Gender'] = 0
targetData.loc[targetData["Gender"] == 'M', 'Gender'] = 1
# print(targetData)


# Train our neural network!

inputData = np.array([inputData.Weight, inputData.Height]).transpose()
targetData = np.array(targetData.Gender).transpose()

# print("\n\n")
# print(inputData.transpose())
# print(targetData)
# raise SystemExit
network = NeuralNetwork()
network.train(inputData, targetData)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M






