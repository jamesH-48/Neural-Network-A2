#####################################################################################################################
#   Assignment 2: Neural Network Programming
#   This is a starter code in Python 3.6 for a 1-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   NeuralNet class init method takes file path as parameter and splits it into train and test part
#         - it assumes that the last column will the label (output) column
#   h - number of neurons in the hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   W_hidden - weight matrix connecting input to hidden layer
#   Wb_hidden - bias matrix for the hidden layer
#   W_output - weight matrix connecting hidden layer to output layer
#   Wb_output - bias matrix connecting hidden layer to output layer
#   deltaOut - delta for output unit (see slides for definition)
#   deltaHidden - delta for hidden unit (see slides for definition)
#   other symbols have self-explanatory meaning
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class NeuralNet:
    def __init__(self, dataFile, activation, state, header=True, h=4):
        self.activation = activation
        #np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h represents the number of neurons in the hidden layer
        raw_input = pd.read_csv(dataFile)
        # TODO: Remember to implement the preprocess method
        processed_data = self.preprocess(raw_input)
        self.train_dataset, self.test_dataset = train_test_split(processed_data, test_size = .2, random_state = state)
        ncols = len(self.train_dataset.columns)
        nrows = len(self.train_dataset.index)
        ncolst = len(self.test_dataset.columns)
        nrowst = len(self.test_dataset.index)

        self.X = self.train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.Xtest = self.test_dataset.iloc[:, 0:(ncolst -1)].values.reshape(nrowst, ncolst-1)
        self.y = self.train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        self.ytest = self.test_dataset.iloc[:, (ncolst - 1)].values.reshape(nrowst, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[1])
        if not isinstance(self.y[0], np.ndarray):
            self.output_layer_size = 1
        else:
            self.output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.W_hidden = 2 * np.random.random((input_layer_size, h)) - 1
        self.Wb_hidden = 2 * np.random.random((1, h)) - 1

        self.W_output = 2 * np.random.random((h, self.output_layer_size)) - 1
        self.Wb_output = np.ones((1, self.output_layer_size))

        self.deltaOut = np.zeros((self.output_layer_size, 1))
        self.deltaHidden = np.zeros((h, 1))
        self.h = h

    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid(self, x)
        if activation == "relu":
            self.__relu(self, x)
        if activation == "tanh":
            self.__tanh(self, x)

    #
    # TODO: Define the derivative function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation):
        if activation == "sigmoid":
            self.__sigmoid_derivative(self, x)
        if activation == "relu":
            self.__relu_derivative(self, x)
        if activation == "tanh":
            self.__tanh_derivative(self, x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __relu(x):
        return x * (x > 0)

    def __relu_derivative(x):
        return 1 * (x > 0)

    def __tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __tanh_derivative(x):
        return 1 - (x * x)

    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):

        return X

    # Below is the training function

    def train(self, max_iterations, learning_rate):
        errGraph = np.zeros((max_iterations,1))
        for iteration in range(max_iterations):
            out = self.forward_pass(self.activation)
            error = 0.5 * np.power((out - self.y), 2)
            errGraph[iteration] = np.sum(error)
            # TODO: I have coded the sigmoid activation, you have to do the rest
            self.backward_pass(out, self.activation)

            update_weight_output = learning_rate * np.dot(self.X_hidden.T, self.deltaOut)
            update_weight_output_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaOut)

            update_weight_hidden = learning_rate * np.dot(self.X.T, self.deltaHidden)
            update_weight_hidden_b = learning_rate * np.dot(np.ones((np.size(self.X, 0), 1)).T, self.deltaHidden)

            self.W_output += update_weight_output
            self.Wb_output += update_weight_output_b
            self.W_hidden += update_weight_hidden
            self.Wb_hidden += update_weight_hidden_b

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_hidden))
        print("The final weight vectors are (starting from input to output layers) \n" + str(self.W_output))

        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_hidden))
        print("The final bias vectors are (starting from input to output layers) \n" + str(self.Wb_output))
        return errGraph
    def forward_pass(self, activation):
        # pass our inputs through our neural network
        in_hidden = np.dot(self.X, self.W_hidden) + self.Wb_hidden
        # Hidden Layer Forward
        if activation == "sigmoid":
            self.X_hidden = self.__sigmoid(in_hidden)
        if activation == "relu":
            self.X_hidden = self.__relu(in_hidden)
        if activation == "tanh":
            self.X_hidden = self.__tanh(in_hidden)
        # Hidden Layer Out dot product with Output Weights
        in_output = np.dot(self.X_hidden, self.W_output) + self.Wb_output
        # Output Layer Forward
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        if activation == "relu":
            out = self.__relu(in_output)
        if activation == "tanh":
            out = self.__tanh(in_output)
        return out

    # Added Function to Propogate Test Data Set through Trained Model
    def forward_test(self, activation, t):
        if t == "train":
            # pass our inputs through our neural network
            in_hidden = np.dot(self.X, self.W_hidden) + self.Wb_hidden
        if t == "test":
            # pass our inputs through our neural network
            in_hidden = np.dot(self.Xtest, self.W_hidden) + self.Wb_hidden
        # Hidden Layer Forward
        if activation == "sigmoid":
            self.X_test_hidden = self.__sigmoid(in_hidden)
        if activation == "relu":
            self.X_test_hidden = self.__relu(in_hidden)
        if activation == "tanh":
            self.X_test_hidden = self.__tanh(in_hidden)
        # Hidden Layer Out dot product with Output Weights
        in_output = np.dot(self.X_test_hidden, self.W_output) + self.Wb_output
        # Output Layer Forward
        if activation == "sigmoid":
            out = self.__sigmoid(in_output)
        if activation == "relu":
            out = self.__relu(in_output)
        if activation == "tanh":
            out = self.__tanh(in_output)
        return out

    def backward_pass(self, out, activation):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation="sigmoid"):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "relu":
            delta_output = (self.y - out) * (self.__relu_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))

        self.deltaOut = delta_output

    def compute_hidden_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__sigmoid_derivative(self.X_hidden))
        if activation == "relu":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__relu_derivative(self.X_hidden))
        if activation == "tanh":
            delta_hidden_layer = (self.deltaOut.dot(self.W_output.T)) * (self.__tanh_derivative(self.X_hidden))

        self.deltaHidden = delta_hidden_layer

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, activation, header = True):
        # TODO: obtain prediction on self.test_dataset
        outputs = self.forward_test(activation, "train")
        print("Train Accuracy Results for ", activation, " activation function:")
        correct = 0
        if len(outputs) == len(self.y):
            for i in range(len(outputs)):
                '''
                print("Model Output for Example ", i, ": ", np.around(outputs[i]))
                print("Full: ", outputs[i])
                print("Actual for Example ", i, ": ", self.y[i])
                '''
                if np.around(outputs[i]) == self.y[i]:
                    correct += 1
            print("Percent Correct: ", (correct/len(outputs))*100, "%")
            print("Error: ", np.sum(0.5 * np.power((outputs - self.y), 2)))
        outputs = self.forward_test(activation, "test")
        print("Test Accuracy Results for ", activation, " activation function:")
        correct = 0
        if len(outputs) == len(self.ytest):
            for i in range(len(outputs)):
                '''
                print("Model Output for Example ", i, ": ", np.around(outputs[i]))
                print("Full: ", outputs[i])
                print("Actual for Example ", i, ": ", self.ytest[i])
                '''
                if np.around(outputs[i]) == self.ytest[i]:
                    correct += 1
            print("Percent Correct: ", (correct/len(outputs))*100, "%")
            print("Error: ", np.sum(0.5 * np.power((outputs - self.ytest), 2)))


if __name__ == "__main__":
    state = 1
    max_iterations = 6000
    LR = .25
    # Train Sigmoid, ReLu, and Tanh Models
    neural_network_sigmoid = NeuralNet("train.csv", "sigmoid", state)
    err_sigmoid = neural_network_sigmoid.train(max_iterations, LR)
    '''
    neural_network_relu = NeuralNet("train.csv", "relu")
    neural_network_relu.train()
    neural_network_tanh = NeuralNet("train.csv", "tanh")
    neural_network_tanh.train()
'''
    # Print Out Test Error for Each Model
    neural_network_sigmoid.predict("sigmoid")
