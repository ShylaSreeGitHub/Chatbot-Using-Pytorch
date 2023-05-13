import torch
import torch.nn as nn
#Pytorch library is a machine learning library used for NLP purposes.
#It is the feedforward neural network with 2 hidden layers
#Bag of words is the input for the neural network
#input size is the number of patterns
#output size is the number of classes.

#here 3 linear layers are created with help of Linear function
#Linear function takes input and output as parameters.
#Hidden size is variable whereas input size and output size is constant.
#ReLu() is used in deep neural networks for purpose of activation.
#Also called as piecewise linear functon.
#Output is either positive or zero for ReLu() function.
class NeuralNetwork(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, num_output_classes):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_layer_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.l3 = nn.Linear(hidden_layer_size, num_output_classes)
        self.relu = nn.ReLU()
#Feed forward neural network is uni directions, i.e., information is sent to destination
#Unlike, back propagation, it does not form the cycle.
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out