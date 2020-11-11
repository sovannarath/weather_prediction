# ***************************************************************** #
# Royal University of Phnom Penh                                    #
# Master of Information Technology and Engineering                  #
# Cohort 13 (November 2018 - 2020)                                  #
#                                                                   #
# Thesis title: Weather Prediction Using Artificial Neural Network  #
# Author Name: Thorn Sovannarath                                    #
# Start Date: 11 November 2020                                      #
# Date of complettion: 31 December 2020                             #
# ***************************************************************** #


# *****************************************************************
# Import Library
#
# Import necessary required libraries to work with ANN
# *****************************************************************
import random
import math


# *****************************************************************
# Start of Neuron Class
#
# Class represented each neuron node of artificial neural network.
# *****************************************************************
class Neuron:

    def __init__(self, bias):
        self.bias = bias
        self.weights = []

    # Method to calculate final output of each neuron node
    # This method used an activation function called sigmoid function
    def calculate_output(self, inputs):
        self.inputs = inputs
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    # Method to calculate weighted sum of each neuron node
    # Following the formular yᵢ = Σxᵢwᵢ + bias
    def calculate_total_net_input(self):
        total = 0
        for i in range(len(self.inputs)):
            total += self.inputs[i] * self.weights[i]
        return total + self.bias

    # This is activation applied the logistic function to the output of the neuron
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # Method calculate partial derivative of total error with respect to actual output (∂E/∂yⱼ)
    # The partial derivate of the error with respect to actual output then is calculated by:
    # = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
    # ∂E/∂yⱼ = -(target output - actual output)
    # Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ
    # => ∂E/∂yⱼ = -(tⱼ - yⱼ)
    def calculate_partial_derivative_error_with_respect_to_output(self, target_output):
        return -(target_output - self.output)

    # Method calculate partial derivative of actual output with respect to total weighted sum (dyⱼ/dNetⱼ)
    # The actual output of the neuron is calculate using sigmoid logistic function:
    # yⱼ = 1 / (1 + e^(-Netⱼ))
    # The derivative (not partial derivative since there is only one variable) of the output then is:
    # dyⱼ/dNetⱼ = yⱼ * (1 - yⱼ)
    def calculate_partial_derivative_total_net_input_with_respect_to_input(self):
        return self.output * (1 - self.output)

    # Partial derivative of weighted sum (net) with respect to wieght (∂Netⱼ/∂wᵢ)
    # The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
    # = Netⱼ = x₁w₁ + x₂w₂ ...
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂Netⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def calculate_partial_derivative_total_net_input_with_respect_to_weight(self, index):
        return self.inputs[index]

    # Method calculate partial derivative of total errors with respect to weight of i (∂E/∂zⱼ) 
    # This value is also known as the delta (δ)
    # δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    def calculate_partial_derivative_error_with_respect_to_total_net_input(self, target_output):
        return self.calculate_partial_derivative_error_with_respect_to_output(target_output) * self.calculate_partial_derivative_total_net_input_with_respect_to_input();

# ******* End of Neuron Class ******* #
# *********************************** #


# *****************************************************************
# Start Neuron Layer Class
#
# Class represented each layer of artificial neural network. 
# There are many nodes in each layer.
# *****************************************************************
class NeuronLayer:
    def __init__(self, num_neurons, bias):

        # Each neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons):
            self.neurons.append(Neuron(self.bias))

    def inspect(self):
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

# ******* End of Neuron Layer Class ******* #
# ***************************************** #


# *******************************************************************
# Start of Aritificial Neural Netowrk Class
#
# This class represente a whole nueral network tha contain all layer
# of node and essential methods in calculating ANN.
# *******************************************************************
class ArtificialNeuralNetwork:
    LEARNING_RATE = 0.5

    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights:
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights:
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs):
        hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
        return self.output_layer.feed_forward(hidden_layer_outputs)

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        pd_errors_wrt_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):

            # ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_pd_error_wrt_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        pd_errors_wrt_hidden_neuron_total_net_input = [0] * len(self.hidden_layer.neurons)
        for h in range(len(self.hidden_layer.neurons)):

            # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
            # dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
            d_error_wrt_hidden_neuron_output = 0
            for o in range(len(self.output_layer.neurons)):
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]

            # ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[h] = d_error_wrt_hidden_neuron_output * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):

                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_pd_total_net_input_wrt_weight(w_ho)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * pd_error_wrt_weight

        # 4. Update hidden neuron weights
        for h in range(len(self.hidden_layer.neurons)):
            for w_ih in range(len(self.hidden_layer.neurons[h].weights)):

                # ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[h] * self.hidden_layer.neurons[h].calculate_pd_total_net_input_wrt_weight(w_ih)

                # Δw = α * ∂Eⱼ/∂wᵢ
                self.hidden_layer.neurons[h].weights[w_ih] -= self.LEARNING_RATE * pd_error_wrt_weight

    def calculate_total_error(self, training_sets):
        total_error = 0
        for t in range(len(training_sets)):
            training_inputs, training_outputs = training_sets[t]
            self.feed_forward(training_inputs)
            for o in range(len(training_outputs)):
                total_error += self.output_layer.neurons[o].calculate_error(training_outputs[o])
        return total_error

# ******* End of Artificial Neural Network Class ******* #
# ****************************************************** #


# Blog post example:

#ann = ArtificialNeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
#for i in range(10000):
#    nn.train([0.05, 0.1], [0.01, 0.99])
#    print(i, round(nn.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))

# XOR example:

# training_sets = [
#     [[0, 0], [0]],
#     [[0, 1], [1]],
#     [[1, 0], [1]],
#     [[1, 1], [0]]
# ]

# nn = NeuralNetwork(len(training_sets[0][0]), 5, len(training_sets[0][1]))
# for i in range(10000):
#     training_inputs, training_outputs = random.choice(training_sets)
#     nn.train(training_inputs, training_outputs)
#     print(i, nn.calculate_total_error(training_sets))
