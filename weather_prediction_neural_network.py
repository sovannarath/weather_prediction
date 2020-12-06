# ***************************************************************** #
# Royal University of Phnom Penh                                    #
# Master of Information Technology and Engineering (MITE)           #
# Cohort 13 (November 2018 - 2020)                                  #
# Supervisor Name: Dr. Srun Sovila                                  #
# Field of Research: Data Science                                   #
#                                                                   #
# Thesis title: Weather Prediction Using Artificial Neural Network  #
# Author Name: Thorn Sovannarath                                    #
# Start Date: 11 November 2020                                      #
# Date of Completion: Not yet defined                               #
# ***************************************************************** #


# *****************************************************************
# Import Libraries
#
# Import necessary required libraries to work with ANN
# *****************************************************************
import random
import math
import csv
import copy

# ******* End of Importing Libraries ******* #
# ****************************************** #

# ******************************************************************
# Define some importants varaible 
# ******************************************************************
# Define weather conditions
weather_conditions = ['sunny', 'cloudy', 'partly cloudy', 'light rain', 'patchy rain possible', 
                      'heavy rain', 'mist', 'fog', 'heavey rain with thunder', 'thunder outbreak possible']
data_column_header = []
# ******* End of Define Variables ******* #
# *************************************** #


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
            total += float(self.inputs[i]) * self.weights[i]
        return total + self.bias

    # This is activation applied the logistic function to the output of the neuron
    def sigmoid(self, total_net_input):
        return 1 / (1 + math.exp(-total_net_input))

    # The error for each neuron is calculated by the Mean Square Error method:
    def calculate_error(self, target_output):
        return 0.5 * (target_output - self.output) ** 2

    # Method calculate partial derivative of total error with respect to actual output (∂E/∂yⱼ)
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

    def __init__(self, num_inputs, num_hidden, num_outputs, num_layers = 1, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs
        self.num_layers = num_layers
        
        self.hidden_layer = [0] * num_layers
        for l in range(num_layers):
            self.hidden_layer[l] = NeuronLayer(num_hidden, hidden_layer_bias)
            self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)

        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # Method used to initialize weights for neuron of each network layer 
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        for l in range(self.num_layers):
            weight_num = 0
            for h in range(len(self.hidden_layer[l].neurons)):
                for i in range(self.num_inputs):
                    if not hidden_layer_weights:
                        self.hidden_layer[l].neurons[h].weights.append(random.random())
                    else:
                        self.hidden_layer[l].neurons[h].weights.append(hidden_layer_weights[weight_num])
                    weight_num += 1

    # Method used to initialize weight for each neuron of output layer
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
            weight_num = 0
            for o in range(len(self.output_layer.neurons)):
                for h in range(len(self.hidden_layer[len(self.hidden_layer)-1].neurons)):
                    if not output_layer_weights:
                        self.output_layer.neurons[o].weights.append(random.random())
                    else:
                        self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                    weight_num += 1

    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        for l in range(self.num_layers):
            print('Hidden Layer: {}'.format(l))
            self.hidden_layer[l].inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    # Method call to calculate feed forward with multilayer neural network
    def feed_forward(self, inputs):
        layer_feed_forward_results = [0] * self.num_layers
        for l in range(self.num_layers):
            if (l == 0) :
                layer_feed_forward_results[l] = self.hidden_layer[l].feed_forward(inputs)
            else:
                layer_feed_forward_results[l] = self.hidden_layer[l].feed_forward(layer_feed_forward_results[l-1])
        return self.output_layer.feed_forward(layer_feed_forward_results[len(self.hidden_layer)-1])

    # Method for back propagation calculation process
    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        self.feed_forward(training_inputs)

        # 1. Output neuron deltas
        # Calculation of ∂E/∂Netⱼ (∂E/∂yⱼ) of each node in output layer
        partial_derivative_errors_with_respect_to_output_neuron_total_net_input = [0] * len(self.output_layer.neurons)
        for o in range(len(self.output_layer.neurons)):
            partial_derivative_errors_with_respect_to_output_neuron_total_net_input[o] = self.output_layer.neurons[o].calculate_partial_derivative_error_with_respect_to_total_net_input(training_outputs[o])

        # 2. Hidden neuron deltas
        # Calculation of ∂E/∂Netⱼ (∂E/∂yⱼ) of each node of each hidden layer
        partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input = [0] * self.num_layers
        for l in range(self.num_layers):
            partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input[l] = [0] * len(self.hidden_layer[l].neurons)
            for h in range(len(self.hidden_layer[l].neurons)):
                # We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
                # dE/dyⱼ = Σ ∂E/∂Netⱼ * ∂z/∂yⱼ = Σ ∂E/∂Netⱼ * wᵢⱼ
                derivative_error_with_respect_to_hidden_neuron_output = 0
                for o in range(len(self.output_layer.neurons)):
                    derivative_error_with_respect_to_hidden_neuron_output += partial_derivative_errors_with_respect_to_output_neuron_total_net_input[o] * self.output_layer.neurons[o].weights[h]
                # ∂E/∂Netⱼ = dE/dyⱼ * ∂Netⱼ/∂
                partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input[l][h] = derivative_error_with_respect_to_hidden_neuron_output * self.hidden_layer[l].neurons[h].calculate_partial_derivative_total_net_input_with_respect_to_input()

        # 3. Update output neuron weights
        for o in range(len(self.output_layer.neurons)):
            for w_ho in range(len(self.output_layer.neurons[o].weights)):
                # ∂Eⱼ/∂wᵢⱼ = ∂E/∂Netⱼ * ∂Netⱼ/∂wᵢⱼ
                partial_derivative_error_with_respect_to_weight = partial_derivative_errors_with_respect_to_output_neuron_total_net_input[o] * self.output_layer.neurons[o].calculate_partial_derivative_total_net_input_with_respect_to_weight(w_ho)
                # Δw = α * ∂Eⱼ/∂wᵢ
                self.output_layer.neurons[o].weights[w_ho] -= self.LEARNING_RATE * partial_derivative_error_with_respect_to_weight

        # 4. Update hidden neuron weights
        for l in range(self.num_layers):
            for h in range(len(self.hidden_layer[l].neurons)):
                for w_ih in range(len(self.hidden_layer[l].neurons[h].weights)):
                    # ∂Eⱼ/∂wᵢ = ∂E/∂Netⱼ * ∂zⱼ/∂wᵢ
                    partial_derivative_error_with_respect_to_weight = partial_derivative_errors_with_respect_to_hidden_neuron_total_net_input[l][h] * float(self.hidden_layer[l].neurons[h].calculate_partial_derivative_total_net_input_with_respect_to_weight(w_ih))
                    # Δw = α * ∂Eⱼ/∂wᵢ
                    self.hidden_layer[l].neurons[h].weights[w_ih] -= self.LEARNING_RATE * partial_derivative_error_with_respect_to_weight

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

# Reading data into array
# 'TestCSVReading.csv'
def readCSV(fileName) :
    original_weather_data = []
    with open(fileName, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tmp_ori_index = 0
        for row in csv_reader:
            if tmp_ori_index == 0 :
                data_column_header.append(row)
            else:
                original_weather_data.append(row)
            tmp_ori_index += 1
    return original_weather_data

def prepareTrainingData(original_weather_data) :
    tmp_origin_data = original_weather_data
    training_data = []
    for data in tmp_origin_data :
        del data[0]
        del data[len(data)-1]
        training_data.append(data)
    return training_data

def prepareTrainingOutput(original_weather_data) :
    tmp_origin_data = original_weather_data
    training_output = []
    for row in tmp_origin_data :
        tmp_w_conditions = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        tmp_w_con_index = 0
        for con in weather_conditions:
            if(row[len(row) - 1].lower() == con.lower()) :
                tmp_w_conditions[tmp_w_con_index] = 0.99
            tmp_w_con_index += 1
        training_output.append(tmp_w_conditions)
    return training_output

# Training process
weather_data = readCSV('TestCSVReading.csv')
training_data = [0] * len(weather_data)
training_data = prepareTrainingData(copy.deepcopy(weather_data))
training_output = [0] * len(weather_data)
training_output = prepareTrainingOutput(copy.deepcopy(weather_data))
input_ele_numbers = len(training_data[0])
output_ele_numbers = len(training_output[0])
#print(len(training_data))
#print(len(training_output))
ann = ArtificialNeuralNetwork(input_ele_numbers, input_ele_numbers, output_ele_numbers, hidden_layer_bias=0.35, output_layer_bias=0.6)
record_count = 0
for t_d in training_data :
    print(record_count+1)
    for i in range(30):
        ann.train(t_d, training_output[record_count])
        print(i, round(ann.calculate_total_error([[t_d, training_output[record_count]]]), 9))
    record_count += 1
    #break
    

# Blog post example:
#ann = ArtificialNeuralNetwork(2, 2, 2, hidden_layer_weights=[0.15, 0.2, 0.25, 0.3], hidden_layer_bias=0.35, output_layer_weights=[0.4, 0.45, 0.5, 0.55], output_layer_bias=0.6)
#for i in range(10000):
#    ann.train([0.05, 0.1], [0.01, 0.99])
#    print(i, round(ann.calculate_total_error([[[0.05, 0.1], [0.01, 0.99]]]), 9))
