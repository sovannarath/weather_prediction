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
import os.path
import copy
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time

# ******* End of Importing Libraries ******* #
# ****************************************** #

# ******************************************************************
# Define some importants varaible 
# ******************************************************************
# Define weather conditions
weather_conditions = []
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
        #print("Neuron: ", self.inputs)
        #print("weight: ", self.weights)
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
        #print(self.inputs[index])
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
        #print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):
        outputs = []
        #print("number of Neuron: ", len(self.neurons))
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
        #print(self.hidden_layer[0].neurons[0].weights)

        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    # Method used to initialize weights for neuron of each network layer 
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        for l in range(self.num_layers):
            weight_num = 0
            for h in range(len(self.hidden_layer[l].neurons)):
                if l == 0 :
                    for i in range(self.num_inputs):
                        if not hidden_layer_weights:
                            self.hidden_layer[l].neurons[h].weights.append(random.random())
                        else:
                            self.hidden_layer[l].neurons[h].weights.append(hidden_layer_weights[weight_num])
                        weight_num += 1
                else :
                    for i in range(len(self.hidden_layer[l].neurons)):
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
        #print(inputs)
        layer_feed_forward_results = [0] * self.num_layers
        for l in range(self.num_layers):
            if (l == 0) :
                #print("OK!")
                layer_feed_forward_results[l] = self.hidden_layer[l].feed_forward(inputs)
                #print(layer_feed_forward_results[l])
            else:
                #print("Ok2!: ", l)
                layer_feed_forward_results[l] = self.hidden_layer[l].feed_forward(layer_feed_forward_results[l-1])
                #print(layer_feed_forward_results[l])
        return self.output_layer.feed_forward(layer_feed_forward_results[len(self.hidden_layer)-1])

    # Method for back propagation calculation process
    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs):
        #print(training_inputs)
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
        if(data[len(data)-1] != '') :
            for i in [1,2,3] :
                del data[0]
            del data[3]
            del data[4]
            del data[len(data)-1]
            training_data.append(data)
    return training_data

def prepareTrainingOutput(original_weather_data) :
    tmp_origin_data = original_weather_data
    training_output = []
    for row in tmp_origin_data :
        tmp_w_conditions = []
        #for con_data in weather_conditions:
        #    tmp_w_conditions.append(0.01)
        #tmp_w_con_index = 0
        for con in weather_conditions:
            if(row[len(row) - 1].lower() == con.lower()) :
                tmp_w_conditions.append(0.99)
            else:
                tmp_w_conditions.append(0.01)
            #tmp_w_con_index += 1
        training_output.append(tmp_w_conditions)
    return training_output

def initialWeatherConditions(weather_data):
    tmp_weather_conditions = {}
    for w in weather_data:
        if(w[len(w)-1] != ''):
            tmp_weather_conditions[w[len(w) - 1]] = 1
    for key, value in tmp_weather_conditions.items() :
        weather_conditions.append(key)
    return weather_conditions

def barChart(weather_data) :
    total_records = len(weather_data)
    w_conditions = {}
    for record in weather_data:
        if (record[len(record)-1]) not in ["Fair / Windy", "T-Storm / Windy", "Thunder / Windy", "Partly Cloudy / Windy", "Light Rain Shower / Windy", "Light Rain / Windy", "Mostly Cloudy / Windy", "Heavy T-Storm / Windy"]:
            if (record[len(record)-1]) not in w_conditions:
                w_conditions[record[len(record)-1]] = 1
            else:
                w_conditions[record[len(record)-1]] = w_conditions[record[len(record)-1]] + 1

    data = w_conditions.values()
    conditions = w_conditions.keys()

    data = w_conditions
    names = list(data.keys())
    values = list(data.values())
    fig, axs = plt.subplots()
    axs.bar(names, values)
    fig.suptitle('Categorical Plotting')    
    plt.show()

def pieChart(weather_data):
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    
    total_records = len(weather_data)
    w_conditions = {}
    for record in weather_data:
        if (record[len(record)-1]) not in ["Fair / Windy", "T-Storm / Windy", "Thunder / Windy", "Partly Cloudy / Windy", "Light Rain Shower / Windy", "Light Rain / Windy", "Mostly Cloudy / Windy", "Heavy T-Storm / Windy"]:
            if (record[len(record)-1]) not in w_conditions:
                w_conditions[record[len(record)-1]] = 1
            else:
                w_conditions[record[len(record)-1]] = w_conditions[record[len(record)-1]] + 1

    data = w_conditions.values()
    conditions = w_conditions.keys()

    def myfunc(pct):
        return "{:.1f}%".format(pct)

    wedges, texts, autotexts = ax.pie(data, labels=conditions, autopct=lambda pct: myfunc(pct), textprops=dict(color="w"))
    ax.legend(wedges, conditions, title="Conditions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Weather Phenomena IN A Year (%)")
    plt.show()

def scatterPlot(weather_data) :
    w_records = {}
    for record in weather_data:
        if (record[len(record)-1]) not in ["Fair / Windy", "T-Storm / Windy", "Thunder / Windy", "Partly Cloudy / Windy", "Light Rain Shower / Windy", "Light Rain / Windy", "Mostly Cloudy / Windy", "Heavy T-Storm / Windy"]:
            if (record[len(record)-1]) not in w_records:
                w_records[record[len(record)-1]] = []
                w_records[record[len(record)-1]].append(record)
            else:
                w_records[record[len(record)-1]].append(record)
    
    fig, ax = plt.subplots()
    colors=['dodgerblue', 'orange', 'green', 'blue', 'teal', 'red', 'gold', 
            'yellow', 'saddlebrown', 'crimson', 'grey', 'cyan', 'olivedrab', 'darkorange',
            'lime', 'violet', 'purple', 'navy', 'olive', 'slategray', 'pink']
    c_count = 0
    for w_key in w_records:
        x = [] 
        y = []
        scale = 25.0
        for row in w_records[w_key]:
            x.append(row[2])
            y.append(row[3])
        ax.scatter(x, y, c=colors[c_count], s=scale, label=w_key, alpha=0.8, edgecolors='none')
        c_count = c_count + 1
    ax.legend()
    #ax.grid(True)
    plt.show()

def writeResultToCSV(pre_data, fileName, is_new_round = True):
    # data to be written row-wise in csv file
    data = [pre_data] 
    resultFileName = fileName
    fileCount = 1
    f = resultFileName + '.csv'
    if(is_new_round == True):
        tmp_file_name = fileName + '.csv'
        while os.path.isfile(tmp_file_name) :
            tmp_file_name = fileName + str(fileCount) + '.csv'
            fileCount = fileCount + 1
        f = tmp_file_name
    else: 
        f = fileName

    # opening the csv file in 'a+' mode 
    file = open(f, 'a+', newline ='')
  
    # writing the data into the file 
    with file:     
        write = csv.writer(file) 
        write.writerows(data)
    
    return f 

def yearlyWeatherCondition(weather_data):
    originGraph = {}
    totalRecordEachMonth = {}
    totalEachYearRecord = {}
    for data in weather_data: 
        dateObj = datetime.datetime.strptime(data[1], '%m/%d/%Y').date()
        if data[len(data)-1] == '':
            continue
        if dateObj.year not in originGraph.keys() : 
            originGraph[dateObj.year] = {}
            totalRecordEachMonth[dateObj.year] = {}
            totalEachYearRecord[dateObj.year] = 0
        if dateObj.month not in originGraph[dateObj.year].keys() :
            originGraph[dateObj.year][dateObj.month] = {}
            totalRecordEachMonth[dateObj.year][dateObj.month] = 0 
            for weather_condition in weather_conditions :
                originGraph[dateObj.year][dateObj.month][weather_condition] = 0
        originGraph[dateObj.year][dateObj.month][data[len(data)-1]] = originGraph[dateObj.year][dateObj.month][data[len(data)-1]] + 1
        totalRecordEachMonth[dateObj.year][dateObj.month] = totalRecordEachMonth[dateObj.year][dateObj.month] + 1
        totalEachYearRecord[dateObj.year] = totalEachYearRecord[dateObj.year] + 1
    

    yearlyConditionData = {}
    tmp_originGrap = copy.deepcopy(originGraph)
    for t_yIdx, t_yVal in tmp_originGrap.items() :
        if t_yIdx not in yearlyConditionData.keys():
            yearlyConditionData[t_yIdx] = {}
        for condition in weather_conditions:
            for t_mIdx, t_mVal in t_yVal.items():
                if condition not in yearlyConditionData[t_yIdx].keys():
                    yearlyConditionData[t_yIdx][condition] = 0
                yearlyConditionData[t_yIdx][condition] = yearlyConditionData[t_yIdx][condition] + t_mVal[condition]
    
    yearlyConditionDataAsPercentage = copy.deepcopy(yearlyConditionData)
    for yl_idx, yl_val in yearlyConditionDataAsPercentage.items() :
        fileNamePeiChart = 'YealyOriginGraph' + str(yl_idx)
        is_new_round = True
        for con_idx, con_val in yl_val.items() :
            yearlyConditionDataAsPercentage[yl_idx][con_idx] = (yearlyConditionDataAsPercentage[yl_idx][con_idx] / totalEachYearRecord[yl_idx]) * 100
            d = []
            d.insert(0, con_idx)
            d.insert(1, yearlyConditionDataAsPercentage[yl_idx][con_idx])
            fileNamePeiChart = writeResultToCSV(d, fileNamePeiChart, is_new_round)
            is_new_round = False

def monthlyWeatherConditon(weather_data):
    originGraph = {}
    totalRecordEachMonth = {}
    totalEachYearRecord = {}
    for data in weather_data: 
        dateObj = datetime.datetime.strptime(data[1], '%m/%d/%Y').date()
        if data[len(data)-1] == '':
            continue
        if dateObj.year not in originGraph.keys() : 
            originGraph[dateObj.year] = {}
            totalRecordEachMonth[dateObj.year] = {}
            totalEachYearRecord[dateObj.year] = 0
        if dateObj.month not in originGraph[dateObj.year].keys() :
            originGraph[dateObj.year][dateObj.month] = {}
            totalRecordEachMonth[dateObj.year][dateObj.month] = 0 
            for weather_condition in weather_conditions :
                originGraph[dateObj.year][dateObj.month][weather_condition] = 0
        originGraph[dateObj.year][dateObj.month][data[len(data)-1]] = originGraph[dateObj.year][dateObj.month][data[len(data)-1]] + 1
        totalRecordEachMonth[dateObj.year][dateObj.month] = totalRecordEachMonth[dateObj.year][dateObj.month] + 1
        totalEachYearRecord[dateObj.year] = totalEachYearRecord[dateObj.year] + 1
            
    # Monthly weather event
    originGraphAsPercentage = copy.deepcopy(originGraph)
    for yIdxP, yValP in originGraphAsPercentage.items() :
        for mIdxP, mValP in yValP.items() :
            for idxP, valP in mValP.items():
                originGraphAsPercentage[yIdxP][mIdxP][idxP] = (originGraphAsPercentage[yIdxP][mIdxP][idxP] / totalRecordEachMonth[yIdxP][mIdxP]) * 100 

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    conditions = {}
    for yIdxP, yValP in originGraphAsPercentage.items() :
        conditions[yIdxP] = {}
        for con in weather_conditions:
            conditions[yIdxP][con] = []
            for mIdxP, mValP in yValP.items() :
                conditions[yIdxP][con].append(mValP[con])

    err_std = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    width = 0.35  # the width of the bars: can also be len(x) sequence

    for c, v in conditions.items():
        fig, ax = plt.subplots()
        fileName = 'barGraph' + str(c)
        is_new_round = True
        for i, a in v.items():
            d = copy.deepcopy(a)
            d.insert(0, i)
            #print(i,d)
            fileName = writeResultToCSV(d, fileName, is_new_round)
            is_new_round = False
            ax.bar(labels, a, width, yerr=err_std, label=i)
        ax.set_ylabel('Percentage')
        ax.set_title('Percentage of Weather Event in ' + str(c))
        ax.legend()
    plt.show()
    #End monthly weather event    

# Training process
def trainingProcess():
    start_time = time.time()
    weather_data = readCSV('Phnom_Penh_Weather_Data - Sheet1.csv')
    tmp_weather_data = copy.deepcopy(weather_data)
    conditions = initialWeatherConditions(copy.deepcopy(weather_data))
    tmp_d_weather_data = copy.deepcopy(weather_data)
    training_data = [0] * len(weather_data)
    training_data = prepareTrainingData(copy.deepcopy(weather_data))
    training_output = [0] * len(weather_data)
    training_output = prepareTrainingOutput(copy.deepcopy(weather_data))
    
    input_ele_numbers = len(training_data[0])
    output_ele_numbers = len(training_output[0])

    # Round 1 epoch 40, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], output 25, 
        # - hid* 6  L1 (Ok, R&RD)
        # - hid* 6*2 L1 (Ok, R1&RD1)
        # - hid* 6*3 L1 (OK, R2&RD2)
        # - hid* 6*4 L1 (Ok, R3&RD3)
        # - hid* 6 L2 (OK, R4&RD4)
        # - hid* 6*2 L2 (Ok, R5&RD5)
        # - hid* 6*3 L2 (OK, R6&RD6)
    # Round 2 epoch 80, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], output 25
        # - hid* 6 L1 (Ok, R7&RD7)
        # - hid* 6*2 L1 (Ok, R8&RD8)
        # - hid* 6*3 L1 (R9&RD9)
        # - hid* 6 L2 (R10&RD10)
        # - hid* 6*2 L2 (R11&RD11)
        # - hid* 6*3 L2 (R12&RD12)
    # Round 3 epoch 160, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], ouotput 25
        # - hid* 6 L1 (R13&RD13)
        # - hid* 6*2 L1 (R14&RD14)
        # - hid* 6*3 L1 (R15&RD15)
        # - hid* 6 L2 (R16&RD16)
        # - hid* 6*2 L2 (R17&RD17)
        # - hid* 6*3 L2 (R18&RD18)
    # Round 4 epoch 500, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], ouotput 25
        # - hid* 6 L1 (R19&RD19)
        # - hid* 6*2 (R20&RD20)
        # - hid* 6*3 (R21&RD21)
        # - hid* 6 L2 (R22&RD22)
        # - hid* 6*2 L2 (R23)
        # - hid* 6*3 L2 (R24)
    # Round 5 epoch 1000, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], ouotput 25
        # - hid* 6 L1 R25
        # - hid* 6*2 R26
        # - hid* 6*3 R27
        # - hid* 6 L2 R28
        # - hid* 6*2 L2 R29
        # - hid* 6*3 L2 R30
    ann = ArtificialNeuralNetwork(input_ele_numbers, input_ele_numbers*3, output_ele_numbers, num_layers=2, hidden_layer_bias=0.35, output_layer_bias=0.6)
    
    record_count = 0
    is_new_round = True
    is_d_new_round = True
    detailFile = 'ResultsDetail'
    resultFile = 'Results'
    for t_d in training_data :
        for i in range(1000) :
            ann.train(t_d, training_output[record_count])
            total_error = ann.calculate_total_error([[t_d, training_output[record_count]]])
            training_record_cpy = copy.deepcopy(tmp_weather_data[record_count])
            training_record_cpy.append(total_error)
            detailFile = writeResultToCSV(training_record_cpy, detailFile, is_d_new_round)
            is_d_new_round = False
            print(total_error)
            if(total_error <= 0.01) :
                break
        tmp_weather_data[record_count].append(total_error)
        resultFile = writeResultToCSV(copy.deepcopy(tmp_weather_data[record_count]), resultFile, is_new_round)
        is_new_round = False
        record_count += 1
    time.sleep(1)

    end_time = time.time()
    execution_time = []
    execution_time.append(start_time)
    execution_time.append(end_time)
    writeResultToCSV(execution_time, detailFile, is_d_new_round)
    

# Method Training
trainingProcess()
