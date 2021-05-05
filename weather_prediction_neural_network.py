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
#trainingProcess()


    # Round 1 epoch 40, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], output 25, 
        # - hid* 6  L1 (Ok, R&RD) 95.40%
        # - hid* 6*2 L1 (Ok, R1&RD1) 99.74%
        # - hid* 6*3 L1 (OK, R2&RD2) 98.42%
        # - hid* 6*4 L1 (Ok, R3&RD3) 74.49%
        # - hid* 6 L2 (OK, R4&RD4) 92.98%
        # - hid* 6*2 L2 (Ok, R5&RD5) 99.74%
        # - hid* 6*3 L2 (OK, R6&RD6) 98.00%
    # Round 2 epoch 80, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], output 25
        # - hid* 6 L1 (Ok, R7&RD7) 99.65%
        # - hid* 6*2 L1 (Ok, R8&RD8) 99.97%
        # - hid* 6*3 L1 (R9&RD9) 98.99%
        # - hid* 6 L2 (R10&RD10) 96.72%
        # - hid* 6*2 L2 (R11&RD11) 99.97%
        # - hid* 6*3 L2 (R12&RD12) 98.49%
    # Round 3 epoch 160, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], ouotput 25
        # - hid* 6 L1 (R13&RD13) 99.95%
        # - hid* 6*2 L1 (R14&RD14) 99.98%
        # - hid* 6*3 L1 (R15&RD15) 99.92%
        # - hid* 6 L2 (R16&RD16) 98.84%
        # - hid* 6*2 L2 (R17&RD17) 99.98%
        # - hid* 6*3 L2 (R18&RD18) 99.82%
    # Round 4 epoch 500, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], ouotput 25
        # - hid* 6 L1 (R19&RD19) 100%
        # - hid* 6*2 (R20&RD20) 99.97%
        # - hid* 6*3 (R21&RD21) 99.91%
        # - hid* 6 L2 (R22&RD22) 100%
        # - hid* 6*2 L2 (R23) 99.99%
        # - hid* 6*3 L2 (R24) 99.90%
    # Round 5 epoch 1000, input 6, hidden [6, 6*2, 6*3, 6*4, 6*5], layer[1, 2], ouotput 25
        # - hid* 6 L1 R25 99.99%
        # - hid* 6*2 R26 100%
        # - hid* 6*3 R27 99.90%
        # - hid* 6 L2 R28 100%
        # - hid* 6*2 L2 R29 100%
        # - hid* 6*3 L2 R30 99.91%