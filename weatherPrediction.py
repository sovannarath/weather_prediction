# Import library
from os import system, name
from time import sleep
import helper
import copy
import artificialNeuralNetwork

# Defining nescessary variables
exit = False
weatherConditions = []
weatherData = []
crrentFileUpload = ""

def uploadDataFile():
    print('\n')
    global weatherData
    global crrentFileUpload
    fileName = input("Please enter file path location: ")
    crrentFileUpload = fileName
    weatherData = helper.readCSV(fileName)
    for data in weatherData :
        print(data)

    sleep(1)
    _ = system('clear')
    print('\n')
    print('Upload success!')
    print('\n')
    sleep(1)
    _ = system('clear')

def trainingNeuralNetwork():
    global weatherData
    if len(weatherData) != 0 :
        print('\n')
        print("Training process starting...")
        print("=======================================")
        sleep(0.5)
        print('-Initial weather condition processing...')
        conditions = helper.initialWeatherConditions(copy.deepcopy(weatherData))
        print('-> Success')
        sleep(0.5)
        print('-Preparing training input data...')
        training_data = [0] * len(weatherData)
        training_data = helper.prepareTrainingData(copy.deepcopy(weatherData))
        print('-> Success')
        sleep(0.5)
        print('-Preparing training output data...')
        training_output = [0] * len(weatherData)
        training_output = helper.prepareTrainingOutput(copy.deepcopy(weatherData), copy.deepcopy(conditions))
        print('-> Success')
        sleep(0.5)
        print('\n')
        print('-Initial neural network processing...')
        input_ele_numbers = len(training_data[0])
        output_ele_numbers = len(training_output[0])
        ann = artificialNeuralNetwork.ArtificialNeuralNetwork(input_ele_numbers, input_ele_numbers*3, output_ele_numbers, num_layers=1, hidden_layer_bias=0.35, output_layer_bias=0.6)
        print('-> Success')
        sleep(0.5)

        record_count = 0
        for t_d in training_data :
            for i in range(80) :
                ann.train(t_d, training_output[record_count])
                total_error = ann.calculate_total_error([[t_d, training_output[record_count]]])
                print(total_error)
                if(total_error <= 0.01) :
                    break
            record_count += 1

        _ = system('clear')
        print('\n')
        print('Training data success!')
        print('\n')
        sleep(2)
        _ = system('clear')
    else :
        _ = system('clear')
        print('\n')
        print('Please upload data first!')
        sleep(2)
        _ = system('clear')

def trainingResult():
    print('\n')
    print("Training Result")
    print('\n')

def exitProgram():
    global exit
    exit = True

while exit == False :
    print('=================================')
    print('WEATHER PREDICTION SYSTEM PROGRAM')
    print('=================================')
    print('1. Upload data file')
    print('2. Train Neural Network')
    print('3. Result')
    print('4. Exit')
    print('=================================')
    case = input('Please choose an option: ')
    
    if case not in ['1', '2', '3', '4'] :
        sleep(1)
        _ = system('clear')
        print('\n')
        print('Invalid option!')
        print('\n')
        sleep(1)
        _ = system('clear')
        continue

    if int(case) == 1 :
        sleep(1)
        _ = system('clear')
        uploadDataFile()
    elif int(case) == 2 : 
        sleep(1)
        _ = system('clear')
        trainingNeuralNetwork()
    elif int(case) == 3 :
        sleep(1) 
        _ = system('clear')
        trainingResult()
    elif int(case) == 4 :
        sleep(1)
        _ = system('clear')
        exitProgram()

print('\n')
print('Goodbye, see you next time!')
print('\n')
sleep(1)
_ = system('clear')

"""
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
"""