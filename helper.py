import csv

# Reading data into array
def readCSV(fileName) :
    original_weather_data = []
    with open(fileName, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        tmp_ori_index = 0
        for row in csv_reader:
            original_weather_data.append(row)
            tmp_ori_index += 1
    return original_weather_data

# Preparation data for training
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

# Preparation output node
def prepareTrainingOutput(original_weather_data, weather_conditions) :
    tmp_origin_data = original_weather_data
    training_output = []
    for row in tmp_origin_data :
        tmp_w_conditions = []
        for con in weather_conditions:
            if(row[len(row) - 1].lower() == con.lower()) :
                tmp_w_conditions.append(0.99)
            else:
                tmp_w_conditions.append(0.01)
        training_output.append(tmp_w_conditions)
    return training_output

def initialWeatherConditions(weather_data):
    weather_conditions = []
    tmp_weather_conditions = {}
    for w in weather_data:
        if(w[len(w)-1] != ''):
            tmp_weather_conditions[w[len(w) - 1]] = 1
    for key, value in tmp_weather_conditions.items() :
        weather_conditions.append(key)
    return weather_conditions

def writeResultToCSV(pre_data, fileName):
    # data to be written row-wise in csv file
    data = [pre_data] 
    """
    resultFileName = fileName
    fileCount = 1
    f = resultFileName
    if(is_new_round == True):
        tmp_file_name = fileName + '.csv'
        while os.path.isfile(tmp_file_name) :
            tmp_file_name = fileName + str(fileCount) + '.csv'
            fileCount = fileCount + 1
        f = tmp_file_name
    else: 
        f = fileName
    """
    # opening the csv file in 'a+' mode 
    file = open(fileName, 'a+', newline ='')
  
    # writing the data into the file 
    with file:     
        write = csv.writer(file) 
        write.writerows(data)
    
    #return f 

def readingResultSet(fileName):
    dataSets = readCSV(fileName)

    tenPercentErr = 0
    for data in dataSets :
        if float(data[len(data)-1]) <= 0.01 :
            tenPercentErr += 1
    print(len(dataSets), tenPercentErr)
