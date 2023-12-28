# 1. Input library and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from constant import features, target, dataName, modelName

data = pd.read_csv('./data/' + dataName + '.csv')

# 2. Prepare data for training AI & Clean data
feature_cols = features + target

dataAI = data[feature_cols]
dataAI.dropna(inplace = True)
dataAI.dropna(axis = 0)
# dataAI.info()
numOfRecord = len(dataAI)
print(f'\n\n\n{numOfRecord}\n\n\n')
split = int(round(numOfRecord * 0.7, 0))
WS = 12
featureNumber = len(feature_cols)
featureNumber = 5

# 3. Split data into training set and test set
trainingSet = dataAI.iloc[:split, 0:featureNumber].values
testSet = dataAI.iloc[split:, 0:featureNumber].values

# 4. Train AI Model
sc = MinMaxScaler(feature_range= (0,1)) 

trainingSetScaled = sc.fit_transform(trainingSet)
testSetScaled =  sc.fit_transform(testSet)
testSetScaled = testSetScaled[:, 0:featureNumber]


# 9. AI Predictions
Model = load_model("LSTM/" + modelName +'.h5')

predictionTest = []

# Use the last window of the training set as the initial batch
BatchOne = trainingSetScaled[-WS:]
BatchNew = BatchOne.reshape((1, WS, featureNumber))

# Predict for each time step in the test set
for i in range(len(testSetScaled)):
    # Predict the next value
    FirstPred = Model.predict(BatchNew)[0, 0]
    predictionTest.append(FirstPred)

    # Prepare the input for the next time step
    NewVar = testSetScaled[i, :]
    NewVar = NewVar.reshape(1, 1, featureNumber)
    
    BatchNew = np.concatenate((BatchNew[:, 1:, :], NewVar), axis=1)

predictionTest = np.array(predictionTest).reshape(-1, 1)

# Invert scaling to get the actual values
temp = np.zeros((len(predictionTest), 4))
dump = np.concatenate((temp,predictionTest), axis=1)
predictions = sc.inverse_transform(dump)[:, 4]

realValues = testSet[:, 4]

# Plotting the results
plt.plot(realValues, color='red', label='Actual Values')
plt.plot(predictions, color='blue', label='Predicted Values')
plt.title(target[0])
plt.xlabel('Time')
plt.ylabel('Target')
plt.legend()
plt.show()
print()