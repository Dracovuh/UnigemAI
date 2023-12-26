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
dataAI.info()
numOfRecord = len(dataAI)
print(f'\n\n\n{numOfRecord}\n\n\n')
split = int(round(numOfRecord * 0.7, 0))
WS = 12
featureNumber = len(feature_cols)
featureNumber = 7

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
temp = np.zeros((len(predictionTest), featureNumber-1))
dump = np.concatenate((temp,predictionTest), axis=1)
predictions = sc.inverse_transform(dump)[:, featureNumber-1]

realValues = dataAI.iloc[split:, featureNumber-1].values
# realValues = testSet[:, featureNumber-1]
# Apply threshold to convert predictions to binary format

binary_threshold = 0.5
binary_predictions = (predictions > binary_threshold).astype(int)

print("Binary Predictions vs Actual Values:")
for i in range(len(binary_predictions)):
    print(f"Binary Prediction: {binary_predictions[i]}, Actual: {realValues[i]}")

# Plotting the results
plt.plot(realValues, color='red', label='Actual Binary Values from TargetTP100M15')
plt.plot(binary_predictions, color='blue', label='Binary Predictions')
plt.title("Binary Predictions vs Actual Binary Values from TargetTP100M15")
plt.xlabel('Time')
plt.ylabel('Binary Value')
plt.legend()
plt.show()


# Print the first binary prediction
# print("First Binary Prediction:", binary_prediction)
# # Optional: Compare with the actual first value
# # Assuming 'realValues' are binary (0 or 1)
# first_real_value = testSet[0, featureNumber-1]
# print("\nFirst Actual Value:", first_real_value)

# binary_predictions = (predictions > binary_threshold).astype(int)

# # Assuming realValues are also binary, or convert them as well
# # If realValues are not binary, you need to convert them to binary format as well
# binary_realValues = (realValues > binary_threshold).astype(int)

# # Plotting the binary results
# plt.plot(binary_realValues, color='red', label='Actual Binary Values')
# plt.plot(binary_predictions, color='blue', label='Predicted Binary Values')
# plt.title("Binary Predictions vs Actual Binary Values")
# plt.xlabel('Time')
# plt.ylabel('Binary Value')
# plt.legend()
# plt.show()
