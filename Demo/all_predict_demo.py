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

realValues = testSet[:, featureNumber-1]
# Apply threshold to convert predictions to binary format

binary_threshold = 0.5
binary_prediction = int(FirstPred > binary_threshold)
# Print the first binary prediction
print("First Binary Prediction:", binary_prediction)
# Optional: Compare with the actual first value
# Assuming 'realValues' are binary (0 or 1)
first_real_value = testSet[0, featureNumber-1]
print("\nFirst Actual Value:", first_real_value)

binary_predictions = (predictions > binary_threshold).astype(int)

# Assuming realValues are also binary, or convert them as well
# If realValues are not binary, you need to convert them to binary format as well
binary_realValues = (realValues > binary_threshold).astype(int)

# Plotting the binary results
plt.plot(binary_realValues, color='red', label='Actual Binary Values')
plt.plot(binary_predictions, color='blue', label='Predicted Binary Values')
plt.title("Binary Predictions vs Actual Binary Values")
plt.xlabel('Time')
plt.ylabel('Binary Value')
plt.legend()
plt.show()

# # Plotting the results
# plt.plot(realValues, color='red', label='Actual Values')
# plt.plot(predictions, color='blue', label='Predicted Values')
# plt.title(target[0])
# plt.xlabel('Time')
# plt.ylabel('Target')
# plt.legend()
# plt.show()
# print()

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

#29/12/2023 (binary predict print out 10 values)
# last_window = trainingSetScaled[-WS:]
# for _ in range(num_future_steps):
#     reshaped_window = last_window.reshape((1, WS, featureNumber))
#     next_pred = Model.predict(reshaped_window)[0, 0]
#     # Convert prediction to binary
#     binary_pred = int(next_pred > binary_threshold)
#     future_binary_predictions.append(binary_pred)
#     # Update last window with the new prediction
#     new_row = np.zeros(featureNumber)
#     new_row[0] = next_pred  # Update with the continuous prediction
#     last_window = np.roll(last_window, -1, axis=0)
#     last_window[-1] = new_row
# # Print future binary values
# print("Future Binary Values Predicted:")
# print(future_binary_predictions)
# # Plotting the future binary predictions
# plt.plot(future_binary_predictions, color='blue', label='Future Binary Predictions')
# plt.title("Future Binary Value Predictions")
# plt.xlabel('Future Steps')
# plt.ylabel('Binary Value')
# plt.legend()
# plt.show()