import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# Constants from your module
from constant import features, target, dataName, modelName

# Load and prepare data
data = pd.read_csv('./data/' + dataName + '.csv')
feature_cols = features + target
dataAI = data[feature_cols]
dataAI.dropna(inplace=True)

# Split data into training and test set
numOfRecord = len(dataAI)
split = int(round(numOfRecord * 0.7, 0))
WS = 12
featureNumber = len(feature_cols)

trainingSet = dataAI.iloc[:split, 0:featureNumber].values
testSet = dataAI.iloc[split:, 0:featureNumber].values

# Scaling features
sc = MinMaxScaler(feature_range=(0, 1))
trainingSetScaled = sc.fit_transform(trainingSet)
testSetScaled = sc.transform(testSet)

# Load the LSTM model
Model = load_model("LSTM/" + modelName + '.h5')

# Predictions on Test Set
predictionTest = []
for i in range(len(testSetScaled)):
    BatchNew = trainingSetScaled[i:i+WS, :].reshape((1, WS, featureNumber))
    pred = Model.predict(BatchNew)[0, 0]
    predictionTest.append(pred)

# Invert scaling to get the actual values
temp = np.zeros((len(predictionTest), featureNumber-1))
dump = np.concatenate((temp, np.array(predictionTest).reshape(-1, 1)), axis=1)
predictions = sc.inverse_transform(dump)[:, -1]

realValues = testSet[:, -1]
binary_threshold = 0.5
binary_predictions = (predictions > binary_threshold).astype(int)

print("Binary Predictions vs Actual Values:")
for i in range(len(binary_predictions)):
    print(f"Binary Prediction: {binary_predictions[i]}, Actual: {realValues[i]}")

plt.plot(realValues, color='red', label='Actual Binary Values')
plt.plot(binary_predictions, color='blue', label='Binary Predictions')
plt.title("Binary Predictions vs Actual Binary Values")
plt.xlabel('Time')
plt.ylabel('Binary Value')
plt.legend()
plt.show()

# Future Value Predictions
num_future_steps = 10
future_predictions = []

last_window = trainingSetScaled[-WS:]
for _ in range(num_future_steps):
    next_pred = Model.predict(last_window.reshape((1, WS, featureNumber)))[0, 0]
    future_predictions.append(next_pred)
    new_row = np.zeros((1, featureNumber))
    new_row[0, 0] = next_pred  # assuming the predicted feature is the first feature
    last_window = np.append(last_window[:, 1:, :], np.reshape(new_row, (1, 1, featureNumber)), axis=1)

# Invert scaling on the future predictions
temp = np.zeros((len(future_predictions), featureNumber-1))
future_array = np.concatenate((np.array(future_predictions).reshape(-1, 1), temp), axis=1)
future_values = sc.inverse_transform(future_array)[:, 0]

print("Future Values Predicted:")
print(future_values)

plt.plot(future_values, color='blue', label='Future Predicted Values')
plt.title("Future Value Predictions")
plt.xlabel('Future Steps')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
