# 1. Input library and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense, LSTM, Dropout
from keras.initializers import Orthogonal
from constant import features, target, dataName
import os

# Load your dataset
data = pd.read_csv('./data/' + dataName + '.csv')

# Selecting features and target
feature_cols = features + target
dataAI = data[feature_cols]
dataAI.dropna(inplace=True)

numOfRecord = len(dataAI)
print(f'\n\n\n{numOfRecord}\n\n\n')
split = int(round(numOfRecord * 0.7, 0))

# Split data into training set and test set
featureNumber = len(feature_cols)
trainingSet = dataAI.iloc[:split, 0:featureNumber].values
testSet = dataAI.iloc[split:, 0:featureNumber].values

# Scaling
sc = MinMaxScaler(feature_range=(0, 1))
trainingSetScaled = sc.fit_transform(trainingSet)
testSetScaled = sc.transform(testSet[:, 0:featureNumber])

# Preparing training data for LSTM
xTrain = []
yTrain = []

WS = 12
for i in range(WS, len(trainingSetScaled)):
    xTrain.append(trainingSetScaled[i-WS:i, 0:featureNumber])
    yTrain.append(trainingSetScaled[i, 2])  # Assuming the third column is the target

xTrain, yTrain = np.array(xTrain), np.array(yTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], featureNumber))

# Load the pre-trained LSTM model
pretrained_model = load_model('path/to/your/pretrained/model.h5')

# Freeze the layers (optional)
for layer in pretrained_model.layers:
    layer.trainable = False

# Modify the model for your new task
new_model = Sequential(pretrained_model.layers[:-1])  # Removing the original output layer
new_model.add(Dense(units=your_output_units, activation='your_activation'))  # Add new output layer

# Compile the new model
new_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the new model
new_model.fit(xTrain, yTrain, epochs=80, batch_size=32)

# Plot the training loss
loss = new_model.history.history['loss']
plt.plot(range(len(loss)), loss)
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()

# Save the model
save_dir = "LSTM/"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, dataName + ".h5")) 
