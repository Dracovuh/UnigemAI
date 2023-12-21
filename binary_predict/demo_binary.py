import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import os

# Load data
dataName = 'your_data_name'  # Replace with your actual data file name
data = pd.read_csv('./data/' + dataName + '.csv')

# Assuming 'features' and 'target' are defined correctly
feature_cols = features + target
dataAI = data[feature_cols]
dataAI.dropna(inplace=True)

# Split data into training and test sets
numOfRecord = len(dataAI)
split = int(round(numOfRecord * 0.7, 0))
featureNumber = len(features)  # Assuming 'features' is a list of feature names
trainingSet = dataAI.iloc[:split, :-1].values  # Exclude target column
testSet = dataAI.iloc[split:, :-1].values

# Target column for training and test set
yTrain = dataAI.iloc[:split, -1].values  # Last column as target
yTest = dataAI.iloc[split:, -1].values

# Scale the features
sc = MinMaxScaler(feature_range=(0,1))
trainingSetScaled = sc.fit_transform(trainingSet)
testSetScaled = sc.transform(testSet)

# Prepare data for LSTM
xTrain, yTrain = [], np.array(yTrain)
WS = 12  # Window size
for i in range(WS, len(trainingSetScaled)):
    xTrain.append(trainingSetScaled[i-WS:i, 0:featureNumber])

xTrain = np.array(xTrain)
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], featureNumber))

# Define LSTM model for binary classification
model = Sequential()
model.add(LSTM(units=70, return_sequences=True, input_shape=(xTrain.shape[1], featureNumber)))
model.add(Dropout(0.2))
# Add more LSTM and Dropout layers as needed
model.add(LSTM(units=70))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Sigmoid activation for binary output

# Compile the model for binary classification
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(xTrain, yTrain, epochs=80, batch_size=32, validation_split=0.2)

# Plot the training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# Save the model
save_dir = "LSTM/"
os.makedirs(save_dir, exist_ok=True)
model.save(os.path.join(save_dir, dataName + ".h5"))

# You can add additional code to evaluate the model on 'testSetScaled' and 'yTest'
