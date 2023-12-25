# 1. Input library and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, save_model
from keras.layers import Dense, LSTM, Dropout
from keras.initializers import Orthogonal
from constant import features, target, dataName
import os

data = pd.read_csv('./data/' + dataName + '.csv')

feature_cols = features + target

dataAI = data[feature_cols]
dataAI.dropna(inplace = True)
dataAI.dropna(axis = 0)
dataAI.info()

numOfRecord = len(dataAI)
print(f'\n\n\n{numOfRecord}\n\n\n')
split = int(round(numOfRecord * 0.7, 0))

# 3. Split data into training set and test set
featureNumber = len(feature_cols)
# featureNumber= 7
trainingSet = dataAI.iloc[:split, 0:featureNumber].values
testSet = dataAI.iloc[split:, 0:featureNumber].values

# 4. Train AI Model
sc = MinMaxScaler(feature_range= (0,1)) 

trainingSetScaled = sc.fit_transform(trainingSet)
testSetScaled =  sc.fit_transform(testSet)
testSetScaled = testSetScaled[:, 0:featureNumber]

xTrain = []
yTrain = []

WS = 12
for i in range(WS, len(trainingSetScaled)):
    xTrain.append(trainingSetScaled[i-WS:i, 0:featureNumber])
    yTrain.append(trainingSetScaled[i, 2])

# 5. Reshape training_set_scaled for LSTM input
xTrain, yTrain = np.array(xTrain), np.array(yTrain)
print(len(xTrain))
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], featureNumber))

# 6. Define LSTM model
# model = Sequential()

# model.add(LSTM(units= 70, return_sequences= True, input_shape = (xTrain.shape[1], 5)))
# model.add(Dropout(0.2))

# model.add(LSTM(units=70,return_sequences=True))
# model.add(Dropout(0.2))

# model.add(LSTM(units=70,return_sequences=True))
# model.add(Dropout(0.2))

# model.add(LSTM(units=70))
# model.add(Dropout(0.2))

# model.add(Dense(units=1))

model = Sequential()

model.add(LSTM(units=70, return_sequences=True, input_shape=(xTrain.shape[1], featureNumber)))
model.add(Dropout(0.2))

model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=70, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=70))
model.add(Dropout(0.2))

model.add(Dense(units=1, activation='sigmoid'))


# 7. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Fit the model
model.fit(xTrain, yTrain, epochs=80, batch_size=32, validation_split=0.2)

loss = model.history.history['loss']
plt.plot(range(len(loss)), loss)
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()
print('')

# model.save("LSTM/" + dataName)
save_dir = "LSTM/"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
# save_model(model, os.path.join(save_dir, dataName+'.h5'), save_format='h5')
model.save(os.path.join(save_dir, dataName + ".h5")) 