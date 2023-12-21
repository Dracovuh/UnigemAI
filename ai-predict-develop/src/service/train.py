import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
from constants import Config, Feature, Intervals
from utils.logger import logger

def training(data, interval):
    ### Validate
    if interval not in Intervals:
        raise ValueError(f"Invalid value for 'interval'. Must be in {Intervals}.")
    try:
        logger.info(f"Train model {interval}")
        output = []

        if interval == '15m':
            output.append(Feature.Outputs.PriceChangeM15)
        elif interval == '1h':
            output.append(Feature.Outputs.PriceChangeH1)
        elif interval == '6h':
            output.append(Feature.Outputs.PriceChangeH4)
        else:
            output.append(Feature.Outputs.PriceChangeH24)

        feature_cols = Feature.Inputs + output
        dataAI = data[feature_cols].copy() 
        dataAI.dropna(inplace=True)

        # print(f"==={interval}===")
        # dataAI.info()
        # print("=================")

        numOfRecord = len(dataAI)
        split = int(round(numOfRecord * 0.7, 0))

        # Split data into training set and test set
        featureNumber = len(feature_cols)
        trainingSet = dataAI.iloc[:split, 0:featureNumber].values
        testSet = dataAI.iloc[split:, 0:featureNumber].values

        WS = Config.Training.WS

        if len(trainingSet)<WS:
            logger.error(f"Not enough training data for {interval} model.")
            return False

        # Train AI Model
        sc = MinMaxScaler(feature_range=(0, 1))

        trainingSetScaled = sc.fit_transform(trainingSet)
        testSetScaled = sc.fit_transform(testSet)
        testSetScaled = testSetScaled[:, 0:featureNumber]

        xTrain = []
        yTrain = []

        WS = min(WS, len(trainingSetScaled) - 1)

        # output_feature_indices = Feature.Outputs

        for i in range(WS, len(trainingSetScaled)):
            xTrain.append(trainingSetScaled[i-WS:i, 0:featureNumber-1])
            yTrain.append(trainingSetScaled[i, featureNumber-1])

        # Reshape training_set_scaled for LSTM input
        xTrain, yTrain = np.array(xTrain), np.array(yTrain)
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], featureNumber-1))

        # Load the pre-trained LSTM model
        modelPath = f"src/models/{interval}/model.h5"

        if os.path.isfile(modelPath):
            pretrained_model = load_model(modelPath)
            # Freeze the layers (optional)
            for layer in pretrained_model.layers:
                layer.trainable = False

            # Modify the model for your new task
            model = Sequential(pretrained_model.layers[:-1])  # Removing the original output layer
            model.add(Dense(units=1, activation='sigmoid'))  # Add new output layer

            # Compile the new model
            model.compile(optimizer='adam', loss='mean_squared_error')
        else:
            # Define LSTM model
            model = Sequential()

            model.add(LSTM(units=70, return_sequences=True, input_shape=(xTrain.shape[1], featureNumber-1)))
            model.add(Dropout(0.2))

            model.add(LSTM(units=70, return_sequences=True))
            model.add(Dropout(0.2))

            model.add(LSTM(units=70, return_sequences=True))
            model.add(Dropout(0.2))

            model.add(LSTM(units=70))
            model.add(Dropout(0.2))

            # Update the output layer for multiple outputs
            model.add(Dense(units=1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')

        # Fit the model
        model.fit(xTrain, yTrain, epochs=80, batch_size=32)

        # Save the model
        save_dir = f"src/models/{interval}"
        os.makedirs(save_dir, exist_ok=True)
        model.save(os.path.join(save_dir, "model.h5"))
    except Exception as e:
        logger.error(e)
        return False
