# 1. Input library and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

dataName = 'ckd-20231204-094300'
modelName = 'ckd-20231129-084117'

features = [
"TransactionBuyM5",
"TransactionBuyH1",
"TransactionBuyH6",
# "TransactionBuyH24",
"TransactionSellM5",
"TransactionSellH1",
"TransactionSellH6",
# "TransactionSellH24",
"TransBuyM5ChangePercent",
"TransBuyH1ChangePercent",
"TransBuyH6ChangePercent",
# "TransBuyH24ChangePercent",
"TransSellM5ChangePercent",
"TransSellH1ChangePercent",
"TransSellH6ChangePercent",
# "TransSellH24ChangePercent",
"VolumeM5",
"VolumeH1",
"VolumeH6",
# "VolumeH24",
"VolumeM5ChangePercent",
"VolumeH1ChangePercent",
"VolumeH6ChangePercent",
# "VolumeH24ChangePercent",
"NarrativeScore",
"SocialMediaScore",
"InfluencerScore",
]

target = ["TokenPriceM15"]

def predicting():
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

    # 3. Split data into training set and test set
    trainingSet = dataAI.iloc[:split, 0:5].values
    testSet = dataAI.iloc[split:, 0:5].values

    # 4. Train AI Model
    sc = MinMaxScaler(feature_range= (0,1)) 

    trainingSetScaled = sc.fit_transform(trainingSet)
    testSetScaled =  sc.fit_transform(testSet)
    testSetScaled = testSetScaled[:, 0:5]


    # 9. AI Predictions
    Model = load_model("LSTM/" + dataName)

    predictionTest = []

    # Use the last window of the training set as the initial batch
    BatchOne = trainingSetScaled[-WS:]
    BatchNew = BatchOne.reshape((1, WS, 5))

    # Predict for each time step in the test set
    for i in range(len(testSetScaled)):
        # Predict the next value
        FirstPred = Model.predict(BatchNew)[0, 0]
        predictionTest.append(FirstPred)

        # Prepare the input for the next time step
        NewVar = testSetScaled[i, :]
        NewVar = NewVar.reshape(1, 1, 5)
        
        BatchNew = np.concatenate((BatchNew[:, 1:, :], NewVar), axis=1)

    predictionTest = np.array(predictionTest).reshape(-1, 1)

    # Invert scaling to get the actual values
    predictions = sc.inverse_transform(np.concatenate((np.zeros((len(predictionTest), 4)), predictionTest), axis=1))[:, 4]

    realValues = testSet[:, 4]

    return realValues[-1]



