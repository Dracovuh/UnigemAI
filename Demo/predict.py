import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import tensorflow as tf
from constants import Config, Feature, INTERVALS, TIME_PRICE
from utils.logger import logger

# @tf.function(reduce_retracing=True)
# @tf.autograph.experimental.do_not_convert
def predict(interval, data):
    ### Validate
    try:
        if interval not in INTERVALS:
            raise ValueError(f"Invalid value for 'interval'. Must be in {INTERVALS}.")
    
        logger.info(f"Predict using {interval} model")
        # output = []

        # data = pd.read_csv('src\data\ckd-20231129-084117.csv')

        # if interval == '15m':
        #     output.append(Feature.Outputs.PriceChangeM15)
        # elif interval == '1h':
        #     output.append(Feature.Outputs.PriceChangeH1)
        # elif interval == '6h':
        #     output.append(Feature.Outputs.PriceChangeH4)
        # else:
        #     output.append(Feature.Outputs.PriceChangeH24)

        feature_cols = Feature.Inputs 
        # + output
        dataRaw = data[TIME_PRICE+ feature_cols]
        dataAI = dataRaw.copy()
        dataAI.dropna(inplace=True)

        numOfRecord = len(dataAI)
        WS = Config.Training.WS
        if numOfRecord < WS:
            logger.info(numOfRecord)
            response = {
            'status': False,
            'message':'Not enough data for predict.'
            }
            return response
        # split = int(round(numOfRecord * 0.7, 0))

        # 3. Split data into training set and test set
        trainingSet = dataAI.iloc[:, :].values
        # testSet = dataAI.iloc[split:, 0:].values

        realValues = trainingSet[:, :len(TIME_PRICE)] 
        trainingSet = trainingSet[:, len(TIME_PRICE):]
 
        # 4
        sc = MinMaxScaler(feature_range=(0, 1))
        trainingSetScaled = sc.fit_transform(trainingSet)
        # testSetScaled = sc.fit_transform(testSet)
        # testSetScaled = testSetScaled[:, 0:5]

        Model = load_model(rf'src\models\{interval}\model.h5')

        # 5 Predict
        # Use the last window of the training set as the initial batch
        BatchOne = trainingSetScaled[-WS:]
        BatchNew = BatchOne.reshape((1, WS, len(Feature.Inputs)
                                    #  + len(output)
                                     )) 

        # Predict for the last time step in the test set
        LastPred = Model.predict(BatchNew)[0, 0]
        predictionTest = np.array(LastPred).reshape(-1, 1)

        # Invert scaling to get the actual values
        # temp = np.zeros((len(predictionTest), 4))
        # dump = np.concatenate((temp,predictionTest), axis=1)
        # predictions = sc.inverse_transform(dump)[:, 4]

        # predictions = sc.inverse_transform(
        #     np.concatenate((np.zeros((len(predictionTest), len(TimeAndPrice) + len(output) - 1)), predictionTest), axis=1))[:, len(TimeAndPrice):]

        priceChange = predictionTest[0][0]
        logger.info(f'Predict: {priceChange}')
        response = {
            'status': True,
            'data':{
                'prices':realValues,
                'priceChange':priceChange
            }
        }
        return response

    except Exception as e:
        logger.error(e)
        response = {
            'status': False,
        }
        return response
    