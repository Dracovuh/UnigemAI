import os
import sys
import math
import requests
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
import statistics as s
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
sys.path.append('src')

#  ! logger
import logging
from colorlog import ColoredFormatter

# Set up logging with a colored formatter
formatter = ColoredFormatter(
    "[%(asctime)s] [%(log_color)s%(levelname)-s%(reset)s] \"%(message)s\"",
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)

handler = logging.StreamHandler()
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ! constants

class Config:
    class Postgres:
        user = os.getenv("PostgresUser")
        database = os.getenv("PostgresDatabase")
        password = os.getenv("PostgresPassword")
        port = os.getenv("PostgresPort")
        host = os.getenv("PostgresHost")
        schema = os.getenv("PostgresSchema")

    PORT = os.getenv("PORT")
    HOST = os.getenv("HOST")
    IS_DEV = True
    URL_API= os.getenv("URL_API")

    PREDICT_TOKEN = os.getenv("PREDICT_TOKEN")
    TRAIN_TOKEN = os.getenv("TRAIN_TOKEN")

    class Training:
        WS = 3
        EPOCH = 100

INTERVALS = ['15m', '1h', '6h', '24h']

class Feature:
    Inputs = [
    
    "Id",
    # "CrawlAt",

    # "TransactionBuyM5",
    # "TransactionSellM5",
    # "TransSellM5ChangePercent",
    # "TransBuyM5ChangePercent",
    # "VolumeM5",
    # "VolumeM5ChangePercent",

    "HolderCount",
    "TransactionBuyH1",
    "TransactionSellH1",
    "VolumeH1",
    "TransactionBuyH6",
    "TransactionSellH6",
    "VolumeH6",

    ]

    class Outputs:

        PriceChangeM15 = "TokenPriceM15"
        PriceChangeH1 = "TokenPriceH1"
        PriceChangeH4 = "TokenPriceH4"
        PriceChangeH24 = "TokenPriceH24"

# ! constants

def training(
        # data: pd.DataFrame,
        interval):
    interval = '15m'
    data = pd.read_csv(fr'./data/training_ai_data01-02_03_24 (2).csv')
    try:
        # ! Validate
        if interval not in INTERVALS:
            logger.info(f"Invalid value for 'interval'. Must be in {INTERVALS}.")
            return False
        if data is None:
            logger.info(f"[train.py] No data found.")  
            return False
        
        logger.info(f"Train model {interval}")

        # ! Preprocess data start.
        # Set output
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
        

        dataAI = data[feature_cols].copy() # Get data base on inputs and outputs
        # dataAI = dataAI[:5000]
        dataAI = dataAI[::-1] # Reverse it 
        original_data_length=len(dataAI)

        # Drop/Fill the N/As
        dataAI.dropna(inplace = True)
        # dataAI.fillna(0, inplace = True)

        # Log the length be/af
        numOfRecord = len(dataAI)
        logger.info(f'Remainings/Original length (dropNA): {numOfRecord}/{original_data_length} ({numOfRecord*100/original_data_length}%)')
        
        # ! start looping from this
        grouped=dataAI.groupby('Id')

        datas=[]

        for group_id, group_df in grouped:
            datas.append(group_df[1:])
        
        # Drop the column "Id" from each DataFrame in the list 'datas'
        for data in datas:
            data.drop("Id", axis=1, inplace=True)

        # print(datas)

        # ! make sure data is enough for training: at least 1 token sequences, and it must have at least 3 records
        WS = Config.Training.WS
        # is_it_enough_data=True
        # for each_data in datas:
        #     if len(each_data)<3:
                
        #     logger.error(f"Not enough training data for {interval} model.")
        #     return False
            
        # ! change the data to be only list of lists of values
        valued_datas =[]
        for each_data in datas:
            valued_data = each_data.iloc[:][1:].values
            valued_datas.append(valued_data)

        temp=valued_datas
        valued_datas=[]
        for each_data in temp:
            if len(each_data)>WS:
                valued_datas.append(each_data)

        # ! this is old method, trying new method which is randomly choose 30% number of tokens's sequence
        featureNumber = len(feature_cols) -1

        total_dfs = len(datas)
        num_dfs_30_percent = int(0.3 * total_dfs)
        training_datas, testing_datas = train_test_split(valued_datas, test_size=num_dfs_30_percent, random_state=42, shuffle=True)

        # Normalize
        sc = MinMaxScaler(feature_range = (0, 1))
        data_sets=[testing_datas, training_datas]
        data_set_scaled=[]
        for data_set in data_sets:
            new_data_set=[]
            for each_group in data_set:
                scaled_group=sc.fit_transform(each_group)
                new_data_set.append(scaled_group)
            data_set_scaled.append(new_data_set)

        testing_datas_scaled=data_set_scaled[0]
        training_datas_scaled=data_set_scaled[1]

        #  ! x_trains = [[], [], []]
        x_trains=[]
        y_trains=[]

        # training_datas_scaled is a list of ndarrays
        # for each nd array, create a list of multiple ws-array
        
        for training_data in training_datas_scaled:
            x_train=[]
            y_train=[]
            for i in range(WS, len(training_data)):
                x_item = training_data[i-WS:i, 0:featureNumber-1]
                y_item = training_data[i, featureNumber-1]

                # print(y_item)

                x_train.append(x_item)
                y_train.append(y_item) 

            x_train=np.array(x_train)
            y_train=np.array(y_train)

            x_trains.append(x_train)
            y_trains.append(y_train)
            
        # Reshape data for LSTM input
        x_trains_reshaped=[]
        for i in range(len(x_trains)):
            each_x_train = x_trains[i]

            if each_x_train.shape[0] == 0:
                continue

            each_x_train=np.reshape(each_x_train, (each_x_train.shape[0], each_x_train.shape[1], featureNumber-1))
            x_trains_reshaped.append(each_x_train)

        # ! Training
        # Load the pre-trained LSTM model
        modelPath = f"src/models/{interval}/model.h5"

        if os.path.isfile(modelPath):
            pretrained_model = load_model(modelPath)
            # Freeze the layers
            for layer in pretrained_model.layers:
                layer.trainable = False

            # Modify the model
            model = Sequential(pretrained_model.layers[:-1])  # Removing the original output layer
            model.add(Dense(units = 1, activation = 'sigmoid'))  # Add new output layer

            # Compile the new model
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        else:
            # Define LSTM model
            model = Sequential()

            model.add(LSTM(units = 70, return_sequences = True, input_shape = (x_trains[0].shape[1], featureNumber-1)))
            model.add(Dropout(0.2))

            model.add(LSTM(units = 70, return_sequences = True))
            model.add(Dropout(0.2))

            model.add(LSTM(units = 70, return_sequences = True))
            model.add(Dropout(0.2))

            model.add(LSTM(units = 70))
            model.add(Dropout(0.2))

            model.add(Dense(units = 1, activation = 'sigmoid'))

            # Compile the model
            model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        # Fit the model
            
        # ? modify this part so it can learn multiple tokens
        '''histories=[]
        logger.error(len(x_trains_reshaped)+1)
        logger.error(len(y_trains))
        # ! training loop
        for i in range(0, len(x_trains_reshaped)):
            logger.info(f"Training token {i+1}:")
            history=model.fit(x_trains_reshaped[i], y_trains[i], epochs =
                               Config.Training.EPOCH, 
                            #   2,
                               batch_size = 32, validation_split=0.2)
            histories.append(history)

        Save the model
        save_dir = rf'src/models/{interval}'
        os.makedirs(save_dir, exist_ok = True)
        save_dir = os.path.join(save_dir, "model.h5")
        model.save(save_dir)


        # ! Loss diagram

        train_loss_list = [ item.history['loss'] for item in histories]
        # val_loss_list = [ item.history['val_loss'] for item in histories]

        if Config.IS_DEV:
            plt.xlabel('Epoch')
            plt.ylabel('Train Loss')
            # plt.plot(range(len(train_loss)), train_loss)
            for item in train_loss_list:
                plt.plot(range(len(item)), item)
            plt.show()
            plt.close()'''

            # for item in val_loss_list:
            #     plt.plot(range(len(item)), item)
            # plt.show()
            # plt.close()
            
            # plt.plot(range(len(train_loss_list[-1])), train_loss_list[-1])
            # plt.show()
            # plt.close()

        # ! Evaluation
        # Save the model
        save_dir = rf'src/models/{interval}'
        os.makedirs(save_dir, exist_ok = True)
        save_dir = os.path.join(save_dir, "model.h5")
        model.save(save_dir)

        predictionTest = []

        x_tests=[]

        for testing_data in testing_datas_scaled:
            x_test_token=[]
            for i in range(WS, len(testing_data)):
                x_item=testing_data[i-WS:i, :featureNumber-1]
                x_test_token.append(x_item)
            x_test_token=np.array(x_test_token)

            x_tests.append(x_test_token)

        '''
        # Predict for each time step in the test set
        for i in range(len(testSetScaled)):
            # Predict the next value
            FirstPred = model.predict(BatchNew)[0, 0]
            predictionTest.append(FirstPred)

            # Prepare the input for the next time step
            nextData = testSetScaled[i, :featureNumber-1]
            nextInput = nextData.reshape(1, 1, featureNumber-1)
            
            BatchNew = np.concatenate((BatchNew[:, 1:, :], nextInput), axis=1)

        predictionTest = np.array(predictionTest).reshape(-1, 1)'''

        # ! code starts here
        # loop từng token trong testing_datas_scaled 
        # vì testing_datas_scaled đã được chia thành x_tests và y_tests
        # nên cứ thế truyền thẳng vào model để predict thôi
        # các data ở đó đã được scaled.
        # mình nghĩ chỉ nên predict 1 token duy nhất thôi
        # vậy thì tập training set sẽ chứa hầu hết các token, còn testing set sẽ là phần
        # nhỏ còn lại

        # lấy token đầu tiên của tập test để predict
        # so sánh với real value
        # đưa ra đánh giá
        # predicted_test sẽ là mảng chứa kết quả predict của từng token

        RMSE_list=[]
        Rsquare_list=[]

        for i in range(len(x_tests)):
            predicted_test=[]

            one_token_data_test=x_tests[i]
            one_token_real_values=testing_datas_scaled[i][3:, -1]

            for i in range(len(one_token_data_test)):
                a_batch=one_token_data_test[i]
                a_batch_reshaped=a_batch.reshape(1, WS, featureNumber-1)

                predicted_value=model.predict(a_batch_reshaped)[0, 0]
                predicted_test.append(predicted_value)

            predicted_test=np.array(predicted_test).reshape(-1, 1)
            predicted_test=sc.fit_transform(predicted_test)
            # ! code ends here

            # Invert scaling to get the actual values

            # Plotting the results.
            if Config.IS_DEV==False:
                plt.plot(one_token_real_values, color='blue', label='Actual Values')
                plt.plot(predicted_test, color='red', label='Predicted Values', linestyle='--')
                plt.title('Token Price Change in M15 Prediction')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.legend()
                plt.show()
                # plt.close()

            # Formula evaluation:
            RMSE = math.sqrt(mean_squared_error(one_token_real_values, predicted_test))   
            Rsquare = r2_score(one_token_real_values, predicted_test) 
            RMSE_list.append(RMSE)
            Rsquare_list.append(Rsquare)

        logger.info(f'RMSE(~0): {s.mean(RMSE_list)}') 
        logger.info(f'Rsquare: {s.mean(Rsquare_list)}')

        # # ! Upload model for Predict Server
        # url = f'{Config.URL_API}/api/upload?interval={interval}'
 
        # headers = {'Authorization': f"Bearer {Config.TRAIN_TOKEN}"}
        # file_path = rf'src\models\{interval}\model.h5'
        
        # with open(file_path, 'rb') as file:
        #     files = {'file': ('model.h5', file, 'application/octet-stream')}
        #     response = requests.post(url, files = files, headers = headers
        #                              )
        #     if response.status_code == 200 and response.headers['content-type'] == 'application/json':
        #         json_data = response.json()
        #         logger.info(json_data)
        #     else:
        #         logger.error(f"Error: {response.status_code}, {response.text}")

        return True
    
    except Exception as e:
        logger.error(e)
        return False
    
training('')