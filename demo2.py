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
        WS = 30
        EPOCH = 100

INTERVALS = ['15m', '1h', '6h', '24h']

class Feature:
    Inputs = [

    "Id",
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

def data_process_scaless(data, feature_cols):
    data_AI = data[feature_cols].copy()
    # dataAI = dataAI[:5000]
    data_AI = data_AI[::-1] # Reverse it 
    original_data_length=len(data_AI)

    # Drop the N/As
    data_AI.dropna(inplace = True)

    # Log the length be/af
    record_num = len(data_AI)
    logger.info(f'Remainings/Original length (dropNA): {record_num}/{original_data_length} ({record_num*100/original_data_length}%)')
    
    data_AI_group_by_ID=data_AI.groupby('Id')

    list_data=[]
    max_length_token_data=data_AI_group_by_ID.size().max()

    for group_id, group_df in data_AI_group_by_ID:
        list_data.append(group_df[1:])
    
    for data in list_data:
        data.drop("Id", axis=1, inplace=True)

    WS = Config.Training.WS
    
        
    data_value_only =[]
    for each_data in list_data:
        valued_data = each_data.iloc[:][1:].values
        data_value_only.append(valued_data)

    temp=data_value_only
    data_value_only=[]
    for each_data in temp:
        if len(each_data)>WS:
            data_value_only.append(each_data)

    feature_count = len(feature_cols) -1

    total_dfs = len(list_data)
    num_dfs_30_percent = int(0.3 * total_dfs)
    training_datas, testing_datas = train_test_split(data_value_only, test_size=num_dfs_30_percent, random_state=42, shuffle=True)

    x_train_reshaped, combined_y_train = prepare_training_data(training_datas, WS,max_length_token_data, feature_count)
    x_test_reshaped, combined_y_test = prepare_training_data(testing_datas, WS,max_length_token_data, feature_count)
    return x_train_reshaped, combined_y_train, x_test_reshaped, combined_y_test, max_length_token_data, feature_count

def prepare_training_data(training_datas_scaled, WS, max_length_token_data, feature_count):
    x_trains = []
    y_trains = []

    x_target_shape = (max_length_token_data, WS, feature_count - 1)
    y_target_shape = (max_length_token_data, 1)

    for training_data in training_datas_scaled:
        x_train = []
        y_train = []
        for i in range(WS, len(training_data)):
            x_item = training_data[i-WS:i, 0:feature_count-1]
            y_item = training_data[i, feature_count-1]

            x_train.append(x_item)
            y_train.append(y_item) 

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_trains.append(x_train)
        y_trains.append(y_train)

    x_trains_padded = []
    y_trains_padded = []

    for each_x_train in x_trains:
        if each_x_train.shape[0] == 0:
            x_trains_padded.append(each_x_train)
            continue

        pad_width = [(0, x_target_shape[0] - each_x_train.shape[0]), (0, 0), (0, 0)]
        x_padded_array = np.pad(each_x_train, pad_width, mode='constant')
        x_padded_array = x_padded_array.reshape(x_target_shape)
        x_trains_padded.append(x_padded_array)

    for each_y_train in y_trains:
        if each_y_train.shape[0] == 0:
            y_trains_padded.append(each_y_train)
            continue

        padding_length = max_length_token_data - len(each_y_train)
        padded_array = np.pad(each_y_train, (0, padding_length), mode='constant')
        padded_array = padded_array.reshape(y_target_shape)
        y_trains_padded.append(padded_array)

    combined_x_train = np.array(x_trains_padded)
    combined_y_train = np.array(y_trains_padded)

    x_train_reshaped = np.reshape(combined_x_train, (combined_x_train.shape[0], combined_x_train.shape[1], -1))

    return x_train_reshaped, combined_y_train

def training(
        # data: pd.DataFrame,
        interval):
    interval = '15m'
    data = pd.read_csv(fr'src/data/training_ai_data01-02_03_24.csv')
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
        
        x_train, y_train, x_test, y_test, max_length_token_data, feature_count = data_process_scaless(data, feature_cols)
        
        # ! Define model
        model=define_model(interval, max_length_token_data, Config.Training.WS, feature_count, False)
        model.summary()
        # ! Define model
        model=define_model(interval, max_length_token_data, Config.Training.WS, feature_count, False)
        model.summary()

        # ! Training model
        train_model(model, x_train, y_train, Config.Training.EPOCH)

        # ! Save model
        save_model(model, interval)

        # ! Evaluation
        evaluation(model, x_test, y_test)

        # ! Upload model for Predict Server
        # upload_model(interval)
        return True
    
    except Exception as e:
        logger.error(e)
        return False
    
def define_model(interval:str, max_length_token_data, WS, feature_count, is_scale:bool=True):
    # Load the pre-trained LSTM model
    modelPath = f"src/models/{interval}/model.h5"

    if os.path.isfile(modelPath):
        pretrained_model = load_model(modelPath)
        # Freeze the layers
        for layer in pretrained_model.layers:
            layer.trainable = False

        # Modify the model
        model = Sequential(pretrained_model.layers[:-1])  # Removing the original output layer
        model.add(Dense(units = max_length_token_data))  
        # Add new output layer

        # Compile the new model
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model
    
    else:
        # Define LSTM model
        model = Sequential()

        model.add(LSTM(units = 70, return_sequences = True, input_shape = (max_length_token_data, WS*( feature_count-1))))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 70, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 70, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units = 70))
        model.add(Dropout(0.2))

        model.add(Dense(units = max_length_token_data))

        # Compile the model
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        return model
    
def train_model(model, x_train, y_train, epoch):
    history = model.fit(x_train, y_train, epochs = epoch, batch_size = 32, validation_split=0.2)
    return history

def save_model(model, interval):
    save_dir = rf'src/models/{interval}'
    os.makedirs(save_dir, exist_ok = True)
    save_dir = os.path.join(save_dir, "model.h5")
    model.save(save_dir)
    pass

def upload_model(interval: str):
    url = f'{Config.URL_API}/api/upload?interval={interval}'
    headers = {'Authorization': f"Bearer {Config.TRAIN_TOKEN}"}
    file_path = rf'src\models\{interval}\model.h5'
    with open(file_path, 'rb') as file:
        files = {'file': ('model.h5', file, 'application/octet-stream')}
        response = requests.post(url, files = files, headers = headers
                                )
        if response.status_code == 200 and response.headers['content-type'] == 'application/json':
            json_data = response.json()
            logger.info(json_data)
        else:
            logger.error(f"Error: {response.status_code}, {response.text}")

def evaluation(model, x_test_reshaped, y_test):

    print(x_test_reshaped.shape)
    print(y_test.shape) 
    
    model.summary()

    # Predict using the trained model
    predictions = model.predict(x_test_reshaped)

    # Get the number of tokens
    num_tokens = x_test_reshaped.shape[0]

    RMSE_list=[]
    Rsquare_list=[]

    # Plot predictions vs. actual values for each token
    for token_index in range(num_tokens):
        token_real_values=y_test[token_index,:,0]
        token_predictions=predictions[token_index,:]



        plt.figure(figsize=(10, 6))
        plt.plot(token_real_values, label='Actual Values')
        plt.plot(token_predictions, label='Predictions', linestyle='--')
        plt.xlabel('Time Steps')
        plt.ylabel('Output Value')
        plt.title(f'Predictions vs. Actual Values for Token {token_index + 1}')
        plt.legend()
        plt.savefig(os.path.join( rf'src/models/15m/test_result', f'plot_token_{token_index + 1}.png'))

        plt.close()
        # plt.show()

        Rsquare = r2_score(token_real_values, token_predictions)
        RMSE = mean_squared_error(token_real_values, token_predictions)

        RMSE_list.append(RMSE)
        Rsquare_list.append(Rsquare)
        

    logger.info(f'RMSE(~0): {s.mean(RMSE_list)}')
    logger.info(f'Rsquare: {s.mean(Rsquare_list)}')


training('')