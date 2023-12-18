import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the data
df = pd.read_csv('ckd-20231208-041519.csv')

# Selecting the relevant columns
features = ["TransactionBuyM5", "TransactionSellM5", "VolumeM5"]

# Check if all the features are in the dataframe
if not all(item in df.columns for item in features):
    missing_cols = [item for item in features if item not in df.columns]
    raise ValueError(f"Missing columns in the dataframe: {missing_cols}")

# Assuming 'TargetPrice' is the column you want to predict
data = df[features]

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :-1] # All features except target
        X.append(a)
        Y.append(data[i + time_step, -1]) # Target price
    return np.array(X), np.array(Y)

# Define time step and reshape data
time_step = 100
X, y = create_dataset(scaled_data, time_step)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2])

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

import matplotlib.pyplot as plt

# Predict and inverse transform the results
predictions = model.predict(X_test)
# predictions = scaler.inverse_transform(predictions) # Use if you scale the target variable

# Plotting the results
plt.plot(y_test, label='True Value')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
