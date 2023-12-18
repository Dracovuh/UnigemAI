import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Example: Load your dataset (replace this with your data loading mechanism)
# Assuming a CSV file with columns for 10 tokens and target token price
data = pd.read_csv('ckd-20231207-044857.csv')

# Data preprocessing
# Assuming 'target_token_price' is the column for the token price to be predicted
target_column = 'target_token_price'

# Extract features (10 tokens) and target variable
features = data.drop(columns=[target_column])
target = data[target_column]

# Normalizing data using MinMaxScaler
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
target_normalized = scaler.fit_transform(np.array(target).reshape(-1, 1))

# Convert data into sequences for LSTM
sequence_length = 10  # Define sequence length
X, y = [], []

for i in range(len(features) - sequence_length):
    X.append(features_normalized[i:(i + sequence_length)])
    y.append(target_normalized[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# Making predictions
predictions = model.predict(X_test)

# Inverse scaling the predictions
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test)

# Optionally, visualize predictions vs actual values for evaluation
# ... (Plotting code or other evaluation metrics)

# Save the model
model.save('token_price_prediction_model.h5')

