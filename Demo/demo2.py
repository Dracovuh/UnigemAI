import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Tạo dữ liệu giả định
def generate_data(n_samples, n_tokens):
    # Tạo dữ liệu giả định cho 10 tokens đầu vào
    X = np.random.rand(n_samples, n_tokens)
    
    # Tạo giá trị dự đoán cho token cần dự đoán (token thứ 0)
    y = X[:, 0] + np.random.normal(0, 0.1, n_samples)
    
    return X, y

n_samples = 1000
n_tokens = 10

X, y = generate_data(n_samples, n_tokens)

# Xây dựng mô hình LSTM
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_tokens, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Reshape dữ liệu để phù hợp với đầu vào của mô hình LSTM
X = np.reshape(X, (n_samples, n_tokens, 1))

# Huấn luyện mô hình
epochs = 10
model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

# Đánh giá mô hình hoặc sử dụng để dự đoán giá trị mới

# Xem thực nghiệm dự đoán
y_pred = model.predict(X)

# Vẽ dự đoán và thực tế so với mỗi giá trị của token
for i in range(n_tokens):
    plt.figure(figsize=(12, 6))
    plt.plot(X[:, i, 0], label='True')
    plt.plot(y_pred[:, i], label='Predicted')
    plt.title(f'Token {i}')
    plt.legend()
    plt.show()

# Dự đoán giá trị mới
# Assuming you have already trained and saved the model

# Load the model
model = tf.keras.models.load_model('token_price_prediction_model.h5')

# Assuming you have new data to predict
new_data = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

# Reshape the new data to match the input shape of the model
new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[1], 1))

# Predict the new values
predictions = model.predict(new_data)

# Inverse scaling the predictions if necessary
predictions = scaler.inverse_transform(predictions)

# Print the predicted values
print(predictions)

