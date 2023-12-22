import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('./data/ckd-20231129-084117.csv')

# Define features and target
features = ["TransactionBuyM5", "TransactionSellM5", "TransBuyM5ChangePercent", "TransSellM5ChangePercent", "VolumeM5", "VolumeM5ChangePercent"]
target = ["TargetTP100M15"]

feature_cols = features + target

dataAI = data[feature_cols]
dataAI.dropna(inplace = True)
dataAI.dropna(axis = 0)
dataAI.info()

X = dataAI[features]
Y = dataAI[target].values.ravel()  # Use ravel() to convert Y to a 1D array

# Split the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.7, random_state=42)

# Scale the features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions and evaluate the model
Y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(Y_test, Y_pred))
# print("Classification Report:\n", classification_report(Y_test, Y_pred, zero_division=0))  # Add zero_division=0 to handle the warning
# print("First 10 Predictions:", Y_pred[:10])
print("\nFirst Predictions:")
for i in range(1):
    print(f"Actual: {Y_test[i]}, Predicted: {Y_pred[i]}")

# Save the model
joblib.dump(model, 'logistic_regression_model.pkl')