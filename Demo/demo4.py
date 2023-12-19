from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the data
df = pd.read_csv('ckd-20231208-041519.csv')

# Selecting relevant features and the target variable
features = ['TransactionBuyM5', 'TransactionSellM5', 'TransBuyM5Change', 'TransSellM5Change', 'VolumeM5', 'VolumeM5Change']
X = df[features]
y = df['TargetTP100M15']  # This should be a binary variable (0 or 1)

# Preprocessing the data (Handle missing values if any)
# Example: X.fillna(X.mean(), inplace=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
