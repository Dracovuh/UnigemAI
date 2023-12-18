import pandas as pd

# Load the dataset
data = pd.read_csv('demo/ckd-20231208-041519.csv')

# Display the first few rows of the dataframe
print(data.head())

# Example: Assuming 'price' is your target variable
data['target'] = data['price'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)

# Handle missing values - Fill or drop
data.fillna(method='ffill', inplace=True)

# Select your features - Adjust this based on the actual feature columns
feature_columns = ['TransactionBuyM5', 'TransactionSellM5', 'VolumeM5', ...]  # Add your feature columns here

# Splitting features and target
X = data[feature_columns]
y = data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")