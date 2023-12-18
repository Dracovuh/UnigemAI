import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Tạo dữ liệu giả lập
data = {
    'Glucose': [130, 115, 140, 110, 135, 120],
    'BloodPressure': [80, 70, 82, 75, 85, 65],
    'Diabetes': [1, 0, 1, 0, 1, 0]  # 1 = Có tiểu đường, 0 = Không có tiểu đường
}

df = pd.DataFrame(data)

# Phân chia dữ liệu thành tính năng (features) và nhãn (label)
X = df[['Glucose', 'BloodPressure']]
y = df['Diabetes']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Logistic Regression
model = LogisticRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập dữ liệu kiểm tra
predictions = model.predict(X_test)

# Đánh giá mô hình
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
