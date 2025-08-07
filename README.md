# 1. Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 2. Load Dataset
data = pd.read_csv("nsap_dataset.csv")
print(data.head())

# 3. Preprocessing
print(data.isnull().sum())
data = data.dropna()

label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# 4. Split Data
X = data.drop("NSAP_Eligible", axis=1)   # Replace with actual target column name
y = data["NSAP_Eligible"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate Model
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))

# 7. Save Model
with open("nsap_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("✅ Model saved to nsap_model.pkl")

# 8. Load & Predict (Later use)
with open("nsap_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

sample_input = X_test.iloc[0:1]
prediction = loaded_model.predict(sample_input)
print("Prediction:", prediction)
