
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle


data = pd.read_csv("nsap_dataset.csv")
print(data.head())

print(data.isnull().sum())
data = data.dropna()

label_encoders = {}
for column in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le


X = data.drop("NSAP_Eligible", axis=1)   # Replace with actual target column name
y = data["NSAP_Eligible"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:\n", classification_report(y_test, y_pred))


with open("nsap_model.pkl", "wb") as file:
    pickle.dump(model, file)
print("✅ Model saved to nsap_model.pkl")


with open("nsap_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

sample_input = X_test.iloc[0:1]
prediction = loaded_model.predict(sample_input)
print("Prediction:", prediction)



![NSAP Snapshot] 
<img width="1883" height="953" alt="Screenshot 2025-07-24 151415" src="https://github.com/user-attachments/assets/c0ee1dc3-82dc-4244-bf6f-84bc33bb4cdd" />
<img width="1696" height="849" alt="Screenshot 2025-07-24 151339" src="https://github.com/user-attachments/assets/331253c0-5cbb-4120-86ec-bc7b0a9c4f27" />
<img width="1857" height="982" alt="Screenshot 2025-07-24 151238" src="https://github.com/user-attachments/assets/a75af016-4cbc-4c36-8ba7-6a5f3f8dbc65" />



