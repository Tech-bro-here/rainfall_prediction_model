import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# laod the dataset to a pandas dataframe
data = pd.read_csv("Rainfall.csv")

print(type(data))
data.shape
data.head()
data.tail()
print("Data Info:")
data.info()
data.columns
# remove extra  spaces in all columns
data.columns = data.columns.str.strip()
data.columns
print("Data Info:")
data.info()
data = data.drop(columns=["day"])
data.head()
# checking the number of missing values
print(data.isnull().sum())
data["winddirection"].unique()
# handle missing values
data["winddirection"] = data["winddirection"].fillna(data["winddirection"].mode()[0])
data["windspeed"] = data["windspeed"].fillna(data["windspeed"].median())
# checking the number of missing values
print(data.isnull().sum())
data["rainfall"].unique()
# converting the yes & no to 1 and 0 respectively
data["rainfall"] = data["rainfall"].map({"yes": 1, "no": 0})
data.head()
# EDA
sns.set(style="whitegrid")
data.describe()
plt.figure(figsize=(15, 10))
for i, column in enumerate(['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity','cloud', 'sunshine', 'windspeed'], 1):
    plt.subplot(3, 3, i)
    sns.histplot(data[column], kde=True)
    plt.title(f"Distribution of {column}")
plt.tight_layout()
plt.show()
print(data["rainfall"].value_counts())
# separate majority and minority class
df_majority = data[data["rainfall"] == 1]
df_minority = data[data["rainfall"] == 0]
print(df_majority.shape)
print(df_minority.shape)
# downsample majority class to match minority count
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(df_minority), random_state=42)
df_majority_downsampled.shape
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled.shape
df_downsampled.head()
# shuffle the final dataframe
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
df_downsampled.head()
df_downsampled["rainfall"].value_counts()
# splitting the data into features and target
X = df_downsampled.drop("rainfall", axis=1)
y = df_downsampled["rainfall"]
print(y)
# splitting the data into training data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model Training
rf_model = RandomForestClassifier(random_state=42)
param_grid_rf = {
    "n_estimators": [50, 100, 200],
    "max_features": ["sqrt", "log2"],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
grid_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_rf.fit(X_train, y_train)
best_rf_model = grid_rf.best_estimator_
y_pred = best_rf_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Save the model
with open("rainfall_rf_model.pkl", "wb") as f:
    pickle.dump({"model": best_rf_model, "feature_names": list(X.columns)}, f)
# Load and predict example
with open("rainfall_rf_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
feature_names = model_data["feature_names"]
input_data = (1015.9, 19.9, 95, 81, 0.0, 40.0, 13.7)
input_df = pd.DataFrame([input_data], columns=feature_names)
prediction = best_rf_model.predict(input_df)
print("Prediction result:", "Rainfall" if prediction[0] == 1 else "No Rainfall") 