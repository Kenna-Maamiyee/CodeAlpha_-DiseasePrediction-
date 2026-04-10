import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# LOAD DATA
df = pd.read_csv("heart.csv")

# FIX TARGET
if df["Heart Disease"].dtype == "object":
    df["Heart Disease"] = df["Heart Disease"].map({
        "Yes": 1,
        "No": 0,
        "Presence": 1,
        "Absence": 0
    })

df = df.dropna()

X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

# TRAIN MODEL
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# SAVE MODEL + FEATURES
joblib.dump(model, "heart_model.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("✅ Model trained and saved")