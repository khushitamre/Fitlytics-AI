import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# data load
df = pd.read_csv("gym.csv")

# only 3 simple features (IMPORTANT)
X = df[["Session_Duration (hours)", "Avg_BPM", "Resting_BPM"]]
y = df["Calories_Burned"]

# model train
model = RandomForestRegressor()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")