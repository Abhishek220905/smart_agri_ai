import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

data = {
    "rainfall": [100, 200, 150, 300, 250],
    "fertilizer": [50, 60, 55, 70, 65],
    "area": [1, 2, 1.5, 3, 2.5],
    "yield": [2, 4, 3, 6, 5]
}

df = pd.DataFrame(data)

X = df[["rainfall", "fertilizer", "area"]]
y = df["yield"]

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "yield_model.pkl")

print("✅ Yield model trained and saved!")