from fastapi import FastAPI
import numpy as np
import pandas as pd

from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.model import train_model
from src.inventory import inventory_policy

app = FastAPI()

print("🚀 Starting API...")

# ================= LOAD ONCE =================
df = load_data()
df = df[(df["store_id"] == 1) & (df["item_id"] == 1)]
df = create_features(df)

model, preds, actual, features = train_model(df)

# ================= ROUTES =================

@app.get("/")
def home():
    return {"message": "Retail Forecast API Running"}

@app.get("/forecast")
def forecast():
    return {
        "forecast_sample": list(preds[:10])
    }

@app.get("/inventory")
def inventory():
    inv = inventory_policy(preds)

    return {
        "Safety Stock": float(inv["Safety Stock"]),
        "Reorder Point": float(inv["Reorder Point"]),
        "Order Quantity": float(inv["Order Quantity"])
    }

@app.get("/metrics")
def metrics():
    errors = actual.values - preds

    return {
        "MAE": float(np.mean(abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2)))
    }