from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ================= MASE =================
def mase(y_true, y_pred, y_naive):
    denom = np.mean(np.abs(y_true - y_naive))
    return np.mean(np.abs(y_true - y_pred)) / denom if denom != 0 else np.nan


# ================= MODEL =================
def train_model(df):
    print("🤖 Training model...")

    features = [
        "lag_1", "lag_4", "rolling_mean",
        "month", "week",
        "Temperature", "Fuel_Price",
        "CPI", "Unemployment",
        "IsHoliday"
    ]

    X = df[features]
    y = df["qty_sold"]

    # ✅ TIME SERIES SPLIT (better than random split)
    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=14,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # ================= METRICS =================
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # naive forecast (last value)
    naive = np.repeat(y_train.iloc[-1], len(y_test))
    mase_score = mase(y_test.values, preds, naive)

    print(f"📊 MAE   : {round(mae,2)}")
    print(f"📊 RMSE  : {round(rmse,2)}")
    print(f"📊 MASE  : {round(mase_score,3)}")

    return model, preds, y_test, features