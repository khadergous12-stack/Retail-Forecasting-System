from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"📊 MAE  : {round(mae,2)}")
    print(f"📊 RMSE : {round(rmse,2)}")

    return model, preds, y_test, features