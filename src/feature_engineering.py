def create_features(df):
    print("⚙️ Creating features...")

    df = df.sort_values("date")

    df["lag_1"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(1)
    df["lag_4"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(4)

    df["rolling_mean"] = df.groupby(["store_id", "item_id"])["qty_sold"].shift(1).rolling(4).mean()

    df["month"] = df["date"].dt.month
    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    df.dropna(inplace=True)

    print("✅ Features created")
    return df