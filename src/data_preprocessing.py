import pandas as pd

def load_data():
    print("📥 Loading datasets...")

    train = pd.read_csv("data/train.csv", parse_dates=["Date"])
    features = pd.read_csv("data/features.csv", parse_dates=["Date"])
    stores = pd.read_csv("data/stores.csv")

    df = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores, on="Store", how="left")

    df.rename(columns={
        "Store": "store_id",
        "Dept": "item_id",
        "Date": "date",
        "Weekly_Sales": "qty_sold"
    }, inplace=True)

    df.fillna(0, inplace=True)
    df = df[df["qty_sold"] > 0]

    print(f"✅ Data Loaded: {len(df)} rows")
    return df