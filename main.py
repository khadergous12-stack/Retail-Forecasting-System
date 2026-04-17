# ================= BACKEND FIX (IMPORTANT) =================
import matplotlib
matplotlib.use('Agg')  # prevents Tkinter errors

# ================= IMPORTS =================
from src.data_preprocessing import load_data
from src.feature_engineering import create_features
from src.model import train_model
from src.inventory import inventory_policy

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil

# ================= STYLE =================
sns.set_theme(style="whitegrid")

# ================= CLEAN OUTPUT FOLDER =================
if os.path.exists("outputs"):
    shutil.rmtree("outputs")
os.makedirs("outputs")

print("\n===== RETAIL FORECASTING SYSTEM =====\n")

# ================= LOAD DATA =================
print("[INFO] Loading data...")
df = load_data()

# ================= FILTER =================
print("[INFO] Selecting Store 1 - Item 1...")
df = df[(df["store_id"] == 1) & (df["item_id"] == 1)]

# ================= FEATURES =================
df = create_features(df)

# ================= MODEL =================
model, preds, actual, feature_names = train_model(df)

# ================= INVENTORY =================
inv = inventory_policy(preds)

print("\n===== INVENTORY RECOMMENDATION =====")
print(f"Safety Stock   : {inv['Safety Stock']}")
print(f"Reorder Point  : {inv['Reorder Point']}")
print(f"Order Quantity : {inv['Order Quantity']}")

# =========================================================
# GRAPH 1 — FORECAST VS ACTUAL
# =========================================================
plt.figure(figsize=(12,5))

n = min(40, len(actual), len(preds))

actual_plot = actual.values[:n]
pred_plot = preds[:n]
x = range(n)

plt.plot(x, actual_plot, color="#2E86C1", linewidth=3, label="Actual")
plt.plot(x, pred_plot, color="#E74C3C", linestyle="--", linewidth=3, label="Forecast")

plt.fill_between(x, actual_plot, pred_plot, color="#AED6F1", alpha=0.4)

plt.title("Sales Forecast vs Actual")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/forecast.png")
plt.close()

# =========================================================
# GRAPH 2 — TREND (WITH PEAK INDICATORS)
# =========================================================
trend = df.set_index("date")["qty_sold"].rolling(7).mean()

plt.figure(figsize=(12,5))

plt.plot(trend, color="#1ABC9C", linewidth=3)

# Highlight peak & low
plt.scatter(trend.idxmax(), trend.max(), color="red", s=80, label="Peak")
plt.scatter(trend.idxmin(), trend.min(), color="blue", s=80, label="Low")

plt.title("Weekly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/trend.png")
plt.close()

# =========================================================
# GRAPH 3 — DISTRIBUTION
# =========================================================
plt.figure(figsize=(8,5))

sns.histplot(df["qty_sold"], bins=25, kde=True, color="#5DADE2")

plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("outputs/distribution.png")
plt.close()

# =========================================================
# GRAPH 4 — RESIDUALS WITH INDICATORS
# =========================================================
errors = actual_plot - pred_plot

plt.figure(figsize=(10,4))

plt.plot(errors, color="#8E44AD", linewidth=3, label="Error")

# Zero line
plt.axhline(0, linestyle="--", color="black", linewidth=1.5)

# Mean error
mean_error = errors.mean()
plt.axhline(mean_error, linestyle=":", color="orange", linewidth=2,
            label=f"Mean Error ({mean_error:.0f})")

# Highlight extreme errors
threshold = errors.std() * 1.5
extreme_idx = abs(errors) > threshold

plt.scatter(
    range(len(errors)),
    errors,
    color=["red" if x else "blue" for x in extreme_idx],
    s=60
)

# Fill zones
plt.fill_between(range(len(errors)), errors, 0,
                 where=(errors > 0),
                 color="green", alpha=0.2)

plt.fill_between(range(len(errors)), errors, 0,
                 where=(errors < 0),
                 color="red", alpha=0.2)

plt.title("Prediction Error Analysis")
plt.xlabel("Time")
plt.ylabel("Error")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/residuals.png")
plt.close()

# =========================================================
# GRAPH 5 — FEATURE IMPORTANCE (FIXED)
# =========================================================
print("\n[INFO] Generating Feature Importance...")

feat_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=True)

plt.figure(figsize=(10,6))

sns.barplot(
    data=feat_df,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="viridis",
    legend=False
)

plt.title("Key Drivers of Sales Prediction")

# Add values
for i, v in enumerate(feat_df["Importance"]):
    plt.text(v, i, f"{v:.3f}", va='center')

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# ================= FINAL =================
print("\n[INFO] All graphs saved in 'outputs/' folder")
print("[SUCCESS] Project Execution Completed Successfully!\n")