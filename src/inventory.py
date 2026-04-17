import numpy as np
from scipy.stats import norm

def inventory_policy(forecast, lead_time=7, service_level=0.95, on_hand=200):
    print("📦 Calculating inventory...")

    z = norm.ppf(service_level)

    demand_mean = np.mean(forecast[:lead_time])
    demand_std = np.std(forecast[:lead_time])

    safety_stock = z * demand_std
    reorder_point = demand_mean + safety_stock
    order_qty = max(0, reorder_point - on_hand)

    return {
        "Safety Stock": float(round(safety_stock, 2)),
        "Reorder Point": float(round(reorder_point, 2)),
        "Order Quantity": float(round(order_qty, 2))
    }