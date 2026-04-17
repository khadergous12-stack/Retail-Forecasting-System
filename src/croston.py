import numpy as np

def croston(y, alpha=0.1, h=7):
    y = np.array(y)

    demand = y[y > 0]

    if len(demand) == 0:
        return np.zeros(h)

    intervals = np.diff(np.where(y > 0)[0], prepend=0)

    z_hat = demand[0]
    p_hat = intervals[1] if len(intervals) > 1 else 1

    for i in range(1, len(demand)):
        z_hat = alpha * demand[i] + (1 - alpha) * z_hat

    for i in range(1, len(intervals)):
        p_hat = alpha * intervals[i] + (1 - alpha) * p_hat

    forecast = (z_hat / p_hat) * np.ones(h)
    return forecast