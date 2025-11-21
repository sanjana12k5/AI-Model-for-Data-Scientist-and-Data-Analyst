# src/profit_loss.py

import pandas as pd

def calculate_financials(df, revenue_col="amount", cost_col=None, fraud_col="label"):
    total_rev = df[revenue_col].sum()

    if cost_col:
        total_cost = df[cost_col].sum()
    else:
        total_cost = total_rev * 0.1  # default assumption

    fraud_loss = df[df[fraud_col] == 1][revenue_col].sum()

    profit = total_rev - total_cost - fraud_loss

    return {
        "total_revenue": total_rev,
        "total_cost": total_cost,
        "fraud_loss": fraud_loss,
        "net_profit": profit
    }
