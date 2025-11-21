# src/generate_data.py
"""
Generates synthetic transaction dataset and saves to data/transactions.csv
Run: python src/generate_data.py
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta, timezone

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

def generate_transactions(n_users=2000, n_merchants=1000, n_rows=50000, seed=42):
    rng = np.random.RandomState(seed)
    user_ids = rng.randint(1, n_users+1, size=n_rows)
    merchant_ids = rng.randint(1, n_merchants+1, size=n_rows)

    # time series spanning last 90 days (timezone-aware, no warnings)
    start = datetime.now(timezone.utc) - timedelta(days=90)
    ts = [start + timedelta(seconds=int(rng.rand()*90*24*3600)) for _ in range(n_rows)]

    # amounts: mixture of log-normal
    amounts = np.exp(rng.normal(loc=4.0, scale=1.2, size=n_rows))  # skewed positive

    # corrected probabilities (sum to exactly 1.0)
    country_probs = np.array([0.3,0.2,0.08,0.06,0.06,0.06,0.05,0.05,0.07,0.03])
    country_probs = country_probs / country_probs.sum()

    countries = rng.choice(
        ["US","IN","GB","FR","DE","CA","AU","BR","NG","ZA"],
        size=n_rows,
        p=country_probs
    )

    device = rng.choice(["android","ios","web"], size=n_rows, p=[0.55,0.35,0.10])

    country_risk = {c: r for c,r in zip(
        ["US","IN","GB","FR","DE","CA","AU","BR","NG","ZA"],
        [0.01,0.04,0.02,0.015,0.015,0.01,0.01,0.03,0.06,0.025]
    )}

    base_prob = 0.005
    probs = []
    seen_pairs = set()

    for i in range(n_rows):
        u = user_ids[i]
        m = merchant_ids[i]
        amt = amounts[i]
        c = countries[i]
        prob = base_prob

        prob += min(0.0006 * (amt/100.0), 0.05)
        prob += country_risk.get(c, 0.01)

        if (u,m) not in seen_pairs:
            prob += 0.02
            seen_pairs.add((u,m))

        prob = min(prob, 0.9)
        probs.append(prob)

    labels = rng.binomial(1, probs)

    df = pd.DataFrame({
        "transaction_id": np.arange(1, n_rows+1),
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "ts": ts,
        "amount": np.round(amounts,2),
        "country": countries,
        "device_type": device,
        "label": labels
    })

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

if __name__ == "__main__":
    print("Generating synthetic data...")
    df = generate_transactions(n_users=2000, n_merchants=1000, n_rows=50000, seed=42)
    out_path = os.path.join(OUT_DIR, "transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
