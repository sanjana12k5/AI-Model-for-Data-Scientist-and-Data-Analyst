# src/eda_full.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

def run_eda(df, name="dataset"):
    report = {}

    # BASIC INFO
    report["shape"] = df.shape
    report["missing"] = df.isnull().sum().to_dict()
    report["describe"] = df.describe(include="all")

    # SAVE SUMMARY
    report["describe"].to_csv(OUT / f"{name}_summary.csv")

    # DISTRIBUTIONS
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(OUT / f"{col}_dist.png")
        plt.close()

    # CORRELATION
    if len(num_cols) > 1:
        plt.figure(figsize=(10,6))
        sns.heatmap(df[num_cols].corr(), annot=False, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.savefig(OUT / "correlation_heatmap.png")
        plt.close()

    return report
