# src/stats_analysis.py

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

def run_ttest(df, col1, col2):
    return sm.stats.ttest_ind(df[col1], df[col2])

def chi_square(df, col1, col2):
    table = pd.crosstab(df[col1], df[col2])
    chi2, p, _, _ = sm.stats.Table(table).test_nominal_association()
    return chi2, p

def run_forecast(df, col):
    model = ARIMA(df[col].dropna(), order=(3,1,3))
    fit = model.fit()
    return fit.forecast(10)
