# app.py
#############################################################
#                STREAMLIT DATA SCIENCE SUITE               #
#       Deep Learning · EDA · Stats · Recs · Financials     #
#                      No APIs. Local only.                 #
#############################################################

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import traceback

# allow importing train_deep from src/
SRC_DIR = Path("src")
if str(SRC_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(SRC_DIR.resolve()))
try:
    import train_deep  # src/train_deep.py must define train_model(df, mappings)
except Exception:
    train_deep = None

# Directories
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Deep Data Science Suite (final)", layout="wide")

# -------------------------
# Synthetic generator
# -------------------------
def generate_synthetic_data(rows=20000):
    rng = np.random.RandomState(42)
    user_ids = rng.randint(1, 2000, size=rows)
    merchant_ids = rng.randint(1, 1000, size=rows)
    start = datetime.utcnow()
    ts = [start for _ in range(rows)]
    amounts = np.exp(rng.normal(4, 1.2, size=rows))
    countries = rng.choice(["US","IN","GB","FR","DE","CA","AU","BR","NG","ZA"],
                           size=rows, p=[0.3,0.2,0.08,0.06,0.06,0.06,0.05,0.05,0.07,0.03])
    devices = rng.choice(["android","ios","web"], size=rows, p=[0.55,0.35,0.10])
    p_base = 0.01
    probs = p_base + (amounts/2000).clip(0,0.2)
    probs += (countries=="NG")*0.05
    probs += (countries=="BR")*0.03
    labels = rng.binomial(1, probs.clip(0,0.8))
    df = pd.DataFrame({
        "transaction_id": np.arange(1, rows+1),
        "user_id": user_ids,
        "merchant_id": merchant_ids,
        "ts": ts,
        "amount": np.round(amounts,2),
        "country": countries,
        "device_type": devices,
        "label": labels
    })
    df.to_csv(DATA_DIR/"transactions.csv", index=False)
    return df

# -------------------------
# Local model class (for prediction / recommendations)
# -------------------------
class FraudModel(nn.Module):
    def __init__(self, n_num, n_users, n_merchants, n_country, n_device):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, 32)
        self.merchant_emb = nn.Embedding(n_merchants, 32)
        self.country_emb = nn.Embedding(n_country, 8)
        self.device_emb = nn.Embedding(n_device, 8)
        emb_dim = 32 + 32 + 8 + 8
        self.fc1 = nn.Linear(n_num + emb_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
    def forward(self, num, user, merchant, country, device):
        x = torch.cat([
            num,
            self.user_emb(user),
            self.merchant_emb(merchant),
            self.country_emb(country),
            self.device_emb(device)
        ], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))

# -------------------------
# Utilities
# -------------------------
def financial_report(df):
    st.subheader("💰 Select Revenue Column")
    possible_cols = ["amount", "amt", "price", "value", "transaction_amount", "revenue", "sales", "total"]
    matches = [c for c in df.columns if c.lower() in possible_cols]
    default_col = matches[0] if matches else df.columns[0]
    revenue_col = st.selectbox("Which column represents transaction amount / revenue?", df.columns,
                               index=df.columns.get_loc(default_col))
    rev = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
    total_revenue = float(rev.sum())
    if "label" in df.columns:
        fraud_loss = float(pd.to_numeric(df.loc[df["label"]==1, revenue_col], errors="coerce").fillna(0).sum())
    else:
        fraud_loss = 0.0
    total_cost = float(total_revenue * 0.10)
    net_profit = float(total_revenue - total_cost - fraud_loss)
    return {
        "Revenue Column Used": revenue_col,
        "Total Revenue": total_revenue,
        "Total Cost": total_cost,
        "Fraud Loss": fraud_loss,
        "Net Profit": net_profit,
    }

def recommend_products_from_model(model, meta, user_id, top_k=5):
    if user_id not in meta["user_map"]:
        return []
    uidx = meta["user_map"][user_id]
    uvec = model.user_emb(torch.tensor([uidx])).detach()
    item_vecs = model.merchant_emb.weight.data
    scores = (item_vecs @ uvec.T).squeeze()
    top = torch.topk(scores, top_k).indices.tolist()
    inv = {v:k for k,v in meta["merchant_map"].items()}
    return [inv[i] for i in top]

# -------------------------
# UI layout
# -------------------------
st.title("🧠 Deep Data Science Suite — Final (Local)")
tabs = st.tabs(["📥 Data","📊 EDA","📈 Stats","🤖 Train","🔮 Predict","🎯 Recs","💰 Profit/Loss"])

# --- Data tab ---
with tabs[0]:
    st.header("📥 Upload or generate dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Generate synthetic dataset (20k rows)"):
        df = generate_synthetic_data(rows=20000)
        st.success("Synthetic data generated and saved to data/transactions.csv")
        st.dataframe(df.head())
    elif uploaded:
        df = pd.read_csv(uploaded)
        df.to_csv(DATA_DIR/"uploaded.csv", index=False)
        st.success("Uploaded and saved to data/uploaded.csv")
        st.dataframe(df.head())
    else:
        if (DATA_DIR/"transactions.csv").exists():
            df = pd.read_csv(DATA_DIR/"transactions.csv")
            st.info("Loaded synthetic dataset (data/transactions.csv)")
            st.dataframe(df.head())
        elif (DATA_DIR/"uploaded.csv").exists():
            df = pd.read_csv(DATA_DIR/"uploaded.csv")
            st.info("Loaded previously uploaded dataset (data/uploaded.csv)")
            st.dataframe(df.head())
        else:
            st.info("No dataset loaded — upload a CSV or generate synthetic data.")

# --- EDA tab ---
with tabs[1]:
    st.header("📊 Exploratory Data Analysis")
    if "df" in locals():
        st.write("Shape:", df.shape)
        st.write(df.describe(include="all"))
        st.write("Missing values:")
        st.write(df.isnull().sum())
        num_cols = df.select_dtypes("number").columns.tolist()
        for c in num_cols:
            st.subheader(f"Distribution: {c}")
            fig, ax = plt.subplots()
            sns.histplot(df[c], kde=True, ax=ax)
            st.pyplot(fig)
        if len(num_cols) > 1:
            st.subheader("Correlation")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.heatmap(df[num_cols].corr(), cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Load or generate a dataset first on the Data tab.")

# --- Stats tab ---
with tabs[2]:
    st.header("📈 Statistical tests")
    if "df" in locals():
        num_cols = df.select_dtypes("number").columns.tolist()
        if len(num_cols) >= 2:
            c1 = st.selectbox("Column 1", num_cols)
            c2 = st.selectbox("Column 2", num_cols, index=1 if len(num_cols) > 1 else 0)
            if st.button("Run t-test"):
                st.write(sm.stats.ttest_ind(df[c1], df[c2]))
    else:
        st.warning("Load or generate a dataset first.")

# --- Train tab ---
with tabs[3]:
    st.header("🤖 Train Deep Model (select columns first)")
    if "df" in locals():
        st.subheader("Select which columns map to the canonical fields")
        user_col = st.selectbox("User column", df.columns)
        merchant_col = st.selectbox("Merchant/Product column", df.columns)
        amount_col = st.selectbox("Amount / price column (numeric)", df.columns)
        country_col = st.selectbox("Country column", df.columns)
        device_col = st.selectbox("Device column", df.columns)
        label_col = st.selectbox("Label / target (0/1)", df.columns)

        mappings = {
            user_col: "user_id",
            merchant_col: "merchant_id",
            amount_col: "amount",
            country_col: "country",
            device_col: "device_type",
            label_col: "label"
        }

        st.write("Selected mappings:", mappings)

        if st.button("Start training (this will call src/train_deep.train_model)"):
            if train_deep is None:
                st.error("train_deep module not found. Ensure src/train_deep.py exists and defines train_model(df, mappings).")
            else:
                df_mapped = df.rename(columns=mappings).copy()

                # Debug info
                st.subheader("Debug: column types and examples (preprocessing)")
                st.write(df_mapped.dtypes)
                sample_display = {}
                for col in ["user_id","merchant_id","amount","country","device_type","label"]:
                    if col in df_mapped.columns:
                        sample_display[col] = df_mapped[col].head(10).tolist()
                st.write(sample_display)

                # Safe coercions
                for c in ["user_id","merchant_id","country","device_type"]:
                    if c in df_mapped.columns:
                        df_mapped[c] = df_mapped[c].astype(str).fillna("UNK")
                if "amount" in df_mapped.columns:
                    df_mapped["amount"] = pd.to_numeric(df_mapped["amount"].astype(str).str.replace(",",""), errors="coerce").fillna(0.0).astype(float)
                if "label" in df_mapped.columns:
                    df_mapped["label"] = df_mapped["label"].replace({"yes":1,"no":0,"True":1,"False":0,"true":1,"false":0})
                    df_mapped["label"] = pd.to_numeric(df_mapped["label"], errors="coerce").fillna(0).astype(int)

                st.write("After coercion dtypes:")
                st.write(df_mapped.dtypes)
                if "label" in df_mapped.columns and df_mapped["label"].nunique() <= 1:
                    st.warning("Label column has <= 1 unique value after coercion — model training will not be meaningful.")

                try:
                    with st.spinner("Training (this may take several minutes)..."):
                        best_auc = train_deep.train_model(df_mapped, mappings)
                        joblib.dump(mappings, MODEL_DIR/"column_mappings.pkl")
                        st.success(f"Training finished. Best AUC: {best_auc:.4f}")
                except Exception as e:
                    st.error(f"❌ Training failed: {type(e).__name__}")
                    st.error(str(e))
                    st.text("Full traceback:")
                    st.text(traceback.format_exc())
    else:
        st.warning("Load or generate a dataset first on the Data tab.")

# --- Predict tab ---
with tabs[4]:
    st.header("🔮 Predict (use trained model)")
    model_path = MODEL_DIR/"model.pt"
    meta_path = MODEL_DIR/"metadata.pkl"
    scaler_path = MODEL_DIR/"scaler.pkl"

    if model_path.exists() and meta_path.exists() and scaler_path.exists():
        meta = joblib.load(meta_path)
        scaler = joblib.load(scaler_path)

        n_num = len(meta["num_cols"])
        n_users = len(meta["user_map"])
        n_merchants = len(meta["merchant_map"])
        n_country = len(meta["country_map"])
        n_device = len(meta["device_map"])

        model = FraudModel(n_num, n_users, n_merchants, n_country, n_device)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        st.write("Enter values to predict (these map to your selected columns when training).")
        amount_val = st.number_input("Amount", value=100.0)
        user_val = st.selectbox("User ID", sorted(list(meta["user_map"].keys())))
        merchant_val = st.selectbox("Merchant ID", sorted(list(meta["merchant_map"].keys())))
        country_val = st.selectbox("Country", sorted(list(meta["country_map"].keys())))
        device_val = st.selectbox("Device", sorted(list(meta["device_map"].keys())))

        if st.button("Predict"):
            num_arr = np.array([[amount_val]])
            num_scaled = scaler.transform(num_arr)
            num_t = torch.tensor(num_scaled, dtype=torch.float32)
            u_idx = meta["user_map"].get(user_val, list(meta["user_map"].values())[0])
            m_idx = meta["merchant_map"].get(merchant_val, list(meta["merchant_map"].values())[0])
            c_idx = meta["country_map"][country_val]
            d_idx = meta["device_map"][device_val]
            u_t = torch.tensor([u_idx], dtype=torch.long)
            m_t = torch.tensor([m_idx], dtype=torch.long)
            c_t = torch.tensor([c_idx], dtype=torch.long)
            d_t = torch.tensor([d_idx], dtype=torch.long)
            with torch.no_grad():
                score = float(model(num_t, u_t, m_t, c_t, d_t).item())
            st.metric("Fraud probability", f"{score:.4f}")
    else:
        st.info("Trained model not found. Train a model first on the Train tab.")

# --- Recommendation tab ---
with tabs[5]:
    st.header("🎯 Recommendations (from trained embeddings)")
    model_path = MODEL_DIR/"model.pt"
    meta_path = MODEL_DIR/"metadata.pkl"
    if model_path.exists() and meta_path.exists():
        meta = joblib.load(meta_path)
        n_num = len(meta["num_cols"])
        n_users = len(meta["user_map"])
        n_merchants = len(meta["merchant_map"])
        n_country = len(meta["country_map"])
        n_device = len(meta["device_map"])
        model = FraudModel(n_num, n_users, n_merchants, n_country, n_device)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        user_id = st.selectbox("User id to recommend for", sorted(list(meta["user_map"].keys())))
        if st.button("Get recommendations"):
            recs = recommend_products_from_model(model, meta, user_id, top_k=5)
            st.write("Recommended merchant ids:", recs)
    else:
        st.info("Train a model first to use recommendations.")

# --- Profit/Loss tab ---
with tabs[6]:
    st.header("💰 Profit / Loss Analysis")
    if "df" in locals():
        report = financial_report(df)
        st.write(report)
    else:
        st.warning("Load or generate a dataset first.")
