# src/train_deep.py
"""
Robust deep-learning trainer for fraud detection.
Call: train_model(df, mappings)
- df: pandas DataFrame (raw or already renamed)
- mappings: dict mapping actual_column_name -> canonical name, e.g.
    { "CustomerID": "user_id", "Vendor": "merchant_id", "TxnAmt": "amount", ... }
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------------
# Dataset
# -------------------------
class FraudDataset(Dataset):
    def __init__(self, df, num_cols, user_map, merchant_map, country_map, device_map, label_col="label"):
        # require 0..N-1 integer index to align with torch indexing
        df = df.reset_index(drop=True).copy()

        # store arrays (positional)
        self.num = df[num_cols].values.astype(np.float32)
        self.user = df["user_id"].map(user_map).astype(np.int64).values
        self.merchant = df["merchant_id"].map(merchant_map).astype(np.int64).values
        self.country = df["country"].map(country_map).astype(np.int64).values
        self.device = df["device_type"].map(device_map).astype(np.int64).values
        self.y = df[label_col].astype(np.float32).values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.num[idx],
            int(self.user[idx]),
            int(self.merchant[idx]),
            int(self.country[idx]),
            int(self.device[idx]),
            float(self.y[idx])
        )

def collate_fn(batch):
    num, user, merchant, country, device, y = zip(*batch)
    return (
        torch.tensor(np.stack(num), dtype=torch.float32),
        torch.tensor(user, dtype=torch.long),
        torch.tensor(merchant, dtype=torch.long),
        torch.tensor(country, dtype=torch.long),
        torch.tensor(device, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32)
    )

# -------------------------
# Model
# -------------------------
class FraudModel(nn.Module):
    def __init__(self, n_num, n_users, n_merchants, n_country, n_device, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, min(emb_dim, max(1, n_users)))
        self.merchant_emb = nn.Embedding(n_merchants, min(emb_dim, max(1, n_merchants)))
        self.country_emb = nn.Embedding(n_country, min(8, max(1, n_country)))
        self.device_emb = nn.Embedding(n_device, min(8, max(1, n_device)))

        emb_total = self.user_emb.embedding_dim + self.merchant_emb.embedding_dim + \
                    self.country_emb.embedding_dim + self.device_emb.embedding_dim

        self.fc1 = nn.Linear(n_num + emb_total, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, num, user, merchant, country, device):
        u = self.user_emb(user)
        m = self.merchant_emb(merchant)
        c = self.country_emb(country)
        d = self.device_emb(device)
        x = torch.cat([num, u, m, c, d], dim=1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return torch.sigmoid(self.out(x)).squeeze(1)

# -------------------------
# Main training function
# -------------------------
def train_model(df, mappings):
    """
    df: pandas DataFrame (raw or already renamed)
    mappings: dict actual_col -> canonical_col, required canonical names:
      user_id, merchant_id, amount, country, device_type, label
    """
    # Defensive copy
    df = df.copy()

    # If mappings provided, perform rename (no-op for keys not present)
    if mappings:
        df = df.rename(columns=mappings)

    # Check required canonical columns exist
    required = ["user_id", "merchant_id", "amount", "country", "device_type", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after mapping: {missing}")

    # Reset index immediately to ensure positional indexing
    df = df.reset_index(drop=True)

    # -------------------------
    # Basic coercions / cleaning
    # -------------------------
    # IDs and categories -> string
    for c in ["user_id", "merchant_id", "country", "device_type"]:
        df[c] = df[c].astype(str).fillna("UNK")

    # Amount -> numeric float
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(",", ""), errors="coerce").fillna(0.0).astype(float)

    # Label -> binary int (0/1)
    df["label"] = df["label"].replace({"yes":1,"no":0,"True":1,"False":0,"true":1,"false":0})
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    df["label"] = df["label"].clip(0,1)

    # -------------------------
    # Build maps for categories
    # -------------------------
    user_vals = sorted(df["user_id"].unique())
    merchant_vals = sorted(df["merchant_id"].unique())
    country_vals = sorted(df["country"].unique())
    device_vals = sorted(df["device_type"].unique())

    user_map = {v: i for i, v in enumerate(user_vals)}
    merchant_map = {v: i for i, v in enumerate(merchant_vals)}
    country_map = {v: i for i, v in enumerate(country_vals)}
    device_map = {v: i for i, v in enumerate(device_vals)}

    # -------------------------
    # Numeric features
    # -------------------------
    num_cols = ["amount"]

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # -------------------------
    # Train / Val split (stratify by label if possible)
    # -------------------------
    if df["label"].nunique() > 1:
        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    else:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Reset indices for datasets (VERY IMPORTANT)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # -------------------------
    # Datasets & loaders
    # -------------------------
    train_ds = FraudDataset(train_df, num_cols, user_map, merchant_map, country_map, device_map, label_col="label")
    val_ds = FraudDataset(val_df, num_cols, user_map, merchant_map, country_map, device_map, label_col="label")

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=2048, shuffle=False, collate_fn=collate_fn)

    # -------------------------
    # Model
    # -------------------------
    model = FraudModel(
        n_num=len(num_cols),
        n_users=len(user_map),
        n_merchants=len(merchant_map),
        n_country=len(country_map),
        n_device=len(device_map)
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    best_auc = 0.0

    # -------------------------
    # Training loop
    # -------------------------
    epochs = 6
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for num, user, merchant, country, device, y in train_loader:
            num = num.to(DEVICE)
            user = user.to(DEVICE)
            merchant = merchant.to(DEVICE)
            country = country.to(DEVICE)
            device = device.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            preds = model(num, user, merchant, country, device)
            loss = loss_fn(preds, y)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        preds_all = []
        y_all = []
        with torch.no_grad():
            for num, user, merchant, country, device, y in val_loader:
                num = num.to(DEVICE)
                user = user.to(DEVICE)
                merchant = merchant.to(DEVICE)
                country = country.to(DEVICE)
                device = device.to(DEVICE)
                out = model(num, user, merchant, country, device)
                preds_all.append(out.cpu().numpy())
                y_all.append(y.numpy())

        preds_all = np.concatenate(preds_all)
        y_all = np.concatenate(y_all)

        # safe AUC calculation — requires both classes present
        try:
            auc = roc_auc_score(y_all, preds_all)
        except Exception:
            auc = 0.0

        avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
        print(f"Epoch {epoch} | train_loss {avg_loss:.4f} | val_auc {auc:.4f}")

        # Save best
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "model.pt"))
            joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))
            # metadata for app.py
            meta = {
                "mappings": mappings,
                "num_cols": num_cols,
                "user_map": user_map,
                "merchant_map": merchant_map,
                "country_map": country_map,
                "device_map": device_map
            }
            joblib.dump(meta, os.path.join(OUT_DIR, "metadata.pkl"))
            print("Saved best model with AUC:", best_auc)

    print("Training finished. Best AUC:", best_auc)
    return best_auc


# If executed directly for debug / CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/transactions.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    # a minimal mapping for synthetic data names
    mappings = {}
    if "user_id" not in df.columns:
        # no mapping provided, try to auto-detect common names
        auto = {}
        colmap = {c.lower(): c for c in df.columns}
        if "customerid" in colmap: auto[colmap["customerid"]] = "user_id"
        if "userid" in colmap: auto[colmap["userid"]] = "user_id"
        if "merchant_id" not in df.columns and "merchantid" in colmap: auto[colmap["merchantid"]] = "merchant_id"
        if "amount" not in df.columns:
            if "txn_amt" in colmap: auto[colmap["txn_amt"]] = "amount"
            if "transaction_amount" in colmap: auto[colmap["transaction_amount"]] = "amount"
        mappings = auto
        print("Auto mappings used:", mappings)
    train_model(df, mappings)
