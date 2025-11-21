# src/test_model.py

import pandas as pd
import torch
from train_deep import FraudModel, collate_fn, FraudDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

def train_on_dataset(df, target):
    df = df.copy()
    num_cols = df.select_dtypes("number").columns.drop(target)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    train_df, val_df = train_test_split(df, test_size=0.2)

    train_ds = FraudDataset(train_df, num_cols)
    val_ds = FraudDataset(val_df, num_cols)

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=512, collate_fn=collate_fn)

    model = FraudModel(len(num_cols))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(5):
        for num_batch, y_batch in train_loader:
            opt.zero_grad()
            preds = model(num_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            opt.step()

    # validation
    model.eval()
    preds_all = []
    y_all = []
    with torch.no_grad():
        for num_batch, y_batch in val_loader:
            preds_all.extend(model(num_batch).numpy())
            y_all.extend(y_batch.numpy())

    auc = roc_auc_score(y_all, preds_all)
    return model, auc
