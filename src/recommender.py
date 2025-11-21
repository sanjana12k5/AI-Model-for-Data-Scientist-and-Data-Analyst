# src/recommender.py

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class ProductEmbeddingModel(nn.Module):
    def __init__(self, n_users, n_items, dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

    def forward(self, u, i):
        return (self.user_emb(u) * self.item_emb(i)).sum(1)

def recommend_products(model, user_id, top_k=5):
    user_vec = model.user_emb(torch.tensor([user_id]))
    item_vecs = model.item_emb.weight.data
    scores = (item_vecs @ user_vec.T).squeeze()
    top_items = torch.topk(scores, top_k).indices.tolist()
    return top_items
