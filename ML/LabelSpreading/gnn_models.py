"""GCN and GAT models for semi-supervised node classification on KNN graphs.

Pure PyTorch + scipy sparse implementation — no PyTorch Geometric dependency.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


def sparse_mx_to_torch(sparse_mx):
    """Convert scipy sparse matrix to PyTorch sparse tensor (COO format)."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col))).long()
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def normalize_adjacency(adj):
    """Symmetrically normalize adjacency: D^{-1/2} A D^{-1/2}."""
    adj = adj.tocoo()
    deg = np.array(adj.sum(axis=1)).ravel()
    deg_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    row, col = adj.row, adj.col
    norm_values = adj.data * deg_inv_sqrt[row] * deg_inv_sqrt[col]
    return sparse.csr_matrix((norm_values, (row, col)), shape=adj.shape)


class GCNLayer(nn.Module):
    """Single graph convolutional layer: H' = sigma(D^{-1/2} A D^{-1/2} H W)."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj_norm):
        support = x @ self.weight
        out = adj_norm @ support  # sparse @ dense
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """2-layer GCN (Kipf & Welling, 2017)."""
    def __init__(self, in_features, hidden=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNLayer(in_features, hidden)
        self.conv2 = GCNLayer(hidden, 1)  # single logit for binary classification
        self.dropout = dropout

    def forward(self, x, adj_norm):
        x = F.relu(self.conv1(x, adj_norm))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, adj_norm)
        return x.squeeze(-1)  # (N,) logits


class GATLayer(nn.Module):
    """Single graph attention layer (Velickovic et al., 2018)."""
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.3, alpha=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.dropout = dropout
        self.W = nn.Parameter(torch.FloatTensor(in_features, n_heads * out_features))
        self.a = nn.Parameter(torch.FloatTensor(1, n_heads, 2 * out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        """Simplified GAT: mean-pool attention over edges."""
        device = x.device
        N = x.size(0)
        Wh = x @ self.W  # (N, H*F')
        Wh = Wh.view(N, self.n_heads, self.out_features)  # (N, H, F')

        # Edge index from sparse adjacency tensor
        if hasattr(adj, 'tocoo'):  # scipy sparse matrix
            adj_coo = adj.tocoo()
            edge_index = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long().to(device)
        else:  # PyTorch sparse tensor — must be coalesced
            edge_index = adj.coalesce().indices().to(device)

        row, col = edge_index[0], edge_index[1]
        Wh_i = Wh[row]  # (E, H, F')
        Wh_j = Wh[col]  # (E, H, F')

        # Attention coefficients
        e = (torch.cat([Wh_i, Wh_j], dim=-1) * self.a).sum(dim=-1)  # (E, H)
        e = F.leaky_relu(e, 0.2)

        # Aggregate: mean over neighbors, then mean over heads
        alpha = torch.exp(e)  # (E, H)
        out = torch.zeros(N, self.n_heads, self.out_features, device=device)
        out = out.index_add(0, row, alpha.unsqueeze(-1) * Wh_j)
        norm = torch.zeros(N, self.n_heads, 1, device=device)
        norm = norm.index_add(0, row, alpha.unsqueeze(-1))
        out = out / (norm + 1e-12)
        out = F.elu(out.mean(dim=1))  # mean over heads → (N, F')
        return F.dropout(out, p=self.dropout, training=self.training)


class GAT(nn.Module):
    """2-layer GAT."""
    def __init__(self, in_features, hidden=32, n_heads=4, dropout=0.3):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden, n_heads=n_heads, dropout=dropout)
        self.gat2 = nn.Linear(hidden, 1)  # GAT1 mean-pools heads → output dim = hidden
        self.dropout = dropout

    def forward(self, x, adj):
        x = self.gat1(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x)
        return x.squeeze(-1)


def build_model(model_type: str, in_features: int, device: torch.device = None):
    """Factory for GNN models.

    Parameters
    ----------
    model_type : 'gcn' or 'gat'.
    in_features : input feature dimension.
    """
    if model_type == "gcn":
        model = GCN(in_features, hidden=64, dropout=0.5)
    elif model_type == "gat":
        model = GAT(in_features, hidden=32, n_heads=4, dropout=0.3)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    if device is not None:
        model = model.to(device)
    return model
