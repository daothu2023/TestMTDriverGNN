"""
Model definitions for MTDriverGNN: ResidualGCNEncoder, MultiTaskGCN, and
LearnableAlpha. See each class's docstring for implementation details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ResidualGCNEncoder(nn.Module):
    """
    GCN encoder with a residual pathway from the raw input features.

    `hidden_dims` is a list of output dimensions for successive GCNConv
    layers (e.g. [64, 128] for a 2-layer encoder). After the GCN stack,
    a residual term is added: H = H^(L) + W_R X, where W_R is the
    residual projection's weight matrix, matching the readout
    formulation in the manuscript. GCN weights are Xavier-initialized;
    the residual projection, when present, is Xavier-initialized as well.
    """

    def __init__(self, in_dim, hidden_dims, dropout=0.5):
        super().__init__()
        assert len(hidden_dims) >= 1
        self.convs = nn.ModuleList()
        last_dim = in_dim
        for h in hidden_dims:
            self.convs.append(GCNConv(last_dim, h))
            last_dim = h
        self.residual_proj = (
            nn.Linear(in_dim, hidden_dims[-1], bias=False)
            if in_dim != hidden_dims[-1]
            else nn.Identity()
        )
        self.dropout = dropout

        for conv in self.convs:
            nn.init.xavier_uniform_(conv.lin.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        residual = self.residual_proj(x) if isinstance(self.residual_proj, nn.Linear) else x
        return h + residual


class MultiTaskGCN(nn.Module):
    """
    Wraps a ResidualGCNEncoder with a shared MLP layer and two linear
    output heads. `forward` returns a tuple (driver_logit,
    telomere_logit), both of shape [num_nodes]; these are raw logits
    intended for `BCEWithLogitsLoss`. The shared layer and both heads
    use Kaiming initialization.
    """

    def __init__(self, in_dim, hidden_dims, dropout=0.5, head_hidden=None):
        super().__init__()
        self.encoder = ResidualGCNEncoder(in_dim, hidden_dims, dropout=dropout)
        encoder_out_dim = hidden_dims[-1]
        head_hidden = head_hidden or encoder_out_dim

        self.shared = nn.Sequential(
            nn.Linear(encoder_out_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.driver_head = nn.Linear(head_hidden, 1)
        self.telomere_head = nn.Linear(head_hidden, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.driver_head.weight, nonlinearity="sigmoid")
        nn.init.kaiming_uniform_(self.telomere_head.weight, nonlinearity="sigmoid")

    def forward(self, x, edge_index):
        h = self.encoder(x, edge_index)
        h = self.shared(h)
        driver_logit = self.driver_head(h).squeeze(-1)
        telomere_logit = self.telomere_head(h).squeeze(-1)
        return driver_logit, telomere_logit


class LearnableAlpha(nn.Module):
    """
    Sigmoid(logit_alpha), a single trainable parameter shared across the
    whole model and updated jointly with the rest of the network. Used
    as L_total = alpha * L_driver + (1 - alpha) * L_telomere.
    """

    def __init__(self, init_alpha=0.7):
        super().__init__()
        init_logit = torch.logit(torch.tensor(float(init_alpha)))
        self.logit_alpha = nn.Parameter(init_logit)

    def forward(self):
        return torch.sigmoid(self.logit_alpha)
