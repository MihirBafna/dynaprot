import torch.nn as nn
import torch
import torch.nn.functional as F


class LowRankDiagonalReadout(nn.Module):
    def __init__(self, d_model, rank=8, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.epsilon = 1e-6

        input_dim = 2 * d_model + 1  # h_i, h_j, attn_ij
        self.mlp = self._make_mlp(input_dim, hidden_dim, rank, num_layers, dropout)

        self.project_diag = nn.Linear(d_model, 1)

    def _make_mlp(self, in_dim, hidden_dim, out_dim, num_layers, dropout):
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, residue_features, attention):
        B, N, _ = residue_features.shape
        tril = torch.tril_indices(N, N, 0, device=residue_features.device)

        h_i = residue_features[:, tril[0], :]
        h_j = residue_features[:, tril[1], :]
        attn_ij = attention[:, tril[0], tril[1]].unsqueeze(-1)

        pair_input = torch.cat([h_i, h_j, attn_ij], dim=-1)
        pairwise_latents = self.mlp(pair_input)  # (B, num_pairs, rank)

        # Reconstruct full U matrix
        U = torch.zeros(B, N, self.rank, device=residue_features.device)
        U[:, tril[0]] += pairwise_latents
        U[:, tril[1]] += pairwise_latents  # symmetric update

        # Normalize
        U = U / 2

        # Diagonal variance
        d_raw = self.project_diag(residue_features).squeeze(-1)
        d = F.softplus(d_raw) + self.epsilon

        Sigma = U @ U.transpose(-1, -2) + torch.diag_embed(d)
        return Sigma