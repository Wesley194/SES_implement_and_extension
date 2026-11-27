"""Surrogate-Assisted Spectral Evolution Search with a Fixed-Level 2D Grid GNN.

Pure PyTorch implementation (no torch_geometric dependency).
Uses a 2-layer residual GCN trained with Pairwise Ranking Loss (RankNet)
to pre-screen CEM candidates in a fixed J-level low-frequency wavelet subspace.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spaces import LowFreqSpace
from .base import SearchResult


def build_grid_adj(h: int, w: int, device: torch.device) -> torch.Tensor:
    """Build symmetric normalized adjacency D^{-1/2} A D^{-1/2} for a 2D 4-neighbor grid."""
    n = h * w
    adj = torch.eye(n, dtype=torch.float32, device=device)  # self-loops
    for r in range(h):
        for c in range(w):
            u = r * w + c
            if r + 1 < h:
                v = (r + 1) * w + c
                adj[u, v] = 1.0
                adj[v, u] = 1.0
            if c + 1 < w:
                v = r * w + (c + 1)
                adj[u, v] = 1.0
                adj[v, u] = 1.0

    deg = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def make_pos_feats(h: int, w: int) -> np.ndarray:
    """Precompute normalized 2D spatial coordinates (h*w, 2) in [-1, 1]."""
    ys = np.linspace(-1.0, 1.0, h) if h > 1 else np.zeros(1)
    xs = np.linspace(-1.0, 1.0, w) if w > 1 else np.zeros(1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1)], axis=1).astype(np.float32)


def vecs_to_grid_features(
    vecs: np.ndarray, c_dim: int, pos_feats: np.ndarray
) -> np.ndarray:
    """Convert batch of low-freq vectors (B, C*H*W) to node features (B, H*W, C+2)."""
    b = vecs.shape[0]
    n_nodes = pos_feats.shape[0]
    ch_feats = vecs.reshape(b, c_dim, n_nodes).transpose(0, 2, 1).astype(np.float32)
    pos_batch = np.broadcast_to(pos_feats[None, :, :], (b, n_nodes, 2))
    return np.concatenate([ch_feats, pos_batch], axis=-1)


class GridGNNSurrogate(nn.Module):
    """Lightweight 2-layer residual GCN with mean+max readout for ranking noise grids."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.15):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """x: (B, N, in_dim), adj_norm: (N, N) -> scores: (B,)"""
        h = F.silu(self.in_proj(x))
        h = h + self.drop(F.silu(self.norm1(self.conv1(torch.matmul(adj_norm, h)))))
        h = h + self.drop(F.silu(self.norm2(self.conv2(torch.matmul(adj_norm, h)))))
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values], dim=-1)
        return self.head(pooled).squeeze(-1)


def train_surrogate_pairwise(
    model: GridGNNSurrogate,
    archive: List[Tuple[np.ndarray, float]],
    adj_norm: torch.Tensor,
    device: torch.device,
    epochs: int = 40,
    lr: float = 3e-3,
):
    """Train GNN on all archived evaluations using Pairwise Ranking Loss (RankNet)."""
    if len(archive) < 6:
        return

    x_all = torch.tensor(
        np.stack([f for f, _ in archive]), dtype=torch.float32, device=device
    )
    y = torch.tensor(
        [s for _, s in archive], dtype=torch.float32, device=device
    )

    diff_y = y[:, None] - y[None, :]
    pair_mask = diff_y > 1e-4
    if not pair_mask.any():
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(x_all, adj_norm)
        diff_p = preds[:, None] - preds[None, :]
        loss = F.softplus(-diff_p[pair_mask]).mean()
        loss.backward()
        optimizer.step()
    model.eval()


def ses_gnn(
    objective,
    shape,
    budget: int = 200,
    level: int = 3,
    pop: int = 10,
    elite: int = 5,
    gamma: float = 1e-5,
    pool_mult: int = 10,
    explore_ratio: float = 0.2,
    warmup_evals: int = 20,
    gnn_hidden: int = 32,
    gnn_epochs: int = 40,
    wavelet: str = "haar",
    seed: int = 0,
    verbose: bool = False,
    **_kwargs,
) -> SearchResult:
    """Fixed-level GNN-assisted Spectral Evolution Search."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    space = LowFreqSpace(shape, level, wavelet, rng)
    c_dim, h_dim, w_dim = shape
    ll_h = max(1, (h_dim + (1 << level) - 1) >> level)
    ll_w = max(1, (w_dim + (1 << level) - 1) >> level)
    assert c_dim * ll_h * ll_w == space.dim, (
        f"Grid shape ({c_dim}, {ll_h}, {ll_w}) does not match space.dim={space.dim}"
    )

    adj_norm = build_grid_adj(ll_h, ll_w, device)
    pos_feats = make_pos_feats(ll_h, ll_w)
    model = GridGNNSurrogate(in_dim=c_dim + 2, hidden_dim=gnn_hidden).to(device)

    mu = np.zeros(space.dim)
    var = np.ones(space.dim)
    pool: List[Tuple[np.ndarray, float]] = []
    archive: List[Tuple[np.ndarray, float]] = []

    best_u, best_score = None, -np.inf
    n_eval, history = 0, []

    while n_eval < budget:
        n = min(pop, budget - n_eval)

        # 1. Sample candidate pool & pre-screen with GNN after warm-up
        if n_eval >= warmup_evals and len(archive) >= 6:
            train_surrogate_pairwise(model, archive, adj_norm, device, epochs=gnn_epochs)
            n_cand = max(n * pool_mult, n + 1)
            cand_us = mu + np.sqrt(var) * rng.standard_normal((n_cand, space.dim))
            cand_feats = vecs_to_grid_features(cand_us, c_dim, pos_feats)

            x_cand = torch.tensor(cand_feats, dtype=torch.float32, device=device)
            with torch.no_grad():
                pred_scores = model(x_cand, adj_norm).cpu().numpy()

            n_explore = min(int(round(n * explore_ratio)), max(n - 1, 0))
            n_exploit = n - n_explore
            order = np.argsort(pred_scores)[::-1]
            chosen_idx = list(order[:n_exploit])
            if n_explore > 0:
                rest = order[n_exploit:]
                chosen_idx.extend(
                    rng.choice(rest, size=min(n_explore, len(rest)), replace=False)
                )
            selected_us = cand_us[chosen_idx]
            selected_feats = cand_feats[chosen_idx]
        else:
            selected_us = mu + np.sqrt(var) * rng.standard_normal((n, space.dim))
            selected_feats = vecs_to_grid_features(selected_us, c_dim, pos_feats)

        # 2. Expensive evaluation on selected candidates
        for u, feats in zip(selected_us, selected_feats):
            s = float(objective(space.to_noise(u)))
            archive.append((feats, s))
            pool.append((u, s))
            if s > best_score:
                best_score = s
                best_u = u.copy()

        n_eval += n

        # 3. CEM elite update
        pool.sort(key=lambda t: t[1], reverse=True)
        pool = pool[:elite]
        U = np.stack([u for u, _ in pool])
        mu = (1 - gamma) * U.mean(0) + gamma * mu
        var = (1 - gamma) * np.maximum(U.var(0), 1e-6) + gamma * var

        history.append((n_eval, best_score))
        if verbose:
            print(
                f"[SES-GNN] NRE={n_eval:4d}  level={level}  "
                f"dim={space.dim}  best={best_score:.4f}",
                flush=True,
            )

    return SearchResult(
        best_x0=space.to_noise(best_u),
        best_score=best_score,
        history=history,
    )