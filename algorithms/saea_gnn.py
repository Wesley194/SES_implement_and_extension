"""Surrogate-Assisted Spectral Evolution Search with a Multi-Scale Quad-Tree GNN.

Pure PyTorch implementation (no torch_geometric dependency).
Supports both:
  - bandpass=False : cumulative coarse-to-fine unfreezing (LL_5 -> LL_4 -> LL_3)
  - bandpass=True  : strict band-pass (optimize LL_5 -> freeze, optimize D_5 -> freeze, optimize D_4)
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from spaces import WaveletSpace
from .base import SearchResult


def build_normalized_adj(grid_shapes: List[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    """Build symmetric normalized adjacency D^{-1/2} A D^{-1/2} for a multi-scale quad-tree."""
    offsets = []
    total = 0
    for h, w in grid_shapes:
        offsets.append(total)
        total += h * w

    adj = torch.eye(total, dtype=torch.float32, device=device)  # self-loops

    for s, (h, w) in enumerate(grid_shapes):
        off = offsets[s]
        # 1. Intra-level 4-neighbor spatial edges
        for r in range(h):
            for c in range(w):
                u = off + r * w + c
                if r + 1 < h:
                    v = off + (r + 1) * w + c
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0
                if c + 1 < w:
                    v = off + r * w + (c + 1)
                    adj[u, v] = 1.0
                    adj[v, u] = 1.0

        # 2. Cross-scale quad-tree parent-child edges (parent at s-1 -> child at s)
        if s > 0:
            hp, wp = grid_shapes[s - 1]
            off_p = offsets[s - 1]
            for r in range(h):
                for c in range(w):
                    child = off + r * w + c
                    parent = off_p + (r // 2) * wp + (c // 2)
                    adj[child, parent] = 1.0
                    adj[parent, child] = 1.0

    deg = adj.sum(dim=1).clamp(min=1.0)
    deg_inv_sqrt = deg.pow(-0.5)
    return deg_inv_sqrt[:, None] * adj * deg_inv_sqrt[None, :]


def vec_to_pyramid_features(
    space: WaveletSpace, full_vec: np.ndarray, k: int, max_k: int
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Convert an active wavelet vector into multi-scale node features (N_nodes, C + 3)."""
    i = space.ll.size
    cur = full_vec[:i].reshape(space.ll.shape)
    grids = [cur]

    for j in range(k):
        tri = []
        for a in space.details[j]:
            tri.append(full_vec[i : i + a.size].reshape(a.shape))
            i += a.size
        cur = pywt.idwt2((cur, tuple(tri)), space.wavelet, mode="periodization", axes=(-2, -1))
        grids.append(cur)

    node_blocks = []
    grid_shapes = []
    denom_s = max(max_k, 1)

    for s, g in enumerate(grids):
        c, h, w = g.shape
        grid_shapes.append((h, w))
        # Channel features: (h*w, c)
        ch_feats = g.reshape(c, h * w).T
        # Spatial + scale positional features: (h*w, 3)
        ys = np.linspace(-1.0, 1.0, h) if h > 1 else np.zeros(1)
        xs = np.linspace(-1.0, 1.0, w) if w > 1 else np.zeros(1)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        scale_col = np.full((h * w, 1), s / denom_s, dtype=np.float32)
        pos_feats = np.concatenate([yy.reshape(-1, 1), xx.reshape(-1, 1), scale_col], axis=1)
        node_blocks.append(np.concatenate([ch_feats, pos_feats], axis=1).astype(np.float32))

    return np.concatenate(node_blocks, axis=0), grid_shapes


class MultiScaleGNNSurrogate(nn.Module):
    """Lightweight 2-layer residual GCN with mean+max readout for ranking noise graphs."""

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
    model: MultiScaleGNNSurrogate,
    archive: List[Tuple[int, np.ndarray, float]],
    adj_cache: Dict[int, torch.Tensor],
    device: torch.device,
    epochs: int = 40,
    lr: float = 3e-3,
):
    """Train GNN across all archived stages jointly using Pairwise Ranking Loss (RankNet)."""
    if len(archive) < 6:
        return

    # Group archive by stage k so each stage runs as one fast batched forward pass
    by_k: Dict[int, Tuple[List[int], List[np.ndarray]]] = {}
    targets = []
    for idx, (k, feats, score) in enumerate(archive):
        by_k.setdefault(k, ([], []))
        by_k[k][0].append(idx)
        by_k[k][1].append(feats)
        targets.append(score)

    y = torch.tensor(targets, dtype=torch.float32, device=device)
    batches = {
        k: (
            torch.tensor(idxs, dtype=torch.long, device=device),
            torch.tensor(np.stack(f_list), dtype=torch.float32, device=device),
        )
        for k, (idxs, f_list) in by_k.items()
    }

    diff_y = y[:, None] - y[None, :]
    pair_mask = diff_y > 1e-4
    if not pair_mask.any():
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        preds = torch.empty_like(y)
        for k, (idxs, x_batch) in batches.items():
            preds[idxs] = model(x_batch, adj_cache[k])
        diff_p = preds[:, None] - preds[None, :]
        loss = F.softplus(-diff_p[pair_mask]).mean()
        loss.backward()
        optimizer.step()
    model.eval()


def ses_gnn(
    objective,
    shape,
    budget: int = 200,
    levels: Optional[Tuple[int, ...]] = None,
    level: int = 3,
    bandpass: bool = False,
    pop: int = 10,
    elite: int = 5,
    gamma: float = 1e-5,
    sigma_new: float = 1.0,
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
    """Multi-scale GNN-assisted Spectral Evolution Search (cumulative or strict band-pass)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-derive safe coarse-to-fine schedule from target `level` if `levels` not specified
    c_dim, h_dim, w_dim = shape
    max_possible = int(np.log2(min(h_dim, w_dim)))
    if levels is None:
        start_lvl = min(level + 2, max_possible)
        levels = tuple(range(start_lvl, level - 1, -1))
    if not levels:
        levels = (level,)

    space = WaveletSpace(shape, max_level=levels[0], wavelet=wavelet, rng=rng)
    max_k = space.level_to_k(levels[-1])
    n_stage = len(levels)

    model = MultiScaleGNNSurrogate(in_dim=c_dim + 3, hidden_dim=gnn_hidden).to(device)
    adj_cache: Dict[int, torch.Tensor] = {}
    archive: List[Tuple[int, np.ndarray, float]] = []

    mu = var = pool = None
    frozen_prefix = np.array([], dtype=np.float64)
    best_full_vec, best_k, best_score = None, 0, -np.inf
    prev_k, prev_dim = None, 0
    n_eval, history = 0, []
    tag = "SES-GNN-BP" if bandpass else "SES-GNN"

    while n_eval < budget:
        stage = min(int(n_eval / budget * n_stage), n_stage - 1)
        k = space.level_to_k(levels[stage])

        # Stage transition: expand dimension (cumulative) or freeze & move (band-pass)
        if k != prev_k:
            cur_dim = space.dim(k)
            ref = space.ref_vec(k)
            if mu is None:
                mu = np.zeros(cur_dim)
                var = np.ones(cur_dim)
                pool = []
                frozen_prefix = np.array([], dtype=np.float64)
            elif bandpass:
                # Freeze previous best vector; only optimize newly unlocked band
                if best_full_vec is None:
                    frozen_prefix = ref[:prev_dim].copy()
                elif best_full_vec.size < prev_dim:
                    frozen_prefix = np.concatenate(
                        [best_full_vec, ref[best_full_vec.size : prev_dim]]
                    )
                else:
                    frozen_prefix = best_full_vec[:prev_dim].copy()

                new_band_ref = ref[prev_dim:cur_dim]
                mu = new_band_ref.copy()
                var = np.full(cur_dim - prev_dim, sigma_new**2)
                pool = [(new_band_ref.copy(), best_score)]
            else:
                # Cumulative: warm-start existing coords, append new coords
                mu = np.concatenate([mu, ref[prev_dim:cur_dim]])
                var = np.concatenate([var, np.full(cur_dim - prev_dim, sigma_new**2)])
                pool = [(np.concatenate([u, ref[prev_dim:cur_dim]]), s) for u, s in pool]

            prev_k, prev_dim = k, cur_dim

        stage_end = int(budget * (stage + 1) / n_stage) if stage < n_stage - 1 else budget
        n = min(pop, stage_end - n_eval) or min(pop, budget - n_eval)

        # 1. Sample candidate pool & pre-screen with GNN after warm-up
        if n_eval >= warmup_evals and len(archive) >= 6:
            train_surrogate_pairwise(model, archive, adj_cache, device, epochs=gnn_epochs)
            n_cand = max(n * pool_mult, n + 1)
            cand_active = mu + np.sqrt(var) * rng.standard_normal((n_cand, mu.size))

            feats_list = []
            for u_act in cand_active:
                full_v = np.concatenate([frozen_prefix, u_act]) if bandpass else u_act
                feats, g_shapes = vec_to_pyramid_features(space, full_v, k, max_k)
                if k not in adj_cache:
                    adj_cache[k] = build_normalized_adj(g_shapes, device)
                feats_list.append(feats)

            x_cand = torch.tensor(np.stack(feats_list), dtype=torch.float32, device=device)
            with torch.no_grad():
                pred_scores = model(x_cand, adj_cache[k]).cpu().numpy()

            n_explore = min(int(round(n * explore_ratio)), max(n - 1, 0))
            n_exploit = n - n_explore
            order = np.argsort(pred_scores)[::-1]
            chosen_idx = list(order[:n_exploit])
            if n_explore > 0:
                rest = order[n_exploit:]
                chosen_idx.extend(rng.choice(rest, size=min(n_explore, len(rest)), replace=False))
            selected_active = cand_active[chosen_idx]
            selected_feats = [feats_list[i] for i in chosen_idx]
        else:
            selected_active = mu + np.sqrt(var) * rng.standard_normal((n, mu.size))
            selected_feats = None

        # 2. Expensive evaluation on selected candidates
        for idx_u, u_act in enumerate(selected_active):
            full_v = np.concatenate([frozen_prefix, u_act]) if bandpass else u_act
            s = float(objective(space.to_noise(full_v, k)))

            if selected_feats is not None:
                feats = selected_feats[idx_u]
            else:
                feats, g_shapes = vec_to_pyramid_features(space, full_v, k, max_k)
                if k not in adj_cache:
                    adj_cache[k] = build_normalized_adj(g_shapes, device)

            archive.append((k, feats, s))
            pool.append((u_act, s))
            if s > best_score:
                best_score = s
                best_full_vec = full_v.copy()
                best_k = k

        n_eval += n

        # 3. CEM elite update on active coordinates
        pool.sort(key=lambda t: t[1], reverse=True)
        pool = pool[:elite]
        U = np.stack([u for u, _ in pool])
        mu = (1 - gamma) * U.mean(0) + gamma * mu
        var = (1 - gamma) * np.maximum(U.var(0), 1e-6) + gamma * var

        history.append((n_eval, best_score))
        if verbose:
            print(
                f"[{tag}] NRE={n_eval:4d} level={levels[stage]} "
                f"active_dim={mu.size:4d} best={best_score:.4f}",
                flush=True,
            )

    return SearchResult(
        best_x0=space.to_noise(best_full_vec, best_k),
        best_score=best_score,
        history=history,
    )