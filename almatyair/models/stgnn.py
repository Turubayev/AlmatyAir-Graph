"""Пространственно-временная графовая сеть (ST-GNN) для прогноза PM2.5 на 1–48 ч.

Архитектура (в духе Graph WaveNet / STGCN, без зависимости от PyTorch Geometric —
граф маленький, поэтому достаточно плотной матрицы смежности):

1. Временной энкодер: GRU по 48 ч истории каждой станции (веса общие для узлов).
2. Пространственный блок: K слоёв графовой свёртки по смеси
   * физической матрицы (расстояние + перепад высот, см. graph.py),
   * адаптивной матрицы softmax(relu(E1·E2ᵀ)) — модель сама учит связи,
     которых нет в физике (например, общий источник выбросов).
3. Декодер: GRU по метеопрогнозу на 48 ч вперёд (ветер, высота слоя перемешивания…),
   инициализированный состоянием узла. Именно он «видит» приход фронта и конец эпизода.
4. Выход: три квантиля (0.1, 0.5, 0.9) log1p(PM2.5) на каждый час и станцию.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as e:  # pragma: no cover
    raise ImportError("Для ST-GNN нужен PyTorch: pip install torch") from e

from ..graph import normalize
from .baselines import QUANTILES

PM_SCALE = 3.0  # см. features.make_sequences: pm_scaled = log1p(pm)/3 - 1


class GraphConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.nbr_lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, A):  # h: (B, N, D), A: (N, N)
        msg = torch.einsum("ij,bjd->bid", A, h)
        return self.norm(h + torch.relu(self.self_lin(h) + self.nbr_lin(msg)))


class STGNN(nn.Module):
    def __init__(self, n_nodes: int, adj: np.ndarray, f_hist: int, f_fut: int, hidden: int = 64,
                 emb_dim: int = 8, n_layers: int = 2, n_quantiles: int = len(QUANTILES), dropout: float = 0.1):
        super().__init__()
        self.register_buffer("A_phys", torch.tensor(normalize(adj), dtype=torch.float32))
        self.E1 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.mix = nn.Parameter(torch.tensor(0.0))
        self.node_emb = nn.Embedding(n_nodes, emb_dim)
        self.enc = nn.GRU(f_hist + emb_dim, hidden, batch_first=True)
        self.gconv = nn.ModuleList([GraphConv(hidden) for _ in range(n_layers)])
        self.dec = nn.GRU(f_fut + emb_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_quantiles))
        self.n_nodes = n_nodes

    def adjacency(self):
        adaptive = torch.softmax(torch.relu(self.E1 @ self.E2.T), dim=1)
        g = torch.sigmoid(self.mix)
        return (1 - g) * self.A_phys + g * adaptive

    def forward(self, xh, xf):
        # xh: (B, Th, N, Fh), xf: (B, H, N, Ff) → (B, H, N, Q)
        B, Th, N, _ = xh.shape
        H = xf.shape[1]
        emb = self.node_emb.weight  # (N, E)
        eh = emb[None, None].expand(B, Th, N, -1)
        ef = emb[None, None].expand(B, H, N, -1)
        x = torch.cat([xh, eh], -1).permute(0, 2, 1, 3).reshape(B * N, Th, -1)
        _, h = self.enc(x)  # (1, B*N, D)
        h = h[0].reshape(B, N, -1)
        A = self.adjacency()
        for layer in self.gconv:
            h = layer(h, A)
        h0 = self.drop(h).reshape(1, B * N, -1).contiguous()
        y = torch.cat([xf, ef], -1).permute(0, 2, 1, 3).reshape(B * N, H, -1)
        out, _ = self.dec(y, h0)
        q = self.head(out).reshape(B, N, H, -1).permute(0, 2, 1, 3)
        # Монотонность квантилей: медиана + положительные отступы.
        mid = q[..., 1:2]
        lo = mid - torch.nn.functional.softplus(q[..., 0:1])
        hi = mid + torch.nn.functional.softplus(q[..., 2:3])
        return torch.cat([lo, mid, hi], -1)


def pinball_loss(pred, target, quantiles=QUANTILES):
    """pred: (B, H, N, Q) в шкале log1p; target: (B, H, N), NaN маскируются."""
    mask = torch.isfinite(target)
    t = torch.log1p(torch.nan_to_num(target, nan=0.0).clamp(min=0))
    losses = []
    for i, q in enumerate(quantiles):
        e = t - pred[..., i]
        losses.append(torch.maximum(q * e, (q - 1) * e))
    loss = torch.stack(losses, -1).mean(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


class STGNNForecaster:
    name = "stgnn"

    def __init__(self, n_nodes: int, adj: np.ndarray, f_hist: int, f_fut: int, hidden: int = 64, seed: int = 0):
        torch.manual_seed(seed)
        self.cfg = dict(n_nodes=n_nodes, f_hist=f_hist, f_fut=f_fut, hidden=hidden)
        self.adj = adj
        self.model = STGNN(n_nodes, adj, f_hist, f_fut, hidden)

    def fit(self, Xh, Xf, Y, epochs: int = 15, batch_size: int = 64, lr: float = 2e-3,
            val: tuple | None = None, verbose: bool = True):
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        Xh, Xf, Y = map(torch.from_numpy, (Xh, Xf, Y))
        n = len(Xh)
        best, best_state = np.inf, None
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, batch_size):
                b = perm[i : i + batch_size]
                opt.zero_grad()
                loss = pinball_loss(self.model(Xh[b], Xf[b]), Y[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tot += loss.item() * len(b)
            sched.step()
            msg = f"эпоха {ep + 1}/{epochs}: train pinball {tot / n:.4f}"
            if val is not None:
                vl = self.evaluate_loss(*val)
                msg += f", val {vl:.4f}"
                if vl < best:
                    best, best_state = vl, {k: v.clone() for k, v in self.model.state_dict().items()}
            if verbose:
                print(msg, flush=True)
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def evaluate_loss(self, Xh, Xf, Y, batch_size: int = 256) -> float:
        self.model.eval()
        tot, n = 0.0, 0
        for i in range(0, len(Xh), batch_size):
            p = self.model(torch.from_numpy(Xh[i : i + batch_size]), torch.from_numpy(Xf[i : i + batch_size]))
            tot += pinball_loss(p, torch.from_numpy(Y[i : i + batch_size])).item() * len(p)
            n += len(p)
        return tot / max(n, 1)

    @torch.no_grad()
    def predict_quantiles(self, Xh, Xf, batch_size: int = 256) -> np.ndarray:
        """Возвращает (B, H, N, 3) в мкг/м³."""
        self.model.eval()
        outs = []
        for i in range(0, len(Xh), batch_size):
            p = self.model(torch.from_numpy(Xh[i : i + batch_size]), torch.from_numpy(Xf[i : i + batch_size]))
            outs.append(torch.expm1(p).clamp(min=0).numpy())
        return np.concatenate(outs)

    def save(self, path: str | Path):
        torch.save({"cfg": self.cfg, "adj": self.adj, "state": self.model.state_dict()}, path)

    @classmethod
    def load(cls, path: str | Path) -> "STGNNForecaster":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(adj=ck["adj"], **ck["cfg"])
        obj.model.load_state_dict(ck["state"])
        return obj

    def export_onnx(self, path: str | Path, hist: int = 48, horizon: int = 48):
        self.model.eval()
        xh = torch.zeros(1, hist, self.cfg["n_nodes"], self.cfg["f_hist"])
        xf = torch.zeros(1, horizon, self.cfg["n_nodes"], self.cfg["f_fut"])
        torch.onnx.export(self.model, (xh, xf), str(path), input_names=["x_hist", "x_future"],
                          output_names=["log1p_pm25_quantiles"], dynamic_axes={"x_hist": {0: "batch"}, "x_future": {0: "batch"}},
                          opset_version=17, dynamo=False)
