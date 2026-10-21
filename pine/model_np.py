"""Causal motion transformer, written out by hand in NumPy (forward and backward).

This is the reference implementation: it trains on CPU here, and the PyTorch/CUDA trainer
(gpu/train_torch.py) is checked against it, gradient for gradient, before any GPU time is spent.

Architecture (GPT-style, pre-LayerNorm): each token is one frame of a humanoid's state, plus a
mask saying which channels the wearer's suit actually measured. The model reads up to T frames and
predicts the next K frames of the full body at every position.
"""
from __future__ import annotations
import numpy as np

C_GELU = np.sqrt(2.0 / np.pi)


def count_params(din, dout, d, L, T):
    return din * d + d + T * d + L * (12 * d * d + 13 * d) + 2 * d + d * dout + dout


class MotionTransformer:
    def __init__(self, din, dout, d=64, L=2, H=4, T=32, seed=0, dtype=np.float32):
        assert d % H == 0
        self.din, self.dout, self.d, self.L, self.H, self.T, self.dtype = din, dout, d, L, H, T, dtype
        r = np.random.default_rng(seed)
        n = lambda *s, std=0.02: (r.standard_normal(s) * std).astype(dtype)
        z = lambda *s: np.zeros(s, dtype)
        o = lambda *s: np.ones(s, dtype)
        res = 0.02 / np.sqrt(2 * L)
        p = dict(win=n(din, d), bin=z(d), pos=n(T, d), lnfg=o(d), lnfb=z(d), wout=n(d, dout), bout=z(dout))
        for l in range(L):
            p.update({f"{l}.ln1g": o(d), f"{l}.ln1b": z(d), f"{l}.wqkv": n(d, 3 * d), f"{l}.bqkv": z(3 * d),
                      f"{l}.wo": n(d, d, std=res), f"{l}.bo": z(d), f"{l}.ln2g": o(d), f"{l}.ln2b": z(d),
                      f"{l}.w1": n(d, 4 * d), f"{l}.b1": z(4 * d), f"{l}.w2": n(4 * d, d, std=res), f"{l}.b2": z(d)})
        self.p = p
        self.n_params = sum(v.size for v in p.values())
        m = np.triu(np.ones((T, T), bool), 1)
        self.neg = np.where(m, -1e9, 0.0).astype(dtype)

    # ---------------------------------------------------------------- pieces
    @staticmethod
    def _ln(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdims=True)
        rstd = 1.0 / np.sqrt(var + eps)
        xh = xc * rstd
        return xh * g + b, (xh, rstd, g)

    @staticmethod
    def _ln_back(dy, cache):
        xh, rstd, g = cache
        dxh = dy * g
        D = xh.shape[-1]
        dx = rstd / D * (D * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
        return dx, (dy * xh).reshape(-1, D).sum(0), dy.reshape(-1, D).sum(0)

    # ---------------------------------------------------------------- forward
    def forward(self, x):
        p, B, T = self.p, x.shape[0], x.shape[1]
        d, H = self.d, self.H; dh = d // H
        cache = {"x": x}
        h = x @ p["win"] + p["bin"] + p["pos"][:T]
        for l in range(self.L):
            a, c1 = self._ln(h, p[f"{l}.ln1g"], p[f"{l}.ln1b"])
            qkv = a @ p[f"{l}.wqkv"] + p[f"{l}.bqkv"]
            q, k, v = (qkv[..., i * d:(i + 1) * d].reshape(B, T, H, dh).transpose(0, 2, 1, 3) for i in range(3))
            s = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / np.sqrt(dh)) + self.neg[:T, :T]
            s = s - s.max(-1, keepdims=True)
            P = np.exp(s); P /= P.sum(-1, keepdims=True)
            o = (P @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            h = h + o @ p[f"{l}.wo"] + p[f"{l}.bo"]
            m, c2 = self._ln(h, p[f"{l}.ln2g"], p[f"{l}.ln2b"])
            u = m @ p[f"{l}.w1"] + p[f"{l}.b1"]
            t = np.tanh(C_GELU * (u + 0.044715 * u ** 3))
            gu = 0.5 * u * (1 + t)
            h = h + gu @ p[f"{l}.w2"] + p[f"{l}.b2"]
            cache[l] = (a, c1, q, k, v, P, o, m, c2, u, t, gu)
        f, cf = self._ln(h, p["lnfg"], p["lnfb"])
        y = f @ p["wout"] + p["bout"]
        cache["f"], cache["cf"] = f, cf
        return y, cache

    # ---------------------------------------------------------------- backward
    def backward(self, dy, cache):
        p, g = self.p, {}
        B, T = dy.shape[0], dy.shape[1]
        d, H = self.d, self.H; dh = d // H
        f2 = lambda z: z.reshape(-1, z.shape[-1])
        g["wout"] = f2(cache["f"]).T @ f2(dy); g["bout"] = f2(dy).sum(0)
        dh_, g["lnfg"], g["lnfb"] = self._ln_back(dy @ p["wout"].T, cache["cf"])
        for l in reversed(range(self.L)):
            a, c1, q, k, v, P, o, m, c2, u, t, gu = cache[l]
            # MLP
            g[f"{l}.w2"] = f2(gu).T @ f2(dh_); g[f"{l}.b2"] = f2(dh_).sum(0)
            dgu = dh_ @ p[f"{l}.w2"].T
            du = dgu * (0.5 * (1 + t) + 0.5 * u * (1 - t * t) * C_GELU * (1 + 3 * 0.044715 * u * u))
            g[f"{l}.w1"] = f2(m).T @ f2(du); g[f"{l}.b1"] = f2(du).sum(0)
            dm = du @ p[f"{l}.w1"].T
            dx, g[f"{l}.ln2g"], g[f"{l}.ln2b"] = self._ln_back(dm, c2)
            dh_ = dh_ + dx
            # attention
            g[f"{l}.wo"] = f2(o).T @ f2(dh_); g[f"{l}.bo"] = f2(dh_).sum(0)
            do = (dh_ @ p[f"{l}.wo"].T).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
            dP = do @ v.transpose(0, 1, 3, 2)
            dv = P.transpose(0, 1, 3, 2) @ do
            ds = P * (dP - (dP * P).sum(-1, keepdims=True)) * (1.0 / np.sqrt(dh))
            dq = ds @ k; dk = ds.transpose(0, 1, 3, 2) @ q
            dqkv = np.concatenate([z.transpose(0, 2, 1, 3).reshape(B, T, d) for z in (dq, dk, dv)], -1)
            g[f"{l}.wqkv"] = f2(a).T @ f2(dqkv); g[f"{l}.bqkv"] = f2(dqkv).sum(0)
            da = dqkv @ p[f"{l}.wqkv"].T
            dx, g[f"{l}.ln1g"], g[f"{l}.ln1b"] = self._ln_back(da, c1)
            dh_ = dh_ + dx
        g["win"] = f2(cache["x"]).T @ f2(dh_); g["bin"] = f2(dh_).sum(0)
        gp = np.zeros_like(p["pos"]); gp[:T] = dh_.sum(0); g["pos"] = gp
        return g


class AdamW:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr, clip=1.0):
        self.t += 1
        gn = float(np.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values())))
        sc = min(1.0, clip / (gn + 1e-12))
        b1t, b2t = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, w in params.items():
            g = grads[k] * sc
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            if w.ndim == 2 and k != "pos":
                w -= lr * self.wd * w
            w -= lr * (self.m[k] / b1t) / (np.sqrt(self.v[k] / b2t) + self.eps)
        return gn
