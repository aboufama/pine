"""PyTorch twin of pine.model_np.MotionTransformer (same parameter names, shapes, init and maths).

Parameter names are exactly the NumPy dict keys: win, bin, pos, lnfg, lnfb, wout, bout and, per layer l,
{l}.ln1g {l}.ln1b {l}.wqkv {l}.bqkv {l}.wo {l}.bo {l}.ln2g {l}.ln2b {l}.w1 {l}.b1 {l}.w2 {l}.b2.
Weights are stored (in, out) like the NumPy model, so x @ W + b is the same expression in both.
Pre-LayerNorm (eps 1e-5, biased variance), tanh-GELU, causal softmax attention scaled by 1/sqrt(d_head).

gpu/parity.py checks this module against the NumPy reference (loss, every gradient, AdamW steps) before any
training is allowed to start.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

LN_EPS = 1e-5


class Block(nn.Module):
    def __init__(self, d: int, H: int):
        super().__init__()
        assert d % H == 0, f"d_model {d} is not divisible by heads {H}"
        self.d, self.H = d, H
        P = lambda *s: nn.Parameter(torch.zeros(*s))
        self.ln1g, self.ln1b = nn.Parameter(torch.ones(d)), P(d)
        self.wqkv, self.bqkv = P(d, 3 * d), P(3 * d)
        self.wo, self.bo = P(d, d), P(d)
        self.ln2g, self.ln2b = nn.Parameter(torch.ones(d)), P(d)
        self.w1, self.b1 = P(d, 4 * d), P(4 * d)
        self.w2, self.b2 = P(4 * d, d), P(d)

    def forward(self, h: torch.Tensor, attn: str = "sdpa") -> torch.Tensor:
        B, T, d = h.shape
        H = self.H
        dh = d // H
        a = Fn.layer_norm(h, (d,), self.ln1g, self.ln1b, LN_EPS)
        qkv = Fn.linear(a, self.wqkv.t(), self.bqkv)                       # a @ wqkv + bqkv
        q, k, v = (z.reshape(B, T, H, dh).transpose(1, 2) for z in qkv.split(d, dim=-1))
        scale = 1.0 / math.sqrt(dh)
        if attn == "sdpa":
            o = Fn.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
        elif attn == "math":                                                # the NumPy expression, literally
            s = (q @ k.transpose(-2, -1)) * scale
            causal = torch.ones(T, T, dtype=torch.bool, device=h.device).triu(1)
            s = s.masked_fill(causal, float("-inf"))
            o = torch.softmax(s, dim=-1) @ v
        else:
            raise ValueError(f"attn must be 'sdpa' or 'math', got {attn!r}")
        o = o.transpose(1, 2).reshape(B, T, d)
        h = h + Fn.linear(o, self.wo.t(), self.bo)
        m = Fn.layer_norm(h, (d,), self.ln2g, self.ln2b, LN_EPS)
        u = Fn.linear(m, self.w1.t(), self.b1)
        gu = Fn.gelu(u, approximate="tanh")                               # 0.5 u (1 + tanh(sqrt(2/pi)(u + 0.044715 u^3)))
        return h + Fn.linear(gu, self.w2.t(), self.b2)


class MotionTransformerTorch(nn.Module):
    """Same constructor signature as pine.model_np.MotionTransformer (din, dout, d, L, H, T)."""

    def __init__(self, din: int, dout: int, d: int = 64, L: int = 2, H: int = 4, T: int = 32, attn: str = "sdpa"):
        super().__init__()
        assert d % H == 0, f"d_model {d} is not divisible by heads {H}"
        self.din, self.dout, self.d, self.L, self.H, self.T, self.attn = din, dout, d, L, H, T, attn
        self.win = nn.Parameter(torch.zeros(din, d))
        self.bin = nn.Parameter(torch.zeros(d))
        self.pos = nn.Parameter(torch.zeros(T, d))
        self.lnfg = nn.Parameter(torch.ones(d))
        self.lnfb = nn.Parameter(torch.zeros(d))
        self.wout = nn.Parameter(torch.zeros(d, dout))
        self.bout = nn.Parameter(torch.zeros(dout))
        for l in range(L):                      # registered as "0", "1", ... so names read "0.ln1g" like NumPy
            self.add_module(str(l), Block(d, H))

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def blocks(self):
        return [getattr(self, str(l)) for l in range(self.L)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.din:
            raise ValueError(f"expected input (B, T, {self.din}), got {tuple(x.shape)}")
        T = x.shape[1]
        if T > self.T:
            raise ValueError(f"sequence length {T} exceeds the context {self.T}")
        h = Fn.linear(x, self.win.t(), self.bin) + self.pos[:T]
        for blk in self.blocks():
            h = blk(h, self.attn)
        f = Fn.layer_norm(h, (self.d,), self.lnfg, self.lnfb, LN_EPS)
        return Fn.linear(f, self.wout.t(), self.bout)

    # ------------------------------------------------------------------ NumPy interop
    def expected_shapes(self) -> dict:
        return {k: tuple(v.shape) for k, v in self.named_parameters()}

    @torch.no_grad()
    def load_from_numpy(self, params: dict) -> "MotionTransformerTorch":
        """Copy a pine.model_np parameter dict in. Names and shapes must match exactly."""
        mine = dict(self.named_parameters())
        missing, extra = sorted(set(mine) - set(params)), sorted(set(params) - set(mine))
        if missing or extra:
            raise KeyError(f"parameter names differ: missing {missing[:6]} extra {extra[:6]}")
        for k, p in mine.items():
            v = np.asarray(params[k])
            if tuple(v.shape) != tuple(p.shape):
                raise ValueError(f"{k}: NumPy shape {tuple(v.shape)} != torch shape {tuple(p.shape)}")
            p.copy_(torch.from_numpy(np.ascontiguousarray(v)).to(device=p.device, dtype=p.dtype))
        return self

    @torch.no_grad()
    def to_numpy(self, dtype=np.float32) -> dict:
        return {k: p.detach().to("cpu", torch.float64).numpy().astype(dtype) for k, p in self.named_parameters()}

    @classmethod
    def like_numpy(cls, din, dout, d, L, H, T, seed=0, attn="sdpa") -> "MotionTransformerTorch":
        """Build with exactly the NumPy model's initialisation (it runs pine.model_np's own init code)."""
        from pine.model_np import MotionTransformer
        ref = MotionTransformer(din, dout, d, L, H, T, seed=seed, dtype=np.float32)
        m = cls(din, dout, d, L, H, T, attn=attn)
        m.load_from_numpy(ref.p)
        assert m.n_params == ref.n_params, (m.n_params, ref.n_params)
        return m


def unwrap(model) -> MotionTransformerTorch:
    """Strip DistributedDataParallel / torch.compile wrappers."""
    seen = 0
    while not isinstance(model, MotionTransformerTorch):
        model = getattr(model, "module", None) or getattr(model, "_orig_mod", None)
        seen += 1
        if model is None or seen > 4:
            raise TypeError("not a MotionTransformerTorch (or a DDP / compiled wrapper of one)")
    return model


# ---------------------------------------------------------------------- optimizer, identical to pine.model_np.AdamW
def decays(name: str, p: torch.Tensor) -> bool:
    """Decoupled weight decay only on 2-D parameters except the positional table (pine.model_np.AdamW rule)."""
    return p.dim() == 2 and name != "pos"


def make_optimizer(model, lr: float, wd: float = 0.1, b1: float = 0.9, b2: float = 0.95, eps: float = 1e-8,
                   fused: bool | None = None) -> torch.optim.AdamW:
    named = list(unwrap(model).named_parameters())
    groups = [dict(params=[p for n, p in named if decays(n, p)], weight_decay=wd),
              dict(params=[p for n, p in named if not decays(n, p)], weight_decay=0.0)]
    kw = dict(lr=lr, betas=(b1, b2), eps=eps)
    dev = named[0][1].device
    if fused is None:
        fused = dev.type == "cuda" and named[0][1].dtype == torch.float32
    if fused:
        try:
            return torch.optim.AdamW(groups, fused=True, **kw)
        except (RuntimeError, TypeError, ValueError):
            pass
    return torch.optim.AdamW(groups, foreach=dev.type == "cuda", **kw)


@torch.no_grad()
def clip_grads(params, max_norm: float) -> torch.Tensor:
    """Global-norm clipping with NumPy's formula: scale = min(1, clip / (norm + 1e-12)). Returns the norm (no sync)."""
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        raise RuntimeError("no gradients to clip")
    try:
        norms = torch._foreach_norm(grads)
    except (AttributeError, RuntimeError):
        norms = [torch.linalg.vector_norm(g) for g in grads]
    total = torch.linalg.vector_norm(torch.stack([n.to(grads[0].dtype) for n in norms]))
    scale = torch.clamp(max_norm / (total + 1e-12), max=1.0)
    try:
        torch._foreach_mul_(grads, scale)
    except (AttributeError, RuntimeError, TypeError):
        for g in grads:
            g.mul_(scale.to(g.dtype))
    return total


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for g in opt.param_groups:
        g["lr"] = lr


def lr_at(step: int, steps: int, lr: float) -> float:
    """pine.train_np.run's schedule: linear warmup over max(20, steps // 20) steps, cosine to 10%."""
    warm = max(20, steps // 20)
    if step <= warm:
        return lr * step / warm
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))


# ---------------------------------------------------------------------- batches, identical to pine.features.batch
def batch_torch(w: torch.Tensor, T: int, K: int, mask: torch.Tensor):
    """w (B, T+K, D) normalised frames, mask (B, D) -> (inputs (B, T, 2D), targets (B, T, K*D))."""
    x = w[:, :T] * mask[:, None, :]
    inp = torch.cat([x, mask[:, None, :].expand_as(x)], -1)
    tgt = torch.cat([w[:, 1 + j:T + 1 + j] for j in range(K)], -1)
    return inp, tgt


# ---------------------------------------------------------------------- masks, same distribution as pine.features.make_masks
_MASK_CONSTS: dict = {}


def _mask_consts(device):
    from pine import features as F
    key = str(device)
    if key not in _MASK_CONSTS:
        hidden = torch.zeros(F.D, dtype=torch.bool)
        hidden[list(F.HIDDEN)] = True
        groups = torch.zeros(len(F.GROUPS), F.D, dtype=torch.bool)
        for gi, idx in enumerate(F.GROUPS.values()):
            groups[gi, list(idx)] = True
        _MASK_CONSTS[key] = (hidden.to(device), groups.to(device))
    return _MASK_CONSTS[key]


def masks_torch(B: int, mode: str, generator=None, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """'full' | 'exo' | 'mixed' (50% full, 30% EXO channels only, 20% each channel group dropped with p=0.5)."""
    hidden, groups = _mask_consts(device)
    D = hidden.numel()
    if mode == "full":
        return torch.ones(B, D, device=device, dtype=dtype)
    if mode == "exo":
        return (~hidden).to(dtype)[None].expand(B, D).clone()
    if mode != "mixed":
        raise ValueError(f"unknown mask mode {mode!r}")
    u = torch.rand(B, generator=generator, device=device)
    exo = (u >= 0.5) & (u < 0.8)
    rnd = u >= 0.8
    drop_g = (torch.rand(B, groups.shape[0], generator=generator, device=device) < 0.5) & rnd[:, None]
    drop = (drop_g[:, :, None] & groups[None]).any(1) | (exo[:, None] & hidden[None])
    return (~drop).to(dtype)


# ---------------------------------------------------------------------- evaluation, identical to pine.train_np.evaluate
@torch.no_grad()
def evaluate_torch(model, val_w, std, T: int, K: int, batch_size: int = 512) -> dict:
    """val_w: (N, T+K, D) normalised windows (numpy or tensor). Runs in the model's dtype, no autocast."""
    from pine import features as F
    m = unwrap(model)
    dev, dt = m.win.device, m.win.dtype
    hid = [j for j in F.HIDDEN if j in F.ANGLE_DIMS]
    std_h = torch.as_tensor(np.asarray(std, dtype=np.float64)[hid], device=dev)
    out = {}
    with torch.autocast(device_type=dev.type, enabled=False):
        for mode in ("full", "exo"):
            mask = torch.as_tensor(F.make_masks(1, mode, np.random.default_rng(0))[0], device=dev, dtype=dt)
            losses, se, se_base, n = [], 0.0, 0.0, 0
            for i in range(0, len(val_w), batch_size):
                w = torch.as_tensor(val_w[i:i + batch_size], device=dev).to(dt)
                inp, tgt = batch_torch(w, T, K, mask[None].expand(len(w), -1))
                y = m(inp)
                losses.append(((y - tgt) ** 2).mean(dim=(1, 2)).double().cpu())
                if mode == "exo":   # horizon-1 error on the joints the suit never saw, in degrees
                    err = (y[..., hid] - tgt[..., hid]).double() * std_h
                    se += float((torch.rad2deg(err) ** 2).sum()); n += err.numel()
                    se_base += float((torch.rad2deg(tgt[..., hid].double() * std_h) ** 2).sum())
            pw = torch.cat(losses).numpy()
            out[mode] = float(pw.mean())
            if mode == "full":
                out["per_window"] = np.round(pw, 5).tolist()
            if mode == "exo":
                out["hidden_joint_rmse_deg"] = math.sqrt(se / n)
                out["hidden_joint_rmse_deg_meanpose"] = math.sqrt(se_base / n)
    return out
