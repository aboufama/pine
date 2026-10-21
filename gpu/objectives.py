"""The optional "dct16mix" objective for one extra run at the end of the GPU sweep (the main ladder never uses it).

  chunk     16 future frames per position, executed 8 at a time in evaluation (pi0-style action chunking)
  dct       the chunk stored as its 6 lowest DCT-II coefficients per channel (pi0-FAST style; scripts/pi_pass.dct_basis)
  mix       targets are offsets from the input frame for channels the input mask marks as measured, and absolute values
            for masked channels. Equivalently: offsets from the MASKED input frame (masked channels are 0 there).
            A pure residual target breaks suit fill-in: decoding it needs the current angle of a joint the suit never
            measures. With an all-ones mask this is exactly pi_pass's residual "dct16" objective.
  noisy     noisy-history augmentation of the context during training (research/history_corruption.noisy_history,
            ported to torch): on half the windows a random 4-32 frame suffix of the context gets smooth drift plus
            jitter, on measured channels only; targets are written from the corrupted frames the model saw.

The math is written once for NumPy and torch (`lib` is the numpy or torch module), so gpu/tests checks it against
scripts/pi_pass.Objective on CPU, and train_torch.py checks it again on the pod before the run starts.
"""
from __future__ import annotations
import numpy as np

NAME = "dct16mix"
H, K_COEF, EXECUTE = 16, 6, 8


def dct_basis(h: int, k: int) -> np.ndarray:
    """Orthonormal DCT-II basis (h, k): identical to scripts/pi_pass.dct_basis."""
    n = np.arange(h)[:, None]; j = np.arange(k)[None]
    B = np.cos(np.pi * (n + 0.5) * j / h) * np.sqrt(2.0 / h); B[:, 0] /= np.sqrt(2.0)
    return B.astype(np.float32)


class ChunkObjective:
    def __init__(self, D: int, h: int = H, k: int = K_COEF, execute: int = EXECUTE, name: str = NAME):
        self.D, self.H, self.k, self.execute, self.name = D, h, k, execute, name
        self.B = dct_basis(h, k)
        self.rows = k
        self.out_dim = k * D

    def targets(self, lib, Bm, w, base, T: int):
        """w (b, T+H, D) clean frames; base (b, >=T, D) = the masked input frames -> (b, T, k*D) coefficients."""
        fut = lib.stack([w[:, 1 + j:T + 1 + j] for j in range(self.H)], 2)          # (b, T, H, D)
        fut = fut - base[:, :T, None, :]
        return lib.einsum("hk,bthd->btkd", Bm, fut).reshape(w.shape[0], T, -1)

    def decode(self, lib, Bm, y, base_last):
        """y (n, k*D) at one position, base_last (n, D) masked input frame -> (n, H, D) future frames."""
        c = y.reshape(y.shape[0], self.rows, self.D)
        return lib.einsum("hk,nkd->nhd", Bm, c) + base_last[:, None, :]

    def first_frame(self, lib, Bm, y, base):
        """Horizon-1 prediction at every position: y (b, T, k*D), base (b, T, D) -> (b, T, D)."""
        c = y.reshape(y.shape[0], y.shape[1], self.rows, self.D)
        return lib.einsum("k,btkd->btd", Bm[0], c) + base


def noisy_history_torch(ctx, mask, gen, p=0.5, umax=2.0, rate=0.15, cap=1.0, nmin=4, jitter=0.1):
    """research/history_corruption.noisy_history in torch (same parameters and distribution, its own random stream)."""
    import torch
    b, T, D = ctx.shape
    dev = ctx.device
    sel = (torch.rand(b, generator=gen, device=dev) < p).float()
    n = torch.randint(nmin, T + 1, (b,), generator=gen, device=dev)
    end = torch.clamp(torch.rand(b, generator=gen, device=dev) * umax * rate * torch.sqrt(n.float()), max=cap) * sel
    on = (torch.arange(T, device=dev)[None] >= (T - n)[:, None]).float()[..., None]
    acc = torch.randn(b, T, D, generator=gen, device=dev) * on
    drift = torch.cumsum(torch.cumsum(acc, 1), 1)
    nn = n.double()
    drift = drift / torch.sqrt(nn * (nn + 1) * (2 * nn + 1) / 6).float()[:, None, None]   # unit std at the end
    jit = torch.randn(b, T, D, generator=gen, device=dev) * on * jitter
    corr = (drift + jit) * end[:, None, None] * mask[:, None, :]
    return ctx + corr


def check_against_pi_pass(obj: ChunkObjective, lib=np, Bm=None, to_lib=None, to_np=None) -> dict:
    """Targets and decode vs scripts/pi_pass.Objective('dct16', 16, residual, k=6) with an all-ones mask, plus the
    mixed rule on a random mask. Returns max abs differences (should be ~1e-6 or smaller)."""
    import importlib, os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    pi = importlib.import_module("scripts.pi_pass")
    ref = pi.Objective("dct16", obj.H, True, k=obj.k, execute=obj.execute)
    to_lib = to_lib or (lambda a: a)
    to_np = to_np or (lambda a: np.asarray(a))
    Bm = to_lib(obj.B) if Bm is None else Bm
    r = np.random.default_rng(3)
    T = 32
    w = r.standard_normal((8, T + obj.H, obj.D)).astype(np.float32)
    out = dict(basis=float(np.abs(pi.dct_basis(obj.H, obj.k) - obj.B).max()))
    got = to_np(obj.targets(lib, Bm, to_lib(w), to_lib(w[:, :T]), T))
    out["targets_all_measured"] = float(np.abs(got - ref.targets(w, T)).max())
    y = r.standard_normal((8, obj.out_dim)).astype(np.float32)
    out["decode_all_measured"] = float(np.abs(to_np(obj.decode(lib, Bm, to_lib(y), to_lib(w[:, T - 1]))) - ref.decode(y, w[:, T - 1])).max())
    mask = (r.random((8, obj.D)) < 0.5).astype(np.float32)
    x = w[:, :T] * mask[:, None, :]
    got = to_np(obj.targets(lib, Bm, to_lib(w), to_lib(x), T)).reshape(8, T, obj.k, obj.D)
    fut = np.stack([w[:, 1 + j:T + 1 + j] for j in range(obj.H)], 2)
    want = np.einsum("hk,bthd->btkd", obj.B, np.where(mask[:, None, None, :] > 0, fut - w[:, :T, None, :], fut))
    out["targets_mixed"] = float(np.abs(got - want).max())
    full = to_np(obj.decode(lib, Bm, to_lib(got[:, -1].reshape(8, -1)), to_lib(x[:, -1])))
    out["roundtrip_mixed"] = float(np.abs(full - (obj.B @ obj.B.T @ fut[:, -1])).max())  # projection of the truth
    return out
