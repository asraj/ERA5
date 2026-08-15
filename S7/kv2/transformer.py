"""A small but genuine transformer, written in numpy with hand-derived gradients.

One causal self-attention block + MLP, LayerNorm, cross-entropy. The backward
pass is verified against numerical differentiation (`gradcheck`), so the training
results in the paper rest on arithmetic that is provably correct rather than on a
framework we are trusting.

The input path is swappable, which is the entire point:
    dense    - a learned V x d table (the control arm)
    codec    - frozen Kronecker code (V1 or Elastic) -> one trainable projection
"""
from __future__ import annotations
import numpy as np


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class LayerNorm:
    def __init__(self, d, eps=1e-5):
        self.g, self.b, self.eps = np.ones(d), np.zeros(d), eps

    def __call__(self, x):
        mu = x.mean(-1, keepdims=True)
        var = x.var(-1, keepdims=True)
        self.cache = (x, mu, var)
        self.xhat = (x - mu) / np.sqrt(var + self.eps)
        return self.g * self.xhat + self.b

    def backward(self, dout):
        x, mu, var = self.cache
        D = x.shape[-1]
        std = np.sqrt(var + self.eps)
        dg = (dout * self.xhat).reshape(-1, D).sum(0)
        db = dout.reshape(-1, D).sum(0)
        dxhat = dout * self.g
        dx = (dxhat - dxhat.mean(-1, keepdims=True)
              - self.xhat * (dxhat * self.xhat).mean(-1, keepdims=True)) / std
        return dx, dg, db


class TinyTransformer:
    """[B,T] token ids -> [B,T,V] logits."""

    def __init__(self, vocab, d=64, max_T=8, seed=0, input_path="dense",
                 codes=None, ff_mult=2):
        rng = np.random.default_rng(seed)
        self.V, self.d, self.max_T, self.input_path = vocab, d, max_T, input_path
        s = 0.08
        if input_path == "dense":
            self.E = rng.normal(0, s, (vocab, d))          # trainable table
            self.codes = None
        else:
            self.codes = codes                              # [V, code_dim] FROZEN
            self.P = rng.normal(0, s / np.sqrt(codes.shape[1] / d), (codes.shape[1], d))
        self.pos = rng.normal(0, s, (max_T, d))
        self.Wq = rng.normal(0, s, (d, d)); self.Wk = rng.normal(0, s, (d, d))
        self.Wv = rng.normal(0, s, (d, d)); self.Wo = rng.normal(0, s, (d, d))
        self.W1 = rng.normal(0, s, (d, ff_mult * d)); self.b1 = np.zeros(ff_mult * d)
        self.W2 = rng.normal(0, s, (ff_mult * d, d)); self.b2 = np.zeros(d)
        self.Wout = rng.normal(0, s, (d, vocab)); self.bout = np.zeros(vocab)
        self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)

    # ---------------- parameters ----------------
    def params(self):
        names = ["pos", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "Wout", "bout"]
        names += ["E"] if self.input_path == "dense" else ["P"]
        return {n: getattr(self, n) for n in names} | {"ln1.g": self.ln1.g, "ln1.b": self.ln1.b,
                                                       "ln2.g": self.ln2.g, "ln2.b": self.ln2.b}

    def n_params(self):
        return sum(v.size for v in self.params().values())

    def embed_params(self):
        return self.E.size if self.input_path == "dense" else self.P.size

    # ---------------- forward ----------------
    def forward(self, ids):
        B, T = ids.shape
        d = self.d
        if self.input_path == "dense":
            x0 = self.E[ids]
        else:
            C = self.codes[ids]                             # [B,T,code_dim] frozen
            x0 = C @ self.P
        x = x0 + self.pos[:T]
        h = self.ln1(x)
        q, k, v = h @ self.Wq, h @ self.Wk, h @ self.Wv
        att = q @ k.transpose(0, 2, 1) / np.sqrt(d)
        mask = np.triu(np.ones((T, T), bool), 1)
        att = np.where(mask, -1e9, att)
        A = softmax(att, -1)
        ctx = A @ v
        proj = ctx @ self.Wo
        x1 = x + proj
        h2 = self.ln2(x1)
        z1 = h2 @ self.W1 + self.b1
        a1 = np.tanh(z1)
        z2 = a1 @ self.W2 + self.b2
        x2 = x1 + z2
        logits = x2 @ self.Wout + self.bout
        self.cache = (ids, x0, x, h, q, k, v, A, ctx, x1, h2, z1, a1, x2)
        return logits

    def loss(self, ids, targets, mask=None):
        """Cross-entropy on positions where mask==1 (default: last position)."""
        logits = self.forward(ids)
        B, T, V = logits.shape
        if mask is None:
            mask = np.zeros((B, T)); mask[:, -1] = 1
        P = softmax(logits, -1)
        idx = np.take_along_axis(P, targets[:, :, None], -1)[:, :, 0]
        nll = -np.log(np.clip(idx, 1e-12, None))
        n = max(mask.sum(), 1)
        self._bwd_cache = (P, targets, mask, n)
        acc = ((P.argmax(-1) == targets) * mask).sum() / n
        return float((nll * mask).sum() / n), float(acc)

    # ---------------- backward ----------------
    def backward(self):
        P, targets, mask, n = self._bwd_cache
        ids, x0, x, h, q, k, v, A, ctx, x1, h2, z1, a1, x2 = self.cache
        B, T, d = x.shape
        dlogits = P.copy()
        np.put_along_axis(dlogits, targets[:, :, None],
                          np.take_along_axis(dlogits, targets[:, :, None], -1) - 1, -1)
        dlogits *= (mask / n)[:, :, None]
        g = {}
        g["Wout"] = np.einsum("btd,btv->dv", x2, dlogits)
        g["bout"] = dlogits.sum((0, 1))
        dx2 = dlogits @ self.Wout.T
        # MLP
        dz2 = dx2
        g["W2"] = np.einsum("btf,btd->fd", a1, dz2); g["b2"] = dz2.sum((0, 1))
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (1 - a1 ** 2)
        g["W1"] = np.einsum("btd,btf->df", h2, dz1); g["b1"] = dz1.sum((0, 1))
        dh2 = dz1 @ self.W1.T
        dx1_ln, g["ln2.g"], g["ln2.b"] = self.ln2.backward(dh2)
        dx1 = dx2 + dx1_ln
        # attention
        dproj = dx1
        g["Wo"] = np.einsum("btd,bte->de", ctx, dproj)
        dctx = dproj @ self.Wo.T
        dA = dctx @ v.transpose(0, 2, 1)
        dv = A.transpose(0, 2, 1) @ dctx
        datt = A * (dA - (dA * A).sum(-1, keepdims=True))
        dq = datt @ k / np.sqrt(d)
        dk = datt.transpose(0, 2, 1) @ q / np.sqrt(d)
        g["Wq"] = np.einsum("btd,bte->de", h, dq)
        g["Wk"] = np.einsum("btd,bte->de", h, dk)
        g["Wv"] = np.einsum("btd,bte->de", h, dv)
        dh = dq @ self.Wq.T + dk @ self.Wk.T + dv @ self.Wv.T
        dx_ln, g["ln1.g"], g["ln1.b"] = self.ln1.backward(dh)
        dx = dx1 + dx_ln
        gpos = np.zeros_like(self.pos)      # only the first T rows are gathered,
        gpos[:T] = dx.sum(0)                # the rest receive exactly zero
        g["pos"] = gpos
        if self.input_path == "dense":
            gE = np.zeros_like(self.E)
            np.add.at(gE, ids, dx)                          # scatter-add
            g["E"] = gE
        else:
            C = self.codes[ids]
            g["P"] = np.einsum("btc,btd->cd", C, dx)        # one shared projection
        return g

    def step(self, g, lr):
        for name, grad in g.items():
            if "." in name:
                obj, attr = name.split(".")
                cur = getattr(getattr(self, obj), attr)
                cur -= lr * grad
            else:
                getattr(self, name)[...] -= lr * grad


def gradcheck(seed=0, tol=2e-4) -> dict:
    """Numerical verification that the hand-written backward pass is correct."""
    rng = np.random.default_rng(seed)
    V, d, T, B = 11, 16, 4, 3
    codes = rng.normal(0, 1, (V, 40))
    out = {}
    for path in ("dense", "codec"):
        m = TinyTransformer(V, d=d, max_T=T, seed=1, input_path=path, codes=codes)
        ids = rng.integers(0, V, (B, T)); tgt = rng.integers(0, V, (B, T))
        m.loss(ids, tgt); g = m.backward()
        worst = 0.0
        for name in list(g)[:6]:
            arr = (getattr(m, name) if "." not in name
                   else getattr(getattr(m, name.split(".")[0]), name.split(".")[1]))
            flat = arr.reshape(-1); gf = g[name].reshape(-1)
            for i in rng.choice(flat.size, min(6, flat.size), replace=False):
                orig = flat[i]; eps = 1e-5
                flat[i] = orig + eps; lp, _ = m.loss(ids, tgt)
                flat[i] = orig - eps; lm, _ = m.loss(ids, tgt)
                flat[i] = orig
                num = (lp - lm) / (2 * eps)
                worst = max(worst, abs(num - gf[i]) / max(1e-8, abs(num) + abs(gf[i])))
        out[path] = {"max_relative_error": float(worst), "passed": bool(worst < tol)}
    return out
