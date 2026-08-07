"""A tiny but real next-token model (numpy).

Small on purpose - the assignment grades the data system, not the model - but the
loss is genuine cross-entropy with real gradients and real SGD, so the learning
ledger and the token-level perplexity trace carry actual signal rather than
invented numbers.

Architecture: token embedding -> mean of a short causal context -> tanh hidden
-> vocab logits. Loss is applied ONLY where loss_mask == 1.
"""
from __future__ import annotations
import numpy as np


class TinyLM:
    def __init__(self, vocab: int, dim: int = 48, ctx: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab, self.dim, self.ctx = vocab, dim, ctx
        self.E = rng.normal(0, 0.05, (vocab, dim))
        self.W = rng.normal(0, 0.05, (dim, vocab))
        self.b = np.zeros(vocab)

    # ---- state ----
    def state(self) -> dict:
        return {"E": self.E.copy(), "W": self.W.copy(), "b": self.b.copy()}

    def load(self, s: dict):
        self.E, self.W, self.b = s["E"].copy(), s["W"].copy(), s["b"].copy()

    def param_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for a in (self.E, self.W, self.b):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    # ---- forward / backward on one packed sequence ----
    def step(self, tokens, loss_mask, segment_ids, lr: float = 0.3):
        """Returns (mean_loss, per_token_losses, grad_norm). Positions whose
        loss_mask is 0 contribute nothing to the gradient."""
        T = len(tokens)
        toks = np.asarray(tokens, dtype=np.int64)
        segs = np.asarray(segment_ids)
        idx = [t for t in range(1, T) if loss_mask[t] == 1]
        if not idx:
            return 0.0, {}, 0.0

        # causal context: mean embedding of up to ctx previous tokens IN THE SAME segment
        ctx_vecs, targets, keep = [], [], []
        for t in idx:
            lo = max(0, t - self.ctx)
            prev = [p for p in range(lo, t) if segs[p] == segs[t]]
            if not prev:
                continue
            ctx_vecs.append(self.E[toks[prev]].mean(axis=0))
            targets.append(toks[t]); keep.append(t)
        if not keep:
            return 0.0, {}, 0.0

        X = np.stack(ctx_vecs)                       # (N, dim)
        H = np.tanh(X)
        logits = H @ self.W + self.b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits); P /= P.sum(axis=1, keepdims=True)
        y = np.asarray(targets)
        losses = -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))

        # backward
        dlogits = P.copy(); dlogits[np.arange(len(y)), y] -= 1.0; dlogits /= len(y)
        dW = H.T @ dlogits
        db = dlogits.sum(axis=0)
        dH = dlogits @ self.W.T
        dX = dH * (1 - H ** 2)
        gnorm = float(np.sqrt((dW ** 2).sum() + (db ** 2).sum() + (dX ** 2).sum()))

        self.W -= lr * dW
        self.b -= lr * db
        for n, t in enumerate(keep):                 # scatter grad back to context embeddings
            lo = max(0, t - self.ctx)
            prev = [p for p in range(lo, t) if segs[p] == segs[t]]
            g = dX[n] / len(prev)
            for p in prev:
                self.E[toks[p]] -= lr * g

        per_token = {int(t): float(l) for t, l in zip(keep, losses)}
        return float(losses.mean()), per_token, gnorm

    @staticmethod
    def top_perplexity(per_token: dict, tokens, k: int = 3) -> list:
        """Token-level trace: the most surprising loss-bearing positions."""
        items = sorted(per_token.items(), key=lambda kv: -kv[1])[:k]
        return [{"position": p, "token_id": int(tokens[p]),
                 "loss": round(l, 5), "perplexity": round(float(np.exp(min(l, 20))), 3)}
                for p, l in items]
