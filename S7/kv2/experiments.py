"""The two training experiments.

E1 discrimination probe - the decisive test. A task whose answer depends on
   WHICH of two colliding words appeared. Under Kronecker-V1 those words have
   byte-identical codes, so the model receives literally the same input for both
   classes and cannot beat chance no matter how long it trains. This is not a
   quality gap, it is an information-theoretic wall. Elastic separates them, so
   the same model learns the task.

E2 language-model comparison - checks the fix does not cost anything on ordinary
   next-word prediction over real multilingual text.
"""
from __future__ import annotations
import numpy as np
from .transformer import TinyTransformer


def _codes_matrix(vocab_words, codec):
    """Frozen code table, row-normalised so every input path enters the model at
    a comparable scale (otherwise the comparison measures step size, not codec)."""
    C = np.stack([codec.encode(w) for w in vocab_words]).astype(np.float64)
    n = np.linalg.norm(C, axis=1, keepdims=True)
    return C / np.clip(n, 1e-9, None)


def _train(model, X, Y, Xv, Yv, steps=400, lr=0.05, seed=0, clip=1.0):
    hist = []
    rng = np.random.default_rng(seed)
    B = X.shape[0]
    mv = np.zeros(Xv.shape); mv[:, -1] = 1
    for t in range(steps):
        idx = rng.choice(B, min(48, B), replace=False)
        xb, yb = X[idx], Y[idx]
        m = np.zeros(xb.shape); m[:, -1] = 1
        model.loss(xb, yb, m)
        g = model.backward()
        gn = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))   # global norm
        if gn > clip:
            g = {k: v * (clip / gn) for k, v in g.items()}
        model.step(g, lr)
        if (t + 1) % 25 == 0:
            l, a = model.loss(Xv, Yv, mv)
            hist.append({"step": t + 1, "val_loss": round(l, 4), "val_acc": round(a, 4)})
    l, a = model.loss(Xv, Yv, mv)
    return {"final_val_loss": round(l, 4), "final_val_acc": round(a, 4), "history": hist}


def discrimination_probe(pairs, codecs, seed=0, steps=400, n=600):
    """pairs: [(wordA, wordB, script)]. Builds a task where the label is
    determined solely by which of the two words is present."""
    rng = np.random.default_rng(seed)
    fillers = ["the", "of", "and", "in", "a"]
    results = {}
    pair_reports = []
    for (wa, wb, script) in pairs[:3]:
        # Sequence: [word, filler, filler]  ->  predict <A> or <B> at the last
        # position. The label depends ONLY on which of the two words is at
        # position 0, so a model that cannot tell them apart cannot beat chance.
        vocab_words = [wa, wb] + fillers + ["<A>", "<B>"]
        ia, ib = 0, 1
        LA, LB = len(vocab_words) - 2, len(vocab_words) - 1
        T = 3
        X = np.zeros((n, T), dtype=int); Y = np.zeros((n, T), dtype=int)
        for i in range(n):
            first = ia if i % 2 == 0 else ib
            X[i, 0] = first
            for t in range(1, T):
                X[i, t] = 2 + rng.integers(0, len(fillers))
            Y[i, :] = LA if first == ia else LB              # only last is scored
        perm = rng.permutation(n); X, Y = X[perm], Y[perm]
        split = int(0.8 * n)
        Xtr, Ytr, Xv, Yv = X[:split], Y[:split], X[split:], Y[split:]
        row = {"pair": [wa, wb], "script": script}
        for cname, codec in codecs.items():
            if codec is None:
                m = TinyTransformer(len(vocab_words), d=48, max_T=T, seed=1,
                                    input_path="dense")
            else:
                codes = _codes_matrix(vocab_words, codec)
                m = TinyTransformer(len(vocab_words), d=48, max_T=T, seed=1,
                                    input_path="codec", codes=codes)
                row.setdefault("codes_identical", {})[cname] = bool(
                    np.allclose(codes[ia], codes[ib]))
            r = _train(m, Xtr, Ytr, Xv, Yv, steps=steps, lr=0.08, seed=seed)
            row[cname] = {"val_acc": r["final_val_acc"], "val_loss": r["final_val_loss"],
                          "embed_params": m.embed_params(), "history": r["history"]}
        pair_reports.append(row)
    # aggregate
    for cname in codecs:
        accs = [p[cname]["val_acc"] for p in pair_reports]
        results[cname] = {"mean_val_acc": round(float(np.mean(accs)), 4),
                          "per_pair": accs}
    return {"pairs": pair_reports, "summary": results}


def lm_experiment(words, codecs, seed=0, steps=500, vocab_size=120, T=5):
    """Ordinary next-word prediction over real multilingual text."""
    from .vocab import load_stream
    vocab_words = [w for w, s, c in words[:vocab_size]]
    stream = load_stream(vocab_words)        # REAL ordered text, not shuffled ids
    seqs = np.array([stream[i:i + T + 1] for i in range(0, len(stream) - T - 1)])
    if len(seqs) > 4000:                     # keep the demo fast and deterministic
        seqs = seqs[:4000]
    X, Y = seqs[:, :-1], seqs[:, 1:]
    split = int(0.8 * len(X))
    out = {}
    for cname, codec in codecs.items():
        if codec is None:
            m = TinyTransformer(len(vocab_words), d=48, max_T=T, seed=2, input_path="dense")
        else:
            m = TinyTransformer(len(vocab_words), d=48, max_T=T, seed=2, input_path="codec",
                                codes=_codes_matrix(vocab_words, codec))
        r = _train(m, X[:split], Y[:split], X[split:], Y[split:], steps=steps, lr=0.2, seed=seed)
        out[cname] = {"final_val_loss": r["final_val_loss"], "final_val_acc": r["final_val_acc"],
                      "embed_params": m.embed_params(), "total_params": m.n_params()}
    return {"vocab_size": len(vocab_words), "sequences": len(X), "results": out}
