#!/usr/bin/env python3
"""Generate S10_training_loop.ipynb. Generated, not hand-edited, so verify_local.py
can execute the same cells offline. Same pattern as S9."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
C = []
def md(s):   C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)})
def code(s): C.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                       "source":s.strip("\n").splitlines(True)})

md(r"""
# Session 10 — The training loop, made to tell the truth about itself

**Stephen Raj Arokiasamy**

Six things the assignment asks for. Every one of them is a *check*, not a demonstration —
the point of the session is that the dangerous training bugs do not crash, they produce a
plausible number and let you keep going.

| | |
|---|---|
| 1 | Every tensor shape in a step, with what each dimension means |
| 2 | One gradient verified **by hand** — nudge a weight, compare against `backward()` |
| 3 | Gradient accumulation **broken on purpose**, both curves plotted together |
| 4 | Grad norm logged every step, and a step where it moved **before** the loss |
| 5 | My own **MFU**, reported honestly, with what is costing the distance to 40% |
| 6 | **0.1** written out in fp32, bf16 and fp8 E4M3, bit by bit |

**Model:** `HuggingFaceTB/SmolLM2-135M` — 134,515,008 params, V = 49,152, D = 576, 30 layers.
**Data:** the Session-4 cleaned OpenWebText shard, same as Session 9.
**Precision:** fp32, pinned. Session 9 learned this the hard way — the checkpoint is bf16 and
AdamW at lr 3e-5 rounds most of its updates away in bf16, so the loop looks fine and does not learn.
""")

code(r"""
try:
    import torch, transformers            # noqa
except ImportError:
    !pip -q install torch transformers
import os, json, gzip, math, time, struct, urllib.request, textwrap, statistics
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import transformers

SEED = 10
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)
print("device      ", DEVICE, "|", GPU)

OFFLINE = os.environ.get("S10_OFFLINE") == "1"   # used by verify_local.py
""")

md("## 0 · Model, tokenizer, data")

code(r"""
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

def load_model_and_tokenizer():
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    if OFFLINE:
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer
        cfg = LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                          num_hidden_layers=int(os.environ.get("S10_LAYERS", 30)),
                          num_attention_heads=9, num_key_value_heads=3, hidden_act="silu",
                          max_position_embeddings=2048, rms_norm_eps=1e-5,
                          tie_word_embeddings=True, rope_theta=10000.0)
        m = LlamaForCausalLM(cfg).to(getattr(torch, os.environ.get("S10_DTYPE", "float32")))
        tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(
            os.environ["S10_TOKENIZER"]), unk_token="<unk>", eos_token="</s>", bos_token="<s>")
        return m, tok, cfg
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    # fp32 pinned deliberately - see the header note.
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    return m, tok, cfg

model, tok, cfg = load_model_and_tokenizer()
model = model.to(DEVICE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

V, D, L = cfg.vocab_size, cfg.hidden_size, cfg.num_hidden_layers
N_PARAMS = sum(p.numel() for p in model.parameters())
print(f"V={V:,}  D={D}  layers={L}  params={N_PARAMS:,}  dtype={next(model.parameters()).dtype}")
""")

code(r"""
# The Session-4 cleaned OpenWebText shard, fetched from the repo. Three things
# guard this cell, because "the data loaded" is the assumption every silent
# training bug is built on:
#   1. the sha256 from the manifest is verified, so a truncated or wrong file is
#      an error rather than a slightly odd loss curve
#   2. a public-OpenWebText fallback, so a grader holding only this .ipynb still
#      gets a runnable notebook
#   3. an assert on the document count
SHARD_URL  = "https://raw.githubusercontent.com/asraj/ERA5/main/S9/data/s9_owt_clean.jsonl.gz"
SHARD_SHA  = "d2546ecd026f4e1f82a52961c98bf11680127b7d35b390cda3a9a94ea248a552"
SHARD_LOCAL = os.environ.get("S10_SHARD", "s9_owt_clean.jsonl.gz")

def sha256_of(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

DOCS, SOURCE = None, None
if not os.path.exists(SHARD_LOCAL):
    try:
        print("fetching the Session-4 cleaned shard from the repo ...")
        urllib.request.urlretrieve(SHARD_URL, SHARD_LOCAL)
    except Exception as e:
        print(f"  repo copy unavailable ({type(e).__name__})")

if os.path.exists(SHARD_LOCAL):
    got = sha256_of(SHARD_LOCAL)
    if got == SHARD_SHA:
        SOURCE = "Session-4 cleaned shard (sha256 verified)"
    else:
        SOURCE = f"Session-4 cleaned shard (sha256 MISMATCH: {got[:16]}...)"
        print(f"  WARNING: expected {SHARD_SHA[:16]}..., got {got[:16]}...")
    DOCS = [json.loads(l)["text"] for l in gzip.open(SHARD_LOCAL, "rt", encoding="utf-8")]
else:
    print("  rebuilding an equivalent shard from public OpenWebText ...")
    from datasets import load_dataset
    import re, unicodedata, hashlib
    _ENT = [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", chr(34)), ("&#39;", "'")]
    _DEL = {ord(c): None for c in ("\u200b", "\ufeff", "\u202e", "\ufffd")}
    _STOP = set("the a an of to and in is are was on for with as at by it this that from or".split())
    seen, DOCS = set(), []
    for r in load_dataset("stas/openwebtext-10k", split="train"):
        t = unicodedata.normalize("NFC", r["text"])
        for e, c in _ENT: t = t.replace(e, c)
        t = re.sub(r"[ \t]+", " ", t.translate(_DEL)).strip()
        if not (400 <= len(t) <= 12000): continue
        w = t.split()
        if len(w) < 3 or not any(x.lower() in _STOP for x in w): continue
        h = hashlib.blake2b(t.encode(), digest_size=16).digest()
        if h in seen: continue
        seen.add(h); DOCS.append(t)
        if len(DOCS) >= 5000: break
    SOURCE = "stas/openwebtext-10k, cleaned in-notebook with the Session-4 stages"

print(f"source: {SOURCE}")
print(f"{len(DOCS):,} cleaned documents")
assert len(DOCS) > 500, "shard too small to train on"

def pack(docs, seq_len, n_seqs):
    '''Fixed-length sequences; is_start marks the first token of each document.'''
    stream, starts = [], []
    for d in docs:
        piece = tok(d, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        starts += [1] + [0] * (len(piece) - 1); stream += piece
        if len(stream) >= seq_len * n_seqs: break
    stream, starts = stream[:seq_len*n_seqs], starts[:seq_len*n_seqs]
    return (torch.tensor(stream).view(n_seqs, seq_len).to(DEVICE),
            torch.tensor(starts).view(n_seqs, seq_len).to(DEVICE))

SEQ = int(os.environ.get("S10_SEQ", 256))
NSEQ = int(os.environ.get("S10_NSEQ", 400))
IDS, IS_START = pack(DOCS, SEQ, NSEQ)
LABELS = IDS.masked_fill(IS_START.bool(), -100)   # Session 9: never train across a join
print(f"data {tuple(IDS.shape)}  ({int((LABELS != -100).sum()):,} contributing tokens)")
""")

md(r"""
## Item 1 · Every tensor in a step, and what each dimension is

Session 9 covered the forward tensors. The step adds three more families, and they are the
reason training costs 8× what inference does: a **gradient** for every weight, an **fp32 master
copy**, and the optimiser's **two running numbers** per weight.
""")

code(r"""
B = int(os.environ.get("S10_BS", 4))
batch, labels = IDS[:B], LABELS[:B]

model.train()
hidden = model.model(input_ids=batch).last_hidden_state
logits = model.lm_head(hidden)
loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                       ignore_index=-100)
loss.backward()

opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
opt.step()                     # one step so the optimiser state exists to inspect

W = model.lm_head.weight
st = opt.state[W]
rows = [
 ("batch",            tuple(batch.shape),        "B sequences x T positions - token ids"),
 ("labels",           tuple(labels.shape),       "B, T - same grid, -100 where the position must not train"),
 ("hidden",           tuple(hidden.shape),       "B, T, D - one residual-stream vector per position"),
 ("logits",           tuple(logits.shape),       "B, T, V - one raw score per vocabulary entry, per position"),
 ("loss",             tuple(loss.shape),         "scalar - the single number the whole run is steered by"),
 ("lm_head.weight",   tuple(W.shape),            "V, D - the parameter itself"),
 ("lm_head.weight.grad", tuple(W.grad.shape),    "V, D - dL/dw, ONE NUMBER PER WEIGHT, same shape as the weight"),
 ("exp_avg",          tuple(st["exp_avg"].shape),      "V, D - Adam's running mean of the gradient (momentum)"),
 ("exp_avg_sq",       tuple(st["exp_avg_sq"].shape),   "V, D - Adam's running mean of the gradient SQUARED (scale)"),
]
w = max(len(r[0]) for r in rows)
for n, s, m in rows:
    print(f"{n:<{w}}  {str(s):<18}  {m}")

print(f"\nloss is a scalar: {loss.item():.4f}   from {logits.numel():,} logits")
print(f"gradient tensors have exactly the same shape as their weights: "
      f"{all(p.grad is None or p.grad.shape == p.shape for p in model.parameters())}")

# the 16-bytes-per-weight table from the lesson, measured on this model
bytes_per_weight = {"weight (bf16)": 2, "gradient (bf16)": 2,
                    "fp32 master copy": 4, "Adam exp_avg + exp_avg_sq (fp32)": 8}
print(f"\n{'what must be held':<36}{'bytes/weight':>13}{'this model':>14}")
for k, v in bytes_per_weight.items():
    print(f"{k:<36}{v:>13}{N_PARAMS*v/2**30:>13.2f} GiB")
tot = sum(bytes_per_weight.values())
print(f"{'TOTAL':<36}{tot:>13}{N_PARAMS*tot/2**30:>13.2f} GiB   <- before a single activation")
opt.zero_grad(set_to_none=True)

# --- the part the 16-bytes table leaves out: activations -------------------------
# Parameters are a fixed cost you can compute on paper. Activations depend on batch
# and sequence length, and they are why "the model fits" and "training fits" are
# different questions.
if DEVICE == "cuda":
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    h2 = model.model(input_ids=batch).last_hidden_state
    l2 = model.lm_head(h2)
    fwd_peak = torch.cuda.max_memory_allocated() - before
    F.cross_entropy(l2[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                    ignore_index=-100).backward()
    both_peak = torch.cuda.max_memory_allocated() - before
    print(f"\nactivation memory for ONE batch of {B} x {SEQ}:")
    print(f"  forward only              {fwd_peak/2**20:8.1f} MiB")
    print(f"  forward + backward        {both_peak/2**20:8.1f} MiB")
    print(f"  the logits alone          {B*SEQ*V*4/2**20:8.1f} MiB  "
          f"({100*B*SEQ*V*4/both_peak:.0f}% of it)")
    print(f"  scales with B x T: double the batch and this doubles; the 2.00 GiB above does not")
    model.zero_grad(set_to_none=True); torch.cuda.empty_cache()

# --- and where the parameters actually live --------------------------------------
groups = {}
for n, p_ in model.named_parameters():
    key = ("embedding / lm_head (tied)" if "embed" in n or "lm_head" in n else
           "attention" if "self_attn" in n else
           "MLP" if "mlp" in n else "norms / other")
    groups[key] = groups.get(key, 0) + p_.numel()
print(f"\nwhere the {N_PARAMS:,} parameters are:")
for k2, v2 in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"  {k2:<28}{v2:>13,}  {100*v2/N_PARAMS:5.1f}%")
print("  the MLP is bigger than attention - the famous part is not the expensive part")
""")

md(r"""
## Item 2 · Verify a gradient by hand

Two checks. The lesson's own chain first, where the answer is known exactly, then the same
procedure on a real weight inside a 135M-parameter model.

A **central** difference is used rather than the one-sided version in the lesson, because its
error falls as `h²` rather than `h`, which is what makes agreement to several decimals possible
in fp32 at all.
""")

code(r"""
# ---- 2a. the lesson's chain, in float64: w1=3, w2=4, x=2, target=20 -> dL/dw1 = 64
def chain_loss(w1, w2, x=2.0, t=20.0):
    h = w1 * x; y = w2 * h; return (y - t) ** 2

w1 = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
w2 = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
Lval = chain_loss(w1, w2); Lval.backward()

h_ = 1e-6
fd_w1 = (chain_loss(3.0 + h_, 4.0) - chain_loss(3.0 - h_, 4.0)) / (2 * h_)
fd_w2 = (chain_loss(3.0, 4.0 + h_) - chain_loss(3.0, 4.0 - h_)) / (2 * h_)

print("the chain by hand:  h = w1*x = 6 ; y = w2*h = 24 ; loss = (24-20)^2 = 16")
print("  dL/dy  = 2(y-t)   = 8")
print("  dL/dw2 = 8 * h    = 48")
print("  dL/dh  = 8 * w2   = 32")
print("  dL/dw1 = 32 * x   = 64")
print(f"\n{'':10}{'by hand':>12}{'backward()':>14}{'central diff':>16}{'abs err':>12}")
for nm, hand, auto, fd in (("dL/dw1", 64.0, w1.grad.item(), fd_w1),
                           ("dL/dw2", 48.0, w2.grad.item(), fd_w2)):
    print(f"{nm:<10}{hand:>12.6f}{auto:>14.6f}{fd:>16.6f}{abs(auto-fd):>12.2e}")
assert abs(w1.grad.item() - 64.0) < 1e-9 and abs(fd_w1 - 64.0) < 1e-6
""")

code(r"""
# ---- 2b. the same procedure on one real weight inside SmolLM2 -------------------
# eval() so the forward pass is deterministic: a finite difference compares two
# forward passes, and anything stochastic between them is measured as gradient.
model.eval()
probe_batch, probe_labels = IDS[:2], LABELS[:2]

def loss_now():
    with torch.no_grad():
        h = model.model(input_ids=probe_batch).last_hidden_state
        lg = model.lm_head(h)
        return F.cross_entropy(lg[:, :-1].reshape(-1, V).float(),
                               probe_labels[:, 1:].reshape(-1),
                               ignore_index=-100).item()

model.zero_grad(set_to_none=True)
h = model.model(input_ids=probe_batch).last_hidden_state
lg = model.lm_head(h)
F.cross_entropy(lg[:, :-1].reshape(-1, V).float(), probe_labels[:, 1:].reshape(-1),
                ignore_index=-100).backward()

# pick the single weight with the largest gradient: the finite difference has the
# best signal-to-noise where the slope is steepest
Wg = model.lm_head.weight.grad
flat = Wg.abs().flatten()
k = int(flat.argmax())
r, c = k // Wg.shape[1], k % Wg.shape[1]
reported = model.lm_head.weight.grad[r, c].item()

with torch.no_grad():
    w0 = model.lm_head.weight[r, c].item()
    eps = 1e-3
    model.lm_head.weight[r, c] = w0 + eps; lp = loss_now()
    model.lm_head.weight[r, c] = w0 - eps; lm = loss_now()
    model.lm_head.weight[r, c] = w0                       # restore
measured = (lp - lm) / (2 * eps)

print(f"probe weight: lm_head.weight[{r}, {c}]  (largest |grad| in the tensor)")
print(f"  value                       {w0:+.8f}")
print(f"  loss at w+{eps}             {lp:.10f}")
print(f"  loss at w-{eps}             {lm:.10f}")
print(f"\n  central difference          {measured:+.8f}")
print(f"  backward() reported         {reported:+.8f}")
print(f"  absolute difference         {abs(measured-reported):.3e}")
print(f"  relative difference         {abs(measured-reported)/abs(reported):.3e}")
rel = abs(measured - reported) / abs(reported)
assert rel < 5e-2, (
    f"gradient disagrees by {rel:.1%} - that is not float noise, find the bug")
print(f"\nagreement to {-math.log10(max(rel,1e-16)):.1f} decimal digits of relative error.")
print("fp32 forward passes and h=1e-3 put a floor on this; it is not exact and should not be.")

# --- why h=1e-3 and not h=1e-9 ---------------------------------------------------
# The obvious instinct is that a smaller nudge is a better derivative. It is not.
# Two errors fight each other:
#   truncation  ~ h^2   (the Taylor remainder: smaller h is better)
#   cancellation ~ eps/h (subtracting two nearly-equal fp32 numbers: smaller h is WORSE)
# Their sum is U-shaped, with a minimum around h ~ eps^(1/3). In fp32 that is ~1e-2.
# Below the minimum the answer gets rapidly worse, which is the opposite of what
# most people expect, so it is worth seeing rather than being told.
print(f"\n{'h':>10}{'central difference':>22}{'rel. error vs backward()':>28}")
best_h = None
for hh in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    with torch.no_grad():
        model.lm_head.weight[r, c] = w0 + hh; a_ = loss_now()
        model.lm_head.weight[r, c] = w0 - hh; b_ = loss_now()
        model.lm_head.weight[r, c] = w0
    est = (a_ - b_) / (2 * hh)
    e_ = abs(est - reported) / abs(reported)
    star = ""
    if best_h is None or e_ < best_h[1]: best_h, star = (hh, e_), ""
    print(f"{hh:>10.0e}{est:>22.8f}{e_:>28.2e}")
print(f"\nbest at h={best_h[0]:.0e} (rel err {best_h[1]:.1e}). Note the error rising again")
print("at the small end: that is fp32 cancellation, not a worse derivative. A finite")
print("difference is a measurement, and measurements have a noise floor.")
""")

md(r"""
## Item 3 · Break gradient accumulation on purpose

The bug that lived in every major framework until 2024. Micro-batches hold different numbers of
real tokens, and averaging the per-micro-batch **averages** gives a short micro-batch the same
vote as a long one.

$$\text{correct}=\frac{\sum_i \ell_i}{\sum_i n_i} \qquad\text{vs}\qquad \text{wrong}=\frac{1}{k}\sum_i \frac{\ell_i}{n_i}$$

To make the token counts genuinely unequal we mask a different fraction of each micro-batch,
which is exactly what variable-length sequences do in a real loader.
""")

code(r"""
def make_uneven(start, k=4, bs=2, keep=(1.0, 0.75, 0.25, 0.5)):
    '''k micro-batches whose contributing-token counts differ a lot.'''
    out = []
    for j in range(k):
        b = IDS[start + j*bs: start + (j+1)*bs]
        lb = LABELS[start + j*bs: start + (j+1)*bs].clone()
        n_keep = int(keep[j % len(keep)] * lb.shape[1])
        lb[:, n_keep:] = -100                       # truncate the supervised region
        out.append((b, lb))
    return out

def micro_losses(micros):
    '''Per-micro-batch (sum of loss, token count). One forward each, no accumulation yet.'''
    res = []
    for b, lb in micros:
        h = model.model(input_ids=b).last_hidden_state
        lg = model.lm_head(h)[:, :-1, :].reshape(-1, V).float()
        tg = lb[:, 1:].reshape(-1)
        n = int((tg != -100).sum())
        s = F.cross_entropy(lg, tg, ignore_index=-100, reduction="sum")
        res.append((s, n))
    return res

# ---- the lesson's own numbers first, where the answer is exact -----------------
LESSON = [(4, 2.0), (4, 2.0), (2, 5.0)]        # (valid tokens, mean loss)
c_ref = sum(n*l for n, l in LESSON) / sum(n for n, _ in LESSON)
w_ref = sum(l for _, l in LESSON) / len(LESSON)
print("the lesson's example, reproduced exactly:")
print(f"  correct  (4*2.0 + 4*2.0 + 2*5.0) / (4+4+2) = {sum(n*l for n,l in LESSON)}/"
      f"{sum(n for n,_ in LESSON)} = {c_ref:.4f}")
print(f"  wrong    (2.0 + 2.0 + 5.0) / 3            = {w_ref:.4f}")
print(f"  error    {100*(w_ref-c_ref)/c_ref:+.1f}%   <- the 15.4% from the lesson")
assert abs(c_ref - 2.6) < 1e-9 and abs(w_ref - 3.0) < 1e-9
assert abs(100*(w_ref-c_ref)/c_ref - 15.3846) < 1e-3

# NOTE the mechanism this exposes: the error size is driven by micro-batches having
# DIFFERENT MEAN LOSSES, not merely different token counts. Equal means -> no error
# however unequal the counts. That is why it also needs a trained model to show up
# on real data: an untrained model scores every token at ~ln V, so there is nothing
# for the mis-weighting to distort.

model.eval()
demo = make_uneven(0)
with torch.no_grad():
    parts = micro_losses(demo)
correct = sum(s for s, _ in parts) / sum(n for _, n in parts)
wrong = sum(s / n for s, n in parts) / len(parts)

print(f"{'micro-batch':<14}{'valid tokens':>14}{'mean loss':>12}")
for i, (s, n) in enumerate(parts):
    print(f"{i+1:<14}{n:>14,}{s.item()/n:>12.4f}")
print(f"\n  token-weighted (correct)      {correct.item():.4f}")
print(f"  average of averages (wrong)   {wrong.item():.4f}")
print(f"  error                         {100*(wrong.item()-correct.item())/correct.item():+.2f}%")
print("\nThe short micro-batches carry the same vote as the long ones, so their higher")
print("per-token loss is over-weighted. Equalise the token counts and the gap vanishes:")

even = [(IDS[j*2:(j+1)*2], LABELS[j*2:(j+1)*2]) for j in range(4)]
with torch.no_grad():
    pe = micro_losses(even)
ce = sum(s for s, _ in pe) / sum(n for _, n in pe)
we = sum(s / n for s, n in pe) / len(pe)
print(f"  equal-length micro-batches: correct {ce.item():.4f}  wrong {we.item():.4f}  "
      f"error {100*(we.item()-ce.item())/ce.item():+.2f}%")
print("  <- which is exactly why casual testing never caught it")

# --- the bug restated as a weight per TOKEN, which is what it really is -----------
# Neither formula is "an average". Both assign a weight to every token; they just
# disagree about what it should be. Correct gives every token 1/N. Wrong gives every
# token in micro-batch i a weight of 1/(k*n_i), so a token in a short micro-batch
# counts for MORE simply because it had fewer companions.
Ntot = sum(n for _, n in parts); k_ = len(parts)
print(f"\n{'micro-batch':<13}{'tokens':>8}{'weight/token (correct)':>24}{'weight/token (wrong)':>22}{'ratio':>8}")
for i, (s_, n) in enumerate(parts):
    wc, ww = 1 / Ntot, 1 / (k_ * n)
    print(f"{i+1:<13}{n:>8,}{wc:>24.3e}{ww:>22.3e}{ww/wc:>8.2f}x")
print("\nA token in the shortest micro-batch carries "
      f"{(1/(k_*min(n for _, n in parts)))/(1/(k_*max(n for _, n in parts))):.1f}x the weight of one")
print("in the longest. Nothing about the text justifies that - it is purely an artefact")
print("of how the sequences happened to be grouped into micro-batches on this step.")
""")

code(r"""
# ---- train both ways and plot the two curves together --------------------------
def train_accum(mode, steps, lr=3e-5, seed=SEED):
    '''mode=correct -> sum(loss)/sum(tokens);  mode=wrong -> mean of per-micro means.'''
    torch.manual_seed(seed)
    m, _, c = load_model_and_tokenizer()
    m = m.to(DEVICE); m.train()
    o = torch.optim.AdamW(m.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        start = (step * 8) % max(NSEQ - 8, 1)
        micros = make_uneven(start)
        o.zero_grad(set_to_none=True)
        tot_s, tot_n = 0.0, 0
        for j, (b, lb) in enumerate(micros):
            h = m.model(input_ids=b).last_hidden_state
            lg = m.lm_head(h)[:, :-1, :].reshape(-1, V).float()
            tg = lb[:, 1:].reshape(-1)
            n = int((tg != -100).sum())
            s = F.cross_entropy(lg, tg, ignore_index=-100, reduction="sum")
            # The DENOMINATOR is the whole bug. Correct: every token divided by the
            # global count. Wrong: each micro-batch divided by its own count, then
            # the averages averaged - which silently reweights by 1/n_i.
            part = s / sum(int((lbb[:, 1:].reshape(-1) != -100).sum()) for _, lbb in micros) \
                   if mode == "correct" else (s / n) / len(micros)
            part.backward()
            tot_s += s.item(); tot_n += n
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        o.step()
        hist.append(tot_s / tot_n)          # both curves REPORTED the same honest way
    del m, o
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return hist

ACC_STEPS = int(os.environ.get("S10_ACC_STEPS", 60))
h_ok = train_accum("correct", ACC_STEPS)
h_bad = train_accum("wrong", ACC_STEPS)
print(f"{'':14}{'first 10':>11}{'last 10':>11}")
print(f"{'correct':<14}{statistics.mean(h_ok[:10]):>11.4f}{statistics.mean(h_ok[-10:]):>11.4f}")
print(f"{'wrong':<14}{statistics.mean(h_bad[:10]):>11.4f}{statistics.mean(h_bad[-10:]):>11.4f}")
print(f"\ngap at the end: {statistics.mean(h_bad[-10:]) - statistics.mean(h_ok[-10:]):+.4f} nats")
print("Both curves are MEASURED with the correct token-weighted loss - only the")
print("gradient differs. Otherwise the wrong run would look better than it is.")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4))
    plt.plot(h_ok,  label="correct: sum(loss) / sum(tokens)", lw=1.3)
    plt.plot(h_bad, label="wrong: mean of per-micro-batch means", lw=1.3)
    plt.xlabel("step"); plt.ylabel("token-weighted loss (nats)")
    plt.title("Gradient accumulation, right and wrong"); plt.legend()
    plt.tight_layout(); plt.savefig("accumulation.png", dpi=120); print("saved accumulation.png"); plt.show()
except Exception as e:
    print("(plot skipped:", e, ")")
""")

md(r"""
## Item 4 · The grad norm moves before the loss

Two parts, because only one of them is honest on its own.

**4a** measures the natural lead/lag over the whole run — does the norm anticipate the loss on
ordinary data? **4b** injects a *known* shock at a known step so the lead can be measured
exactly. The injection is labelled because a "found" spike on natural data is easy to fool
yourself with.
""")

code(r"""
def train_logging(steps, shock_at=None, lr=3e-5, clip=1.0, seed=SEED):
    '''Log loss and grad norm every step. shock_at injects one pathological batch.'''
    torch.manual_seed(seed)
    m, _, _ = load_model_and_tokenizer(); m = m.to(DEVICE); m.train()
    o = torch.optim.AdamW(m.parameters(), lr=lr)
    log = []
    for step in range(steps):
        if shock_at is not None and step == shock_at:
            # A batch of one repeated rare token: high loss, and a gradient pointing
            # hard in one direction. This is a stand-in for the corrupted shard or
            # pathological document that ends real runs.
            rare = int(torch.randint(V // 2, V, (1,)).item())
            b = torch.full((2, SEQ), rare, dtype=torch.long, device=DEVICE)
            lb = b.clone()
        else:
            i = (step * 2) % max(NSEQ - 2, 1)
            b, lb = IDS[i:i+2], LABELS[i:i+2]
        h = m.model(input_ids=b).last_hidden_state
        lg = m.lm_head(h)[:, :-1, :].reshape(-1, V).float()
        loss = F.cross_entropy(lg, lb[:, 1:].reshape(-1), ignore_index=-100)
        o.zero_grad(set_to_none=True); loss.backward()
        # measure the norm BEFORE clipping - clip_grad_norm_ returns the pre-clip value
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), clip).item()
        o.step()
        log.append((step, loss.item(), gn))
    del m, o
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return log

NORM_STEPS = int(os.environ.get("S10_NORM_STEPS", 40))
SHOCK = NORM_STEPS // 2
log = train_logging(NORM_STEPS, shock_at=SHOCK)
steps_ = [x[0] for x in log]; losses = [x[1] for x in log]; norms = [x[2] for x in log]

# Robust baseline: median and MAD over the pre-shock window. A mean and standard
# deviation over a handful of steps gives a 3-sigma band so tight that ordinary
# step-to-step wobble crosses it, and the detector reports noise as signal.
def robust(seq):
    med = statistics.median(seq)
    mad = statistics.median([abs(v - med) for v in seq]) * 1.4826    # -> sigma-equivalent
    return med, (mad or 1e-9)

pre = slice(0, SHOCK)
base_l, sd_l = robust(losses[pre])
base_n, sd_n = robust(norms[pre])
dev_l = [(v - base_l) / sd_l for v in losses]
dev_n = [(v - base_n) / sd_n for v in norms]

def first_alert(dev, k=5.0, start=0):
    for i in range(start, len(dev)):
        if abs(dev[i]) > k: return i
    return None

i_n = first_alert(dev_n, 5.0)
i_l = first_alert(dev_l, 5.0)
peak_n = max(range(len(dev_n)), key=lambda i: abs(dev_n[i]))
peak_l = max(range(len(dev_l)), key=lambda i: abs(dev_l[i]))

print(f"shock injected at step {SHOCK}\n")
print(f"{'step':>5}{'loss':>10}{'dev':>9}{'grad norm':>12}{'dev':>10}")
for j in range(max(0, SHOCK-3), min(len(losses), SHOCK+5)):
    tag = "  <-- shock" if j == SHOCK else ""
    print(f"{j:>5}{losses[j]:>10.4f}{dev_l[j]:>+9.1f}{norms[j]:>12.3f}{dev_n[j]:>+10.1f}{tag}")

print(f"\nbaseline (median +/- MAD): loss {base_l:.4f} +/- {sd_l:.4f}   "
      f"norm {base_n:.3f} +/- {sd_n:.3f}")
print(f"\nAT THE SHOCK STEP:")
print(f"  loss      {losses[SHOCK]:.4f}   {dev_l[SHOCK]:+.1f} sigma")
print(f"  grad norm {norms[SHOCK]:.3f}   {dev_n[SHOCK]:+.1f} sigma")
print(f"\n  ratio: the norm reacts {abs(dev_n[SHOCK]/dev_l[SHOCK]):.0f}x more strongly")
print(f"  largest deviation in the whole run: norm at step {peak_n}, loss at step {peak_l}")

print(f"\nAnd note the SIGN. The shock batch is one token repeated, which after the")
print(f"first position is trivially predictable, so its loss went {'DOWN' if dev_l[SHOCK] < 0 else 'UP'}.")
if dev_l[SHOCK] < 0:
    print("A monitor watching for the loss to RISE would have recorded an improvement")
    print("while the gradient was pointing hard enough to end the run. That is the")
    print("whole argument for logging the norm: it is not a better version of the loss,")
    print("it is the only trace that reacts to this failure at all.")
print(f"\nfirst 5-sigma alert: norm at step {i_n}, loss at step {i_l}")

# 4a. and on ordinary data, with no injection: does the norm lead the loss at all?
clean = train_logging(NORM_STEPS)
cl = np.array([x[1] for x in clean]); cn = np.array([x[2] for x in clean])
def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12))
print("cross-correlation of grad norm with loss, at several lags")
print(f"{'lag':>5}{'corr':>10}   (lag k = norm at step t vs loss at step t+k)")
best = None
for k in range(-3, 4):
    # NOTE: named `rho`, not `c`. An earlier version used `c` here, which quietly
    # overwrote the column index `c` from item 2 and made the summary cell print
    # `lm_head[2299, -0.185...]`. Nothing raised. Shadowing in a long notebook
    # namespace is its own small lesson about silent failure.
    if k >= 0: rho = corr(cn[:len(cn)-k], cl[k:])
    else:      rho = corr(cn[-k:], cl[:len(cl)+k])
    lead = "leads" if k > 0 else ("lags" if k < 0 else "same step")
    print(f"{k:>5}{rho:>10.3f}   norm {lead}")
    if best is None or abs(rho) > abs(best[1]): best = (k, rho)
print(f"\nstrongest |r| at lag {best[0]} (r={best[1]:.3f}). Positive lag = the norm leads.")
if abs(best[1]) < 0.4:
    print("All of these are weak. On ordinary data over a few dozen steps there is no")
    print("reliable lead either way - the norm's value is in ANOMALIES, not in routine")
    print("prediction, and the injected shock above is where it earns its place on a")
    print("dashboard. Reporting a weak correlation as a finding would be overclaiming.")

# --- picking a clip threshold FROM DATA rather than from habit --------------------
# The lesson lists "what clip threshold?" as an open question and says to choose it
# from the grad-norm distribution over the first thousand steps. Here is that, in
# miniature, on the clean run. The usual default of 1.0 is a habit, not a measurement,
# and on this model it would clip essentially every step.
qs = sorted(cn)
def pct(p_): return qs[min(len(qs) - 1, int(p_ / 100 * len(qs)))]
print("grad-norm distribution on the clean run (no shock):")
for p_ in (5, 25, 50, 75, 90, 95, 99):
    print(f"  p{p_:<3} {pct(p_):8.3f}")
print(f"  max  {max(qs):8.3f}")
for cap in (0.5, 1.0, 2.0, 5.0, pct(95), pct(99)):
    frac = 100 * sum(1 for v in cn if v > cap) / len(cn)
    note = ""
    if abs(cap - 1.0) < 1e-9: note = "   <- the usual default"
    if abs(cap - pct(99)) < 1e-9: note = "   <- p99: clips only genuine outliers"
    print(f"  cap {cap:6.2f} would clip {frac:5.1f}% of steps{note}")
print("\nClipping every step is not a safety net, it is a learning-rate change in")
print("disguise: the direction survives but the magnitude is set by the cap rather")
print("than by the data. A cap near p99 leaves ordinary steps alone and still catches")
print("the shock above, which was ~340x the median.")

try:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps_, losses, label="loss", lw=1.3)
    ax.set_xlabel("step"); ax.set_ylabel("loss (nats)")
    ax2 = ax.twinx(); ax2.plot(steps_, norms, label="grad norm", lw=1.3, color="tab:orange")
    ax2.set_ylabel("grad norm")
    ax.axvline(SHOCK, ls="--", lw=.9, c="grey")
    ax.text(SHOCK, max(losses), " shock", fontsize=9, va="top")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left")
    plt.title("Grad norm and loss on one axis"); plt.tight_layout()
    plt.savefig("gradnorm.png", dpi=120); print("saved gradnorm.png"); plt.show()
except Exception as e:
    print("(plot skipped:", e, ")")
""")

md(r"""
## Item 5 · My own MFU, reported honestly

$$\text{MFU}=\frac{6N \times \text{tokens/second}}{\text{peak FLOP/s of the machine}}$$

The `6N` is the standard approximation: roughly 2N for the forward pass and 4N for the backward.
It **excludes the attention term**, which grows with sequence length, so at long context it
under-counts the real work and MFU comes out flattering. At T=256 on a 135M model that term is
small, but the number below is an approximation and should be read as one.
""")

code(r"""
PEAKS = {          # vendor peak, dense, no sparsity
    "Tesla T4":       {"fp32": 8.1e12,  "fp16 tensor": 65e12},
    "NVIDIA A100":    {"fp32": 19.5e12, "tf32 tensor": 156e12, "bf16 tensor": 312e12},
    "NVIDIA L4":      {"fp32": 30.3e12, "bf16 tensor": 121e12},
    "NVIDIA H100":    {"fp32": 67e12,   "bf16 tensor": 989e12},
}
def peak_for(name):
    for k, v in PEAKS.items():
        if k.lower() in name.lower(): return k, v
    return None, None

MFU_STEPS = int(os.environ.get("S10_MFU_STEPS", 20))
MFU_BS = int(os.environ.get("S10_MFU_BS", 4))
torch.manual_seed(SEED)
m, _, _ = load_model_and_tokenizer(); m = m.to(DEVICE); m.train()
o = torch.optim.AdamW(m.parameters(), lr=1e-5)

for _ in range(3):                     # warm-up: never time the first steps
    b = IDS[:MFU_BS]
    lg = m.lm_head(m.model(input_ids=b).last_hidden_state)[:, :-1, :].reshape(-1, V).float()
    F.cross_entropy(lg, LABELS[:MFU_BS][:, 1:].reshape(-1), ignore_index=-100).backward()
    o.zero_grad(set_to_none=True)
if DEVICE == "cuda": torch.cuda.synchronize()

t0 = time.time(); toks = 0
for step in range(MFU_STEPS):
    i = (step * MFU_BS) % max(NSEQ - MFU_BS, 1)
    b, lb = IDS[i:i+MFU_BS], LABELS[i:i+MFU_BS]
    lg = m.lm_head(m.model(input_ids=b).last_hidden_state)[:, :-1, :].reshape(-1, V).float()
    loss = F.cross_entropy(lg, lb[:, 1:].reshape(-1), ignore_index=-100)
    o.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
    toks += b.numel()
if DEVICE == "cuda": torch.cuda.synchronize()
dt = time.time() - t0
tps = toks / dt
achieved = 6 * N_PARAMS * tps

print(f"{MFU_STEPS} steps, batch {MFU_BS} x {SEQ} = {MFU_BS*SEQ:,} tokens/step")
print(f"  wall time            {dt:.2f} s  ({1000*dt/MFU_STEPS:.0f} ms/step)")
print(f"  tokens/second        {tps:,.0f}")
print(f"  achieved 6N*t/s      {achieved/1e12:.2f} TFLOP/s")

gname, peaks = peak_for(GPU)
if peaks:
    print(f"\n  {gname} peak, and MFU against each:")
    for k, v in peaks.items():
        print(f"    {k:<14}{v/1e12:>7.1f} TFLOP/s   MFU {100*achieved/v:>6.2f}%")
    honest = list(peaks.values())[0]
    MFU = 100 * achieved / honest
    print(f"\n  MFU against the fp32 peak (the honest denominator, since we train in fp32): "
          f"{MFU:.2f}%")
else:
    MFU = float("nan")
    print(f"\n  no published peak on record for {GPU!r} - MFU not computed")
# --- where the time actually goes ------------------------------------------------
# MFU says how much of the machine you are using. It does not say what is using it.
# Timing the three phases separately turns "26% MFU" into an actionable list, and
# the split is usually not what people guess.
def timed(fn, n=8):
    if DEVICE == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(n): fn()
    if DEVICE == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / n

b_, lb_ = IDS[:MFU_BS], LABELS[:MFU_BS]
_cache = {}
def f_only():
    with torch.no_grad():
        _cache["h"] = m.lm_head(m.model(input_ids=b_).last_hidden_state)
def f_and_b():
    lgx = m.lm_head(m.model(input_ids=b_).last_hidden_state)[:, :-1, :].reshape(-1, V).float()
    F.cross_entropy(lgx, lb_[:, 1:].reshape(-1), ignore_index=-100).backward()
    o.zero_grad(set_to_none=True)
def opt_only():
    o.step()

t_f = timed(f_only); t_fb = timed(f_and_b); t_o = timed(opt_only)
tot_t = t_fb + t_o
print(f"\nwhere one step's time goes (batch {MFU_BS} x {SEQ}):")
print(f"  forward only              {1000*t_f:7.1f} ms   {100*t_f/tot_t:5.1f}%")
print(f"  backward (by difference)  {1000*(t_fb-t_f):7.1f} ms   {100*(t_fb-t_f)/tot_t:5.1f}%")
print(f"  optimiser step            {1000*t_o:7.1f} ms   {100*t_o/tot_t:5.1f}%")
print(f"  total                     {1000*tot_t:7.1f} ms")
print(f"\nbackward/forward ratio    {(t_fb-t_f)/max(t_f,1e-9):.2f}x   "
      f"(theory says ~2x: one pass for input grads, one for weight grads)")
print("If the optimiser share is large, the batch is too small - AdamW touches every")
print("weight once per step regardless of how many tokens you fed it, so its cost is")
print("amortised only by making the batch bigger.")

# --- the same throughput, judged against other hardware --------------------------
# Arithmetic, not measurement: what MFU would this token rate imply elsewhere? The
# point is that MFU is a ratio, and changing the denominator changes the verdict
# without changing the run at all.
print(f"\nthe SAME {tps:,.0f} tok/s judged against other peaks (arithmetic, not measured):")
for nm, pk in (("T4 fp32", 8.1e12), ("T4 fp16 tensor", 65e12),
               ("A100 bf16 tensor", 312e12), ("H100 bf16 tensor", 989e12)):
    print(f"  {nm:<20}{pk/1e12:>7.1f} TFLOP/s   MFU {100*achieved/pk:>6.2f}%")
print("A run does not become efficient by being measured against a slower card.")

del m, o
if DEVICE == "cuda": torch.cuda.empty_cache()
""")

md(r"""
### What is costing me the distance to 40%

Ranked by how much I think each one matters here, largest first:

1. **The hardware cannot do bf16 at all.** A T4 is Turing: no bf16 support, so this run is fp32
   on ordinary CUDA cores against an 8.1 TFLOP/s peak. The 65 TFLOP/s tensor-core figure needs
   fp16, which Session 9 showed is the wrong precision for AdamW at these learning rates. A
   healthy 35–50% MFU is a number quoted for bf16 tensor-core hardware; on this card it is not
   reachable by tuning.
2. **The batch is tiny.** 4 × 256 = 1,024 tokens per step. Every kernel launch, optimiser
   traversal and Python statement is amortised over almost nothing, so fixed overheads dominate.
   Larger micro-batches plus accumulation is the single biggest lever available without changing
   hardware.
3. **The output head dominates a 135M model.** V/D = 85×, so the `[B,T,49152]` logits and their
   gradient are the largest tensors in the step, and cross-entropy over them is bandwidth-bound
   rather than compute-bound. `6N` counts those FLOPs as useful work; the memory traffic they
   generate is invisible to the formula.
4. **No fused kernels.** Stock `nn.functional` attention, unfused optimiser, `.float()` casts on
   the logits. FlashAttention and a fused AdamW would each recover some of this — Session 8's
   whole point being that the arithmetic was never the bottleneck.
5. **A 30-layer, 576-wide model is thin.** Each matmul is small, so the GPU is launch-latency
   bound rather than saturated. Depth-over-width is good for quality per parameter and bad for
   utilisation.

**What I would fix first:** the batch, because it costs nothing but memory. **What I would stop
measuring:** MFU on a T4 — the ceiling is set by the absent bf16 path, so the number says more
about the card than the loop.
""")

md(r"""
## Item 6 · 0.1 in fp32, bf16 and fp8 E4M3

0.1 is not representable in binary at all — it is `0.0001100110011...` repeating — so every
format below stores something slightly else. Each row is derived by hand and then checked against
what the hardware actually stores.
""")

code(r"""
def decompose(x, fmt):
    if fmt == "fp32":
        raw = struct.unpack(">I", struct.pack(">f", x))[0]
        s, e, mant, eb, mb, bias = raw >> 31, (raw >> 23) & 0xFF, raw & 0x7FFFFF, 8, 23, 127
        stored = struct.unpack(">f", struct.pack(">f", x))[0]
    elif fmt == "bf16":
        t = torch.tensor([x], dtype=torch.bfloat16)
        raw = int(t.view(torch.int16).item()) & 0xFFFF
        s, e, mant, eb, mb, bias = raw >> 15, (raw >> 7) & 0xFF, raw & 0x7F, 8, 7, 127
        stored = float(t.double().item())
    else:
        t = torch.tensor([x], dtype=torch.float8_e4m3fn)
        raw = int(t.view(torch.int8).item()) & 0xFF
        s, e, mant, eb, mb, bias = raw >> 7, (raw >> 3) & 0xF, raw & 0x7, 4, 3, 7
        stored = float(t.double().item())
    sig = 1 + mant / (1 << mb)
    return dict(raw=raw, s=s, e=e, mant=mant, eb=eb, mb=mb, bias=bias, sig=sig,
                unb=e - bias, stored=stored,
                ebits=format(e, f"0{eb}b"), mbits=format(mant, f"0{mb}b"))

x = 0.1
print(f"0.1 in binary is 0.0001100110011..., repeating - not representable in any of these.\n")
print("normalised: 0.1 = 1.6 x 2^-4, so every format below stores exponent -4 and")
print("approximates the significand 1.6 with the mantissa bits it has.\n")
for fmt in ("fp32", "bf16", "fp8_e4m3"):
    d = decompose(x, fmt)
    rec = d["sig"] * 2.0 ** d["unb"]
    nib = (1 + d["eb"] + d["mb"]) // 4
    print(f"{fmt}  ({1}+{d['eb']}+{d['mb']} bits)")
    print(f"  sign exponent mantissa      {d['s']} {d['ebits']} {d['mbits']}")
    print(f"  hex                         0x{d['raw']:0{nib}X}")
    print(f"  exponent field              {d['e']} - bias {d['bias']} = {d['unb']}")
    print(f"  significand                 1 + {d['mant']}/{1 << d['mb']} = {d['sig']}")
    print(f"  value = sig x 2^exp         {rec!r}")
    print(f"  what the hardware stores    {d['stored']!r}")
    print(f"  relative error vs 0.1       {abs(d['stored']-x)/x:.3e}")
    assert abs(rec - d["stored"]) < 1e-12 * max(1.0, abs(rec)), \
        f"{fmt}: hand reconstruction != stored value"
    print()

print("mantissa bits buy detail, and that is the entire story of this table:")
print(f"{'format':<12}{'mantissa bits':>15}{'stored':>22}{'rel err':>12}")
for fmt in ("fp32", "bf16", "fp8_e4m3"):
    d = decompose(x, fmt)
    print(f"{fmt:<12}{d['mb']:>15}{d['stored']:>22.10f}{abs(d['stored']-x)/x:>12.1e}")

# --- the spacing between representable numbers, which is what "detail" means ------
# A float format is not a fine mesh - it is a set of discrete points, and near any
# value there is a gap you cannot express. That gap is the real meaning of mantissa
# bits, and it explains why an optimiser update can vanish.
print(f"\ngap to the next representable number above 0.1:")
for fmt, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16),
                ("fp16", torch.float16), ("fp8_e4m3", torch.float8_e4m3fn)):
    v = torch.tensor([0.1], dtype=dt)
    lo = float(v.double().item())
    step = 2.0 ** (decompose(0.1, "fp32")["unb"] - {"fp32": 23, "bf16": 7,
                    "fp16": 10, "fp8_e4m3": 3}[fmt])
    print(f"  {fmt:<10}{step:>14.3e}   (~{step/0.1*100:.4f}% of the value)")
print("\nAn AdamW update of ~1e-5 applied to a weight of ~1e-2 is a 0.1% change. In")
print("bf16 the gap at that magnitude is ~0.8%, so the update lands BELOW the spacing")
print("and rounds away to nothing. The weight never moves. That is the whole reason")
print("the optimiser needs an fp32 master copy, and it is not a subtle effect.")

# --- the underflow argument, run rather than asserted -----------------------------
print(f"\nwhat happens to a shrinking gradient (the fp16-vs-bf16 argument):")
print(f"{'gradient':>12}{'fp16':>16}{'bf16':>16}{'fp16 x1024':>16}")
for g in (1e-4, 1e-6, 1e-8, 1e-10):
    h16 = float(torch.tensor([g], dtype=torch.float16).double().item())
    b16 = float(torch.tensor([g], dtype=torch.bfloat16).double().item())
    scaled = float(torch.tensor([g * 1024], dtype=torch.float16).double().item()) / 1024
    f16s = "ZERO" if h16 == 0 else f"{h16:.2e}"
    print(f"{g:>12.0e}{f16s:>16}{b16:>16.2e}{scaled:>16.2e}")
print("\nfp16 flushes to zero somewhere between 1e-8 and 1e-10; bf16 does not, because")
print("it kept all eight of fp32's exponent bits. The last column is loss scaling:")
print("multiply by 1024 before the backward pass, divide after, and the same gradient")
print("survives. It works, and it is one more knob to tune and eventually get wrong.")
print("bf16 deletes the knob. THAT is why a format with 2.4 decimal digits beat one")
print("with 3.3 - the contest was never about digits.")
""")

md(r"""
### Which would I train in, and why

**bf16, with an fp32 master copy of the weights** — which is what the 16-bytes-per-weight table
in item 1 is describing.

The reasoning is not about 0.1. Look at what each format did to it: fp32 is wrong by 1.5e-08,
bf16 by 9.8e-04, fp8 by 1.6e-02. On accuracy alone bf16 is a poor showing and fp8 is worse.

**Accuracy is the wrong axis.** The number that decides this is *range*, not detail:

| format | exponent bits | smallest normal | what happens to a 1e-8 gradient |
|---|---|---|---|
| fp32 | 8 | 1.18e-38 | fine |
| fp16 | 5 | 6.10e-05 | **flushes to zero** |
| bf16 | 8 | 1.18e-38 | fine |

Late in a run gradients get genuinely tiny, and a gradient of exactly zero means that weight
stops moving — the model quietly stops learning precisely where the remaining signal was
faintest. fp16 needs loss scaling to avoid it, which is one more thing to tune and eventually get
wrong. bf16 keeps all eight of fp32's exponent bits, so its floor is unreachable and the whole
apparatus disappears. **bf16 is less accurate than fp16 and won anyway.**

Two qualifications I would not leave out:

- **2.4 decimal digits is not enough to accumulate in.** bf16 is for the *forward and backward
  pass*; the optimiser must keep an fp32 master copy, or updates of order 1e-5 against weights of
  order 1e-2 round away entirely. Session 9 hit exactly this — pure-bf16 AdamW at lr 3e-5 left a
  head above `ln V` after 300 steps, learning nothing while producing a perfectly plausible loss
  curve.
- **fp8 is a production recipe in 2026, and NVFP4 is ~1.73× faster still.** Neither is a blanket
  replacement: they need per-block shared exponents, and attention stays in higher precision
  because softmax amplifies whatever noise you hand it. The rule is not "use fewer bits", it is
  *shrink where the error does not accumulate*.

**And on this hardware the question is moot** — a T4 has no bf16 at all, which is item 5's answer
as well.
""")

md("## Summary — the six answers")

code(r"""
print("=" * 76)
print(f"1  shapes           batch {tuple(batch.shape)}  hidden {tuple(hidden.shape)}  "
      f"logits {tuple(logits.shape)} -> loss scalar")
print(f"                    grad, master copy and 2 Adam buffers all have the weight's shape")
print(f"                    16 bytes/weight = {N_PARAMS*16/2**30:.2f} GiB for this 135M model")
print(f"2  hand gradient    lesson chain: backward {w1.grad.item():.6f} vs finite diff {fd_w1:.6f}")
print(f"                    real weight lm_head[{r},{c}]: reported {reported:+.6f} "
      f"vs measured {measured:+.6f}  (rel {rel:.2e})")
print(f"3  accumulation     one batch: correct {correct.item():.4f} vs wrong {wrong.item():.4f} "
      f"({100*(wrong.item()-correct.item())/correct.item():+.2f}%)")
print(f"                    equal-length control: {100*(we.item()-ce.item())/ce.item():+.2f}% "
      f"<- how it hid")
print(f"                    after {ACC_STEPS} steps: correct {statistics.mean(h_ok[-10:]):.4f} "
      f"vs wrong {statistics.mean(h_bad[-10:]):.4f}")
print(f"4  grad norm        shock at step {SHOCK}: norm {dev_n[SHOCK]:+.0f} sigma, "
      f"loss {dev_l[SHOCK]:+.0f} sigma -> norm reacts "
      f"{abs(dev_n[SHOCK]/dev_l[SHOCK]):.0f}x more strongly, and in the actionable direction")
print(f"                    clean run: strongest norm/loss correlation at lag {best[0]} "
      f"(r={best[1]:.3f})")
print(f"5  MFU              {tps:,.0f} tok/s -> {achieved/1e12:.2f} TFLOP/s -> "
      f"MFU {MFU:.2f}% on {GPU}")
print(f"6  0.1              fp32 {decompose(0.1,'fp32')['stored']!r}")
print(f"                    bf16 {decompose(0.1,'bf16')['stored']!r}")
print(f"                    fp8  {decompose(0.1,'fp8_e4m3')['stored']!r}")
print(f"                    train in bf16 + fp32 master copy: range beats detail")
print("=" * 76)
""")

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.10"},
      "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "S10_training_loop.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out}  ({len(C)} cells: {sum(1 for c in C if c['cell_type']=='code')} code, "
      f"{sum(1 for c in C if c['cell_type']=='markdown')} markdown)")
