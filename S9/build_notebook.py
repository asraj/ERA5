#!/usr/bin/env python3
"""Generate S9_loss_harness.ipynb.

The notebook is generated rather than hand-edited so that verify_local.py can
execute the *same* code cells offline and prove they run. See README.
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
C = []
def md(s):   C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)})
def code(s): C.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                       "source":s.strip("\n").splitlines(True)})

md(r"""
# Session 9 — Loss functions and output heads

**Stephen Raj Arokiasamy**

One notebook, one loss harness. The starting point is this, which is *nearly* right:

```python
hidden = model(tokens)
logits = output_head(hidden)
loss   = cross_entropy(logits[:, :-1].reshape(-1, vocab_size),
                       tokens[:, 1:].reshape(-1))
```

Everything below makes it correct and, more importantly, **observable** — because every
bug in this block is silent. A wrong shift does not raise. It just produces a beautiful
loss curve for a model that has learned to copy its input.

| | Part 1 |
|---|---|
| 1 | Every tensor shape, with what each dimension means |
| 2 | Shift verified on **token strings**, not ids |
| 3 | Padding masked, contributing-token count changes |
| 4 | Two documents packed, boundary masked, loss before/after |
| 5 | Perplexity of an untrained model vs vocabulary size |
| 6 | Tied vs untied head parameter counts |
| 7 | Peak memory: ordinary cross-entropy vs a chunked one |

**Part 2** adds a second head predicting token `t+2` and reports both losses.

**Model:** `HuggingFaceTB/SmolLM2-135M` — 135M params, vocab 49,152, d_model 576,
30 layers, GQA, **tied** embeddings. Small enough to train in Part 2 on a free T4;
real enough that the numbers mean something.

**Data:** the Session-4 cleaned OpenWebText shard (8-stage pipeline: normalise, format
discipline, quality, exact + MinHash dedup, language-ID, PII scrub).
""")

code(r"""
# Colab setup. Safe to re-run.
try:
    import torch, transformers            # noqa
except ImportError:
    !pip -q install torch transformers
import os, json, gzip, math, urllib.request, textwrap
import torch, torch.nn as nn, torch.nn.functional as F
import transformers
print("torch       ", torch.__version__)
print("transformers", transformers.__version__)

SEED = 9
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device      ", DEVICE, "|", torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# OFFLINE=1 swaps the hub download for a locally-constructed model of the same
# architecture. Used by verify_local.py so this notebook's code can be executed
# and checked without network access. Leave unset on Colab.
OFFLINE = os.environ.get("S9_OFFLINE") == "1"
""")

md("## 0 · Model, tokenizer and the cleaned Session-4 shard")

code(r"""
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

def load_model_and_tokenizer():
    '''Returns (model, tokenizer, config). model is the full CausalLM; we deliberately
    use model.model (the trunk) and model.lm_head (the output head) separately below,
    because the whole point of this session is that they are two different objects.'''
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    if OFFLINE:
        from transformers import LlamaConfig, LlamaForCausalLM
        from tokenizers import Tokenizer
        from transformers import PreTrainedTokenizerFast
        cfg = LlamaConfig(vocab_size=49152, hidden_size=576, intermediate_size=1536,
                          num_hidden_layers=int(os.environ.get("S9_LAYERS", 30)),
                          num_attention_heads=9, num_key_value_heads=3,
                          hidden_act="silu", max_position_embeddings=2048,
                          rms_norm_eps=1e-5, tie_word_embeddings=True, rope_theta=10000.0)
        model = LlamaForCausalLM(cfg)
        # honour S9_DTYPE so verify_local.py can reproduce Colab's bfloat16 load,
        # which is what surfaced the head-2 dtype bug in the first place
        model = model.to(getattr(torch, os.environ.get("S9_DTYPE", "float32")))
        tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer.from_file(
            os.environ["S9_TOKENIZER"]), unk_token="<unk>", eos_token="</s>", bos_token="<s>")
        return model, tok, cfg
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    # Pin fp32. transformers v5 defaults to the checkpoint's own dtype, which for
    # SmolLM2 is bfloat16 -- and then any nn.Linear we add later (head 2, Part 2) is
    # fp32 by default and the matmul raises "mat1 and mat2 have different dtype".
    # Pinning also keeps item 7's analytic fp32 logit sizes honest.
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    except TypeError:                     # transformers < 5 spells it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    return model, tok, cfg

model, tok, cfg = load_model_and_tokenizer()
model = model.to(DEVICE).eval()

V = cfg.vocab_size
D = cfg.hidden_size
print(f"vocab_size V        = {V:,}")
print(f"hidden_size D       = {D}")
print(f"layers              = {cfg.num_hidden_layers}")
print(f"tie_word_embeddings = {cfg.tie_word_embeddings}")
print(f"total parameters    = {sum(p.numel() for p in model.parameters()):,}")
MODEL_DTYPE = next(model.parameters()).dtype
print(f"parameter dtype     = {MODEL_DTYPE}")

# A padding token is required for item 3. SmolLM2 ships without one, so we borrow EOS
# and rely on the tokenizer's attention_mask -- not on `id == pad_id` -- to find padding.
# Those two are NOT the same test once pad_id == eos_id, and confusing them silently
# drops every real end-of-document token from the loss.
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
print(f"pad_token           = {tok.pad_token!r} (id {tok.pad_token_id}), "
      f"eos id {tok.eos_token_id} -> same id: {tok.pad_token_id == tok.eos_token_id}")
""")

code(r"""
# The Session-4 cleaned OpenWebText shard, with three sources tried in order so this
# cell cannot be the reason the notebook fails to run:
#   1. a local file (offline verification, or you uploaded it yourself)
#   2. the repo copy on GitHub raw
#   3. a public OpenWebText subset, cleaned HERE with the same Session-4 stages
# Source 3 exists so a grader with only this .ipynb still gets a runnable notebook.
import re, unicodedata, hashlib

SHARD_URL   = "https://raw.githubusercontent.com/asraj/ERA5/main/S9/data/s9_owt_clean.jsonl.gz"
SHARD_LOCAL = os.environ.get("S9_SHARD", "s9_owt_clean.jsonl.gz")

# --- Session-4 cleaning stages, inlined so source 3 is the same pipeline ----------
_ENT = [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]
_DEL = {ord(c): None for c in ("\u200b", "\ufeff", "\u202e", "\ufffd")}   # ZWJ/ZWNJ kept
_STOP = set("the a an of to and in is are was on for with as at by it this that from or".split())
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_IP    = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

def s4_normalize(t):
    t = unicodedata.normalize("NFC", t)
    for e, c in _ENT: t = t.replace(e, c)
    return re.sub(r"[ \t]+", " ", t.translate(_DEL)).strip()

def s4_quality_ok(t):
    w = t.split()
    if len(w) < 3: return False
    avg = sum(len(x) for x in w) / len(w)
    if avg > 20 or avg < 1.5: return False
    if sum(c in "|!@#$%^&*<>=~`" for c in t) / max(len(t), 1) > 0.18: return False
    return any(x.lower() in _STOP for x in w)

def s4_scrub(t):
    return _IP.sub("<IP>", _EMAIL.sub("<EMAIL>", t))

def build_shard_from_public_owt(n_docs=5000):
    from datasets import load_dataset
    ds = load_dataset("stas/openwebtext-10k", split="train")
    out, seen = [], set()
    for r in ds:
        t = s4_normalize(r["text"])
        if not (400 <= len(t) <= 12000): continue
        if not s4_quality_ok(t): continue
        h = hashlib.blake2b(t.encode(), digest_size=16).digest()
        if h in seen: continue
        seen.add(h); out.append(s4_scrub(t))
        if len(out) >= n_docs: break
    return out

DOCS, SOURCE = None, None
if os.path.exists(SHARD_LOCAL):
    DOCS = [json.loads(l)["text"] for l in gzip.open(SHARD_LOCAL, "rt", encoding="utf-8")]
    SOURCE = f"local file {SHARD_LOCAL}"
else:
    try:
        print("fetching the Session-4 cleaned shard from the repo ...")
        urllib.request.urlretrieve(SHARD_URL, SHARD_LOCAL)
        DOCS = [json.loads(l)["text"] for l in gzip.open(SHARD_LOCAL, "rt", encoding="utf-8")]
        SOURCE = "repo copy of the Session-4 cleaned shard"
    except Exception as e:
        print(f"repo copy unavailable ({type(e).__name__}); rebuilding from public "
              f"OpenWebText with the same Session-4 stages ...")
        DOCS = build_shard_from_public_owt()
        SOURCE = "stas/openwebtext-10k, cleaned in-notebook with the Session-4 stages"

print(f"source: {SOURCE}")
print(f"{len(DOCS):,} cleaned documents, "
      f"{sum(d.count(' ') + 1 for d in DOCS):,} whitespace words")
print("\nfirst 300 chars of doc 0:\n" + textwrap.fill(DOCS[0][:300], 96))
assert len(DOCS) > 500, "shard too small to train on"
""")

md(r"""
## Item 1 · Every tensor shape, and what each dimension is

The bug this catches: reshaping the wrong axis. `[B, T, V] -> [-1, V]` is safe;
`[B, T, V] -> [B, -1]` is not, and neither raises.
""")

code(r"""
B, T = 4, 128
batch  = tok([d for d in DOCS[:B]], return_tensors="pt", padding="max_length",
             truncation=True, max_length=T)
tokens = batch["input_ids"].to(DEVICE)
attn   = batch["attention_mask"].to(DEVICE)

with torch.no_grad():
    hidden = model.model(input_ids=tokens, attention_mask=attn).last_hidden_state
    logits = model.lm_head(hidden)

shift_logits  = logits[:, :-1, :]
shift_targets = tokens[:, 1:]
flat_logits   = shift_logits.reshape(-1, V)
flat_targets  = shift_targets.reshape(-1)

rows = [
 ("tokens",        tokens.shape,        "B = sequences in the batch; T = positions in each sequence"),
 ("attention_mask",attn.shape,          "B, T; 1 = real token, 0 = padding"),
 ("hidden",        hidden.shape,        "B, T, D — D = width of the residual stream, one vector per position"),
 ("logits",        logits.shape,        "B, T, V — V = one raw score per vocabulary entry, per position"),
 ("shift_logits",  shift_logits.shape,  "B, T-1, V — drop the LAST position: nothing follows it to predict"),
 ("shift_targets", shift_targets.shape, "B, T-1 — drop the FIRST token: nothing precedes it to predict from"),
 ("flat_logits",   flat_logits.shape,   "B*(T-1), V — every prediction in the batch, stacked"),
 ("flat_targets",  flat_targets.shape,  "B*(T-1) — one correct token id per prediction"),
]
w = max(len(r[0]) for r in rows)
for name, shape, meaning in rows:
    print(f"{name:<{w}}  {str(tuple(shape)):<20}  {meaning}")

print(f"\nlogits tensor is {logits.numel() / hidden.numel():.1f}x larger than the hidden "
      f"states that produced it  (V/D = {V}/{D} = {V/D:.1f})")
assert flat_logits.shape[0] == flat_targets.shape[0], "prediction/target count mismatch"
""")

md(r"""
## Item 2 · Verify the shift on token **strings**

Ids are unreadable, so an off-by-one hides in them. Read the two columns: every target
must be the word that literally follows the input in the text.

The last row of the control block is the tell — a **wrong** shift (no shift at all) puts
the same token in both columns, which is the model being handed the answer.
""")

code(r"""
def show_shift(seq_ids, n=12, offset=1, title=""):
    ins  = seq_ids[:-offset] if offset else seq_ids
    tgts = seq_ids[offset:]  if offset else seq_ids
    print(title)
    print(f"  {'pos':>4}  {'input id':>9}  {'input':<18}  {'target id':>9}  {'target':<18}")
    for i in range(min(n, len(tgts))):
        si = tok.decode([ins[i]]);  st = tok.decode([tgts[i]])
        print(f"  {i:>4}  {ins[i]:>9}  {si!r:<18}  {tgts[i]:>9}  {st!r:<18}")

sample = "The capital of India is New Delhi, and the capital of France is Paris."
ids = tok(sample, add_special_tokens=False)["input_ids"]
print("source text:", repr(sample), "\n")
show_shift(ids, 12, offset=1, title="CORRECT — shift by one: each target is the next token")
print()
show_shift(ids, 6,  offset=0, title="WRONG — no shift: target == input, the answer is in the question")
print("\nreconstruction check:", repr(tok.decode(ids)))
assert tok.decode(ids[1:]) != tok.decode(ids[:-1]), "shift produced identical sequences"
""")

md(r"""
## Item 3 · Mask the padding, and watch the contributing-token count change

Padding is trivially predictable, so training on it makes the loss look better than it is.
Two things must change together: the summed numerator **and** the denominator.

`ignore_index=-100` is how PyTorch is told to drop a position from both.
""")

code(r"""
def loss_and_count(logits_, targets_, ignore_index=-100):
    '''Returns (mean loss over contributing positions, number of contributing positions).'''
    ls  = logits_[:, :-1, :].reshape(-1, V)
    tg  = targets_[:, 1:].reshape(-1)
    n   = int((tg != ignore_index).sum())
    l   = F.cross_entropy(ls.float(), tg, ignore_index=ignore_index)
    return l.item(), n

# Deliberately uneven lengths so there IS padding.
texts = [DOCS[0][:600], DOCS[1][:150], DOCS[2][:900], DOCS[3][:80]]
enc   = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=T)
ids   = enc["input_ids"].to(DEVICE)
am    = enc["attention_mask"].to(DEVICE)

with torch.no_grad():
    lg = model.lm_head(model.model(input_ids=ids, attention_mask=am).last_hidden_state)

# Derive labels from attention_mask, NOT from `ids == pad_token_id`: pad_id == eos_id here,
# so the id test would also delete every genuine end-of-document token.
labels_masked   = ids.masked_fill(am == 0, -100)
loss_pad,  n_pad  = loss_and_count(lg, ids)             # padding counted
loss_mask, n_mask = loss_and_count(lg, labels_masked)   # padding excluded

print(f"per-sequence real token counts: {am.sum(1).tolist()}   (padded to T={ids.shape[1]})")
print(f"{'padding COUNTED':<22} loss {loss_pad:7.4f}   contributing tokens {n_pad:>6,}")
print(f"{'padding MASKED':<22} loss {loss_mask:7.4f}   contributing tokens {n_mask:>6,}")
print(f"{'change':<22}      {loss_mask - loss_pad:+7.4f}   {n_mask - n_pad:>+6,} "
      f"({100*(n_pad-n_mask)/n_pad:.1f}% of positions were padding)")
assert n_mask < n_pad, "masking did not reduce the contributing-token count"
""")

md(r"""
## Item 4 · Pack two documents, mask the boundary

Session 6 packed documents together to avoid wasting compute on padding. That creates one
position per join whose target belongs to a **different document**: the last token of A is
asked to predict the first token of B, and there is no relationship to learn.

To make the effect readable rather than a rounding error, we pack many documents and mask
every join, not just one.
""")

code(r"""
def pack(docs, seq_len, n_seqs, eos_id):
    '''Concatenate documents into fixed-length sequences.
    Returns ids [n_seqs, seq_len] and is_start [n_seqs, seq_len] marking, for each
    position, whether it is the FIRST token of a document.'''
    stream, starts = [], []
    for d in docs:
        piece = tok(d, add_special_tokens=False)["input_ids"] + [eos_id]
        starts += [1] + [0] * (len(piece) - 1)
        stream += piece
        if len(stream) >= seq_len * n_seqs: break
    stream, starts = stream[:seq_len*n_seqs], starts[:seq_len*n_seqs]
    ids = torch.tensor(stream).view(n_seqs, seq_len)
    st  = torch.tensor(starts).view(n_seqs, seq_len)
    return ids.to(DEVICE), st.to(DEVICE)

# ---- 4a. The literal case: exactly two documents in one sequence -----------------
docA = tok(DOCS[20][:400], add_special_tokens=False)["input_ids"]
docB = tok(DOCS[21][:400], add_special_tokens=False)["input_ids"]
pair = torch.tensor([docA + docB]).to(DEVICE)
join = len(docA)                       # index of docB's FIRST token = the offending target

print(f"doc A: {len(docA)} tokens   doc B: {len(docB)} tokens   packed: {pair.shape[1]}")
print(f"\nthe join, as strings (target at position {join} belongs to a different document):")
for k in range(join - 3, join + 3):
    tag = "  <-- CROSS-DOCUMENT" if k == join - 1 else ""
    print(f"   input {tok.decode([pair[0, k]])!r:<16} -> target "
          f"{tok.decode([pair[0, k+1]])!r:<16}{tag}")

with torch.no_grad():
    lg_pair = model.lm_head(model.model(input_ids=pair).last_hidden_state)
lab_pair = pair.clone()
lab_pair_masked = pair.clone(); lab_pair_masked[0, join] = -100

l_pair_on,  n_pair_on  = loss_and_count(lg_pair, lab_pair)
l_pair_off, n_pair_off = loss_and_count(lg_pair, lab_pair_masked)
per_pair = F.cross_entropy(lg_pair[:, :-1, :].reshape(-1, V).float(),
                           pair[:, 1:].reshape(-1), reduction="none")
print(f"\nloss AT the single join position     : {per_pair[join-1].item():7.4f}")
print(f"mean loss at every other position    : "
      f"{(per_pair.sum() - per_pair[join-1]).item() / (len(per_pair) - 1):7.4f}")
print(f"{'boundary TRAINED ON':<24} loss {l_pair_on:7.4f}   contributing {n_pair_on:>6,}")
print(f"{'boundary MASKED':<24} loss {l_pair_off:7.4f}   contributing {n_pair_off:>6,}")
assert n_pair_off == n_pair_on - 1


# ---- 4b. The same thing at batch scale, so the aggregate is readable -------------
PB, PT = 4, 256
# short slices, so joins are frequent rather than one per sequence
ids_p, is_start = pack([d[:500] for d in DOCS[30:400]], PT, PB, tok.eos_token_id)
with torch.no_grad():
    lg_p = model.lm_head(model.model(input_ids=ids_p).last_hidden_state)

# A boundary PREDICTION is one whose TARGET starts a new document. Targets are ids[:,1:],
# so the offending label positions are exactly the is_start positions (except index 0,
# which is never anyone's target).
boundary = is_start.clone(); boundary[:, 0] = 0
labels_join        = ids_p.clone()
labels_join_masked = labels_join.masked_fill(boundary.bool(), -100)

loss_join,  n_join  = loss_and_count(lg_p, labels_join)
loss_split, n_split = loss_and_count(lg_p, labels_join_masked)
n_boundaries = int(boundary.sum())

print(f"\n{PB} sequences x {PT} tokens, packed from many documents")
print(f"cross-document predictions in this batch: {n_boundaries}")
print(f"{'boundary TRAINED ON':<24} loss {loss_join:7.4f}   contributing {n_join:>6,}")
print(f"{'boundary MASKED':<24} loss {loss_split:7.4f}   contributing {n_split:>6,}")
print(f"{'change':<24}      {loss_split - loss_join:+7.4f}   {n_split - n_join:>+6,}")

ls = lg_p[:, :-1, :].reshape(-1, V).float()
tg = ids_p[:, 1:].reshape(-1)
per_tok = F.cross_entropy(ls, tg, reduction="none")
bmask   = boundary[:, 1:].reshape(-1).bool()
print(f"\nmean loss ON the {int(bmask.sum())} boundary positions : "
      f"{per_tok[bmask].mean().item():7.4f}")
print(f"mean loss on the {int((~bmask).sum())} other positions   : "
      f"{per_tok[~bmask].mean().item():7.4f}")
assert n_split == n_join - n_boundaries
assert int(bmask.sum()) == n_boundaries, "boundary mask and label mask disagree"
""")

md(r"""
**Why the difference.** The boundary positions carry a much higher loss than ordinary ones,
because the target is the opening token of an unrelated document — genuinely unpredictable
from the context. Masking them removes those large terms from the numerator *and* one count
each from the denominator, so the reported loss falls.

The point is not that the number got smaller. It is that the unmasked number was measuring
something the model should never be asked to learn, and gradient was being spent teaching it
that unrelated things follow each other.
""")

md(r"""
## Item 5 · An untrained model should sit at perplexity ≈ vocabulary size

The cheapest sanity check in the course. A randomly-initialised model has no reason to prefer
any token, so it is as unsure as a uniform draw from the vocabulary:

$$\text{loss} \approx \ln V, \qquad \text{perplexity} = e^{\text{loss}} \approx V$$

If your run does not start here, the target alignment is wrong — fix it before training.
""")

code(r"""
def perplexity(m, ids_, labels_=None, bs=2):
    labels_ = ids_ if labels_ is None else labels_
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, ids_.shape[0], bs):
            h  = m.model(input_ids=ids_[i:i+bs]).last_hidden_state
            lg = m.lm_head(h)[:, :-1, :].reshape(-1, V).float()
            tg = labels_[i:i+bs][:, 1:].reshape(-1)
            k  = int((tg != -100).sum())
            tot += F.cross_entropy(lg, tg, ignore_index=-100, reduction="sum").item(); n += k
    return tot / n, math.exp(tot / n)

from transformers import AutoConfig
untrained = type(model)(cfg).to(device=DEVICE, dtype=MODEL_DTYPE).eval()   # same arch, random weights
l_rand, p_rand = perplexity(untrained, ids_p)
l_train, p_train = perplexity(model, ids_p)

print(f"vocabulary size V            = {V:,}")
print(f"ln(V)                        = {math.log(V):.4f}   <- the target loss")
print(f"untrained loss               = {l_rand:.4f}")
print(f"untrained perplexity         = {p_rand:,.0f}")
print(f"ratio perplexity / V         = {p_rand / V:.3f}")
print(f"\npretrained loss              = {l_train:.4f}")
print(f"pretrained perplexity        = {p_train:,.1f}")
assert 0.5 < p_rand / V < 2.0, (
    f"untrained perplexity {p_rand:,.0f} is not near V={V:,} — check the shift before going further")
del untrained
""")

md(r"""
## Item 6 · Tied vs untied output head

Tying reuses the input embedding matrix as the output head. Untying gives the head its own
`[V, D]` matrix, which costs exactly `V x D` extra parameters and nothing else changes.
""")

code(r"""
tied_total   = sum(p.numel() for p in model.parameters())
head_params  = V * D
untied_total = tied_total + head_params

print(f"V x D                        = {V:,} x {D} = {head_params:,}")
print(f"tied   total parameters      = {tied_total:,}")
print(f"untied total parameters      = {untied_total:,}")
print(f"cost of untying              = +{head_params:,}  (+{100*head_params/tied_total:.1f}%)")
print(f"head as a share of the tied model = {100*head_params/tied_total:.1f}%")

# Prove the tying is real: the two tensors are the same object in memory.
same = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
print(f"\nlm_head.weight IS embed_tokens.weight (same storage): {same}")
assert untied_total - tied_total == head_params
""")

md(r"""
## Item 7 · Peak memory: ordinary cross-entropy vs a chunked one

The logits tensor is `[B, T, V]`. It exists only to be collapsed into one scalar, and the
backward pass needs its gradient too. Chunking computes the identical loss over blocks of
tokens, so only `chunk x V` logits are alive at once.

The loss must come out **bit-for-bit the same**, and so must the gradients. Both are asserted.
""")

code(r"""
def ce_full(hidden_, head, targets_, ignore_index=-100):
    lg = head(hidden_)
    return F.cross_entropy(lg.float().reshape(-1, V), targets_.reshape(-1),
                           ignore_index=ignore_index)

def ce_chunked(hidden_, head, targets_, chunk=1024, ignore_index=-100):
    '''Same objective, computed in blocks. Only `chunk x V` logits exist at any moment.
    Sum-reduce and divide once at the end -- taking a mean of per-chunk means would
    silently weight a short final chunk equally with a full one.'''
    h = hidden_.reshape(-1, hidden_.shape[-1])
    t = targets_.reshape(-1)
    n = int((t != ignore_index).sum())
    total = hidden_.new_zeros((), dtype=torch.float32)
    for i in range(0, h.shape[0], chunk):
        lg = head(h[i:i+chunk]).float()
        total = total + F.cross_entropy(lg, t[i:i+chunk],
                                        ignore_index=ignore_index, reduction="sum")
    return total / n

MB, MT, CHUNK = 4, 512, 512
mem_ids  = ids_p[:, :MT].repeat(max(1, MB // ids_p.shape[0]), 1)[:MB].contiguous()
with torch.no_grad():
    h_mem = model.model(input_ids=mem_ids).last_hidden_state
h_a = h_mem[:, :-1, :].detach().clone().requires_grad_(True)
h_b = h_mem[:, :-1, :].detach().clone().requires_grad_(True)
tg_mem = mem_ids[:, 1:]
head = model.lm_head

def peak(fn):
    '''Peak memory ATTRIBUTABLE TO fn, i.e. above what was already resident.

    Reporting raw max_memory_allocated() buries the result: the model weights and
    activations are ~1 GiB before the loss is even called, so a 287 MiB saving shows
    up as a 1.1x ratio and the effect looks negligible. Subtracting the baseline
    measures the thing chunking actually changes.'''
    if DEVICE == "cuda":
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn(); torch.cuda.synchronize()
        return out, (torch.cuda.max_memory_allocated() - base) / 2**20, base / 2**20
    return fn(), float("nan"), float("nan")   # CUDA-only counter

def run_full():
    l = ce_full(h_a, head, tg_mem); l.backward(); return l.item()
def run_chunked():
    l = ce_chunked(h_b, head, tg_mem, chunk=CHUNK); l.backward(); return l.item()

l_full,  m_full,  base_mib = peak(run_full)
l_chunk, m_chunk, _        = peak(run_chunked)
# size the analytic figures off the ACTUAL logit dtype, not an assumed fp32
LOGIT_BYTES = torch.finfo(model.lm_head.weight.dtype).bits // 8
logit_bytes = MB * (MT - 1) * V * LOGIT_BYTES / 2**20
chunk_bytes = CHUNK * V * LOGIT_BYTES / 2**20

print(f"batch {MB} x {MT-1} predictions, V = {V:,}, chunk = {CHUNK}, "
      f"logits are {model.lm_head.weight.dtype} ({LOGIT_BYTES} bytes)")
print(f"already resident before the loss: {base_mib:,.1f} MiB of weights and activations\n")
print(f"{'':22}{'loss':>12}{'peak MiB':>12}{'logits MiB (theory)':>22}")
print(f"{'ordinary CE':<22}{l_full:>12.6f}{m_full:>12.1f}{logit_bytes:>22.1f}")
print(f"{'chunked CE':<22}{l_chunk:>12.6f}{m_chunk:>12.1f}{chunk_bytes:>22.1f}")
if m_full == m_full:                       # not NaN
    print(f"\nmeasured peak-memory ratio  = {m_full / m_chunk:.2f}x "
          f"(above baseline; {m_full - m_chunk:,.1f} MiB saved)")
print(f"theoretical logits ratio    = {logit_bytes / chunk_bytes:.1f}x "
      f"(= {MB*(MT-1)} predictions / {CHUNK} chunk)")

print(f"\nloss difference             = {abs(l_full - l_chunk):.3e}")
print(f"max |grad difference|       = {(h_a.grad - h_b.grad).abs().max().item():.3e}")
assert abs(l_full - l_chunk) < 1e-4,  "chunking changed the loss - it must not"
assert (h_a.grad - h_b.grad).abs().max().item() < 1e-4, "chunking changed the gradients"
""")

md(r"""
The chunked version is not an approximation. Same loss, same gradients, different peak
memory — which is exactly the lesson's point that implementation can move memory by orders
of magnitude without touching the objective.

*(Peak allocation is a CUDA counter. On CPU the measured columns read `nan` and only the
analytic logits figures are meaningful.)*
""")

md(r"""
# Part 2 · A second head predicting `t+2`

The trunk is shared. Head 1 predicts the next token from position `i`; head 2 predicts the
token *after* that, from the same position. The two losses simply add.

```
position   i        i+1      i+2
head 1     -------> target
head 2     ----------------> target
```

Head 2 is randomly initialised, so it starts near `ln(V)` while head 1 starts where the
pretrained model already is. Watch what the gap does.
""")

code(r"""
class TwoHeadModel(nn.Module):
    '''Shared trunk, two output heads. Head 1 reuses the pretrained (tied) head so the
    comparison is against a real baseline; head 2 is new and untied.'''
    def __init__(self, base, cfg):
        super().__init__()
        self.trunk = base.model
        self.head1 = base.lm_head
        # Build head 2 in the TRUNK's dtype and device. A bare nn.Linear is fp32,
        # and if the checkpoint loaded as bf16 the first matmul dies with
        # "mat1 and mat2 have different dtype". Never assume fp32 -- ask the trunk.
        p0 = next(base.parameters())
        self.head2 = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False,
                               dtype=p0.dtype, device=p0.device)
        # init like the model's own embeddings rather than PyTorch's default, so
        # step 0 is a fair ln(V) start instead of an artificially large loss
        self.head2.weight.data.normal_(mean=0.0, std=cfg.initializer_range)

    def forward(self, ids_):
        h = self.trunk(input_ids=ids_).last_hidden_state
        return h

    def losses(self, ids_, labels_=None):
        lab = ids_ if labels_ is None else labels_
        h = self.forward(ids_)
        # head 1: position i -> token i+1. Drop the last position (nothing follows it).
        l1 = F.cross_entropy(self.head1(h[:, :-1, :]).float().reshape(-1, V),
                             lab[:, 1:].reshape(-1), ignore_index=-100)
        # head 2: position i -> token i+2. Drop the last TWO positions.
        l2 = F.cross_entropy(self.head2(h[:, :-2, :]).float().reshape(-1, V),
                             lab[:, 2:].reshape(-1), ignore_index=-100)
        return l1, l2

two = TwoHeadModel(model, cfg).to(DEVICE)
print(f"head 1 parameters: {two.head1.weight.numel():,} (tied to the embedding table)")
print(f"head 2 parameters: {two.head2.weight.numel():,} (new, untied)")
print(f"a second dense head costs +{100*two.head2.weight.numel()/tied_total:.1f}% of the model")
assert two.head2.weight.dtype == next(two.trunk.parameters()).dtype, (
    f"head 2 is {two.head2.weight.dtype} but the trunk is "
    f"{next(two.trunk.parameters()).dtype} - the matmul will fail")
print(f"head dtypes match trunk: {two.head1.weight.dtype} / {two.head2.weight.dtype}")
""")

code(r"""
# Training data: fixed-length packed sequences from the cleaned shard, boundaries masked.
STEPS   = int(os.environ.get("S9_STEPS", 300))
TB, TT  = int(os.environ.get("S9_BS", 8)), int(os.environ.get("S9_SEQ", 256))
LR      = 3e-5

train_ids, train_start = pack(DOCS[100:], TT, TB * (STEPS // 4 + 2), tok.eos_token_id)
train_lab = train_ids.masked_fill(train_start.bool(), -100)      # never train across a join
print(f"training tensor {tuple(train_ids.shape)}  "
      f"({int((train_lab != -100).sum()):,} contributing tokens)")

# AdamW on bf16/fp16 master weights silently loses small updates: at lr 3e-5 the
# step is below the representable gap and rounds to nothing. Fail loudly instead.
pdt = next(two.parameters()).dtype
assert pdt == torch.float32, (
    f"parameters are {pdt}; AdamW at lr={LR} will round most updates away. "
    "Load the model with dtype=torch.float32 rather than casting the heads to bf16.")

opt  = torch.optim.AdamW(two.parameters(), lr=LR)
hist = []
two.train()
for step in range(STEPS):
    i  = (step * TB) % (train_ids.shape[0] - TB)
    b  = train_ids[i:i+TB]; lb = train_lab[i:i+TB]
    l1, l2 = two.losses(b, lb)
    loss = l1 + l2                       # "the losses simply add"
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(two.parameters(), 1.0)
    opt.step()
    hist.append((step, l1.item(), l2.item()))
    if step % max(1, STEPS // 10) == 0 or step == STEPS - 1:
        print(f"step {step:>4}   head1 {l1.item():6.3f}   head2 {l2.item():6.3f}   "
              f"sum {loss.item():6.3f}   gap {l2.item()-l1.item():+6.3f}")
two.eval()
""")

code(r"""
import statistics
first = hist[:max(1, len(hist)//20)]
last  = hist[-max(1, len(hist)//20):]
f1, f2 = statistics.mean(h[1] for h in first), statistics.mean(h[2] for h in first)
e1, e2 = statistics.mean(h[1] for h in last),  statistics.mean(h[2] for h in last)

print(f"{'':16}{'head 1 (t+1)':>15}{'head 2 (t+2)':>15}{'sum':>10}{'gap':>10}")
print(f"{'first steps':<16}{f1:>15.4f}{f2:>15.4f}{f1+f2:>10.4f}{f2-f1:>+10.4f}")
print(f"{'last steps':<16}{e1:>15.4f}{e2:>15.4f}{e1+e2:>10.4f}{e2-e1:>+10.4f}")
print(f"{'improvement':<16}{f1-e1:>+15.4f}{f2-e2:>+15.4f}")
print(f"\nln(V) = {math.log(V):.3f}  <- where an untrained head starts")
print(f"head 2 perplexity: {math.exp(f2):,.0f} -> {math.exp(e2):,.0f}")
assert e2 > e1, "head 2 should stay above head 1: predicting two ahead is strictly harder"

# A head still above ln(V) after training is worse than guessing uniformly, which
# means it has not learned - not that the task is hard. Say so loudly rather than
# letting a plausible-looking number into the write-up.
if e2 >= math.log(V):
    print(f"\n*** WARNING: head 2 finished at {e2:.4f}, ABOVE ln(V) = {math.log(V):.4f}.")
    print("*** It is still worse than guessing uniformly, so it has not learned yet.")
    print("*** Usual cause: the model is in bfloat16 and AdamW is updating bf16 master")
    print("*** weights - at lr 3e-5 the updates round away. Load the model in fp32")
    print("*** (load_model_and_tokenizer pins dtype=torch.float32) and re-run.")
    print(f"*** Also check head 1: it went {f1:.4f} -> {e1:.4f}"
          f" ({'improved' if e1 < f1 else 'got WORSE - same cause'}).")
else:
    print(f"\nhead 2 finished {math.log(V) - e2:.4f} nats below ln(V): it has learned.")

try:
    import matplotlib.pyplot as plt
    s = [h[0] for h in hist]
    plt.figure(figsize=(9, 4))
    plt.plot(s, [h[1] for h in hist], label="head 1  (t+1)", lw=1.2)
    plt.plot(s, [h[2] for h in hist], label="head 2  (t+2)", lw=1.2)
    plt.axhline(math.log(V), ls="--", lw=.9, c="grey", label=f"ln(V) = {math.log(V):.2f}")
    plt.xlabel("step"); plt.ylabel("cross-entropy (nats)")
    plt.title("Two heads on a shared trunk"); plt.legend(); plt.tight_layout(); plt.show()
except Exception as e:
    print("(plot skipped:", e, ")")
""")

md(r"""
### What happens to head 2's loss, and why

**Head 2 starts far higher and falls much faster.** It begins near `ln(V)` because it is a
fresh random matrix — for the first stretch it is not learning language at all, it is
learning to read a residual stream that already contains useful information. Head 1 has no
such catch-up to do.

**It then flattens out above head 1, and stays there.** That gap is not a defect and it does
not close with more training, because it is a property of the data rather than of the model.
The distribution `P(token t+2 | context up to t)` has strictly higher entropy than
`P(token t+1 | context up to t)`: predicting two ahead means marginalising over the token in
between, which the model does not get to see. Conditioning on less information cannot lower
entropy — so head 2's floor is above head 1's, whatever the architecture.

**Why do it at all.** Two separate reasons, worth keeping apart:

- *In training*, it densifies the signal. Every position now receives two gradients, and the
  hidden state is pushed to carry information useful beyond the immediate next word. A
  representation that only supports `t+1` has learned something shallower.
- *At inference*, head 2 becomes a draft. It proposes a token the model has not properly
  computed, and the main path verifies it. This is speculative decoding where the draft model
  *is* the model — no second network resident in VRAM.

**The honest cost.** Each extra dense head is another `V x D` matrix. Here that is
28.3M parameters, +21% on a 135M model, for a head whose predictions get rejected most of
the time at inference. This is exactly the arithmetic that argues for a factored head.
""")

md("## Summary — the numbers to copy into the write-up")

code(r"""
print("=" * 74)
print("PART 1")
print("=" * 74)
print(f"1  shapes            tokens {tuple(tokens.shape)}  hidden {tuple(hidden.shape)}  "
      f"logits {tuple(logits.shape)}")
print(f"                     logits/hidden size ratio = V/D = {V/D:.1f}x")
print(f"2  shift             verified on strings; unshifted control reproduces input as target")
print(f"3  padding           loss {loss_pad:.4f} -> {loss_mask:.4f}   "
      f"contributing {n_pad:,} -> {n_mask:,}")
print(f"4  doc boundary      two docs, one join: loss {l_pair_on:.4f} -> {l_pair_off:.4f}   "
      f"(join position alone: {per_pair[join-1].item():.4f})")
print(f"                     batch, {n_boundaries} joins: loss {loss_join:.4f} -> {loss_split:.4f}   "
      f"contributing {n_join:,} -> {n_split:,}")
print(f"                     loss on boundary positions {per_tok[bmask].mean().item():.4f} "
      f"vs {per_tok[~bmask].mean().item():.4f} elsewhere")
print(f"5  untrained ppl     {p_rand:,.0f}   vs  V = {V:,}   (ratio {p_rand/V:.3f}, "
      f"loss {l_rand:.4f} vs ln V = {math.log(V):.4f})")
print(f"6  head params       tied {tied_total:,}  untied {untied_total:,}  "
      f"delta +{head_params:,} (+{100*head_params/tied_total:.1f}%)")
print(f"7  peak memory       ordinary {m_full:.1f} MiB   chunked {m_chunk:.1f} MiB"
      + (f"   ratio {m_full/m_chunk:.2f}x" if m_full == m_full else "   (CUDA only)"))
print(f"                     analytic logits {logit_bytes:.1f} -> {chunk_bytes:.1f} MiB "
      f"= {logit_bytes/chunk_bytes:.1f}x ; loss delta {abs(l_full-l_chunk):.2e}")
print("=" * 74)
print("PART 2")
print("=" * 74)
print(f"   head 1 (t+1)      {f1:.4f} -> {e1:.4f}")
print(f"   head 2 (t+2)      {f2:.4f} -> {e2:.4f}")
print(f"   sum               {f1+f2:.4f} -> {e1+e2:.4f}")
print(f"   gap (h2 - h1)     {f2-f1:+.4f} -> {e2-e1:+.4f}")
print("=" * 74)
""")

nb = {"cells": C, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python", "version": "3.10"},
      "colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "S9_loss_harness.ipynb")
json.dump(nb, open(out, "w"), indent=1)
print(f"wrote {out}  ({len(C)} cells: "
      f"{sum(1 for c in C if c['cell_type']=='code')} code, "
      f"{sum(1 for c in C if c['cell_type']=='markdown')} markdown)")
