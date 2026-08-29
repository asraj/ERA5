# Fix for the Part 2 dtype error, without re-uploading

**Error:** `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`

**Cause:** transformers v5 loads SmolLM2 in its checkpoint dtype, which is **bfloat16**.
`nn.Linear(...)` builds in **float32** unless told otherwise, so head 2's first matmul gets a
bf16 activation and an fp32 weight and refuses.

If you'd rather not re-upload, patch the two cells below in place. Otherwise just upload the
corrected `S9_loss_harness.ipynb` — it already has both fixes.

---

## Patch 1 — the model-loading cell (section 0)

Find:

```python
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg   = AutoConfig.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    return model, tok, cfg
```

Replace with:

```python
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
```

And just below, after the `total parameters` print, add:

```python
MODEL_DTYPE = next(model.parameters()).dtype
print(f"parameter dtype     = {MODEL_DTYPE}")
```

`MODEL_DTYPE` is used by the item-5 cell, so this line is required, not cosmetic.

## Patch 2 — `TwoHeadModel.__init__` (Part 2, cell 1)

Find:

```python
        self.head2 = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
```

Replace with:

```python
        # Build head 2 in the TRUNK's dtype and device. A bare nn.Linear is fp32,
        # and if the checkpoint loaded as bf16 the first matmul dies with
        # "mat1 and mat2 have different dtype". Never assume fp32 -- ask the trunk.
        p0 = next(base.parameters())
        self.head2 = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False,
                               dtype=p0.dtype, device=p0.device)
```

## Patch 3 — item 5, the random-init twin

Find:

```python
untrained = type(model)(cfg).to(DEVICE).eval()
```

Replace with:

```python
untrained = type(model)(cfg).to(device=DEVICE, dtype=MODEL_DTYPE).eval()
```

Otherwise the random twin is fp32 while the loaded model may not be, and the two perplexities
in that cell are not measured under the same conditions.

---

## Then

**Runtime → Restart session and run all.** Patches 1 and 3 touch cells above Part 2, so
re-running from the top is necessary; re-running only the failed cell will use the bf16 model
still in memory.

---

## Why the offline verification missed it

`verify_local.py` executes the notebook's real cells, but against a locally constructed
substitute model — and that substitute was fp32 throughout. It therefore reproduced the
architecture faithfully and the **dtype** not at all, which was the one property that mattered.
A stand-in only tests what it actually resembles.

Fixed by adding a regression that rebuilds `TwoHeadModel` on bf16, fp16 and fp32 trunks and runs
a real forward pass through each. Confirmed to have teeth: reverting the fix makes it fail with
the identical error you saw.

```
PASS  TwoHeadModel works on a bfloat16 trunk   head2=torch.bfloat16  l1=11.109 l2=10.917
PASS  TwoHeadModel works on a float16 trunk    head2=torch.float16   l1=10.980 l2=11.097
PASS  TwoHeadModel works on a float32 trunk    head2=torch.float32   l1=10.969 l2=10.936

# with the fix reverted:
FAIL  TwoHeadModel works on a bfloat16 trunk   RuntimeError: expected m1 and m2 to have the
                                               same dtype, but got: c10::BFloat16 != float
```

## One thing to watch on the re-run

fp32 makes the model ~540 MB of weights plus ~1.6 GB of AdamW state — comfortable on a T4, but
if you hit an OOM in Part 2, drop `S9_BS` to 4 rather than reaching for bf16. Mixed precision
would reintroduce exactly the dtype question this patch settles, and item 7's analytic figures
assume fp32 logits.
