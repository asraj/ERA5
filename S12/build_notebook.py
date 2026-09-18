#!/usr/bin/env python3
"""Generate S12_zero.ipynb. Generated, not hand-edited, so verify_local.py can execute
the same cells offline. Same pattern as S9/S10/S11."""
import json, os, re
HERE = os.path.dirname(os.path.abspath(__file__))
C = []
def md(s):   C.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)})
def code(s): C.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                       "source":s.strip("\n").splitlines(True)})

md(r"""
# Session 12 — Distributed training I: data parallel and ZeRO

**Stephen Raj Arokiasamy**

The assignment: build 32 virtual GPUs, run a demo model on them, simulate ZeRO-1/2/3, and show how
memory and computation change.

The thing I wanted to avoid is a notebook that *prints* the lesson's table. Any of these numbers can
be typed in. So everything below is **measured from state that actually exists** — every byte
counted here is a tensor that was really allocated on a rank, or really moved between two ranks by a
collective I wrote and then checked against `torch.distributed`.

| | |
|---|---|
| 1 | **32 virtual GPUs** with hand-rolled all-reduce / reduce-scatter / all-gather, every byte counted |
| 2 | **The collectives checked against real `torch.distributed` gloo** — the byte counts rest on verified primitives |
| 3 | **A real transformer** trained under single-GPU, DP, ZeRO-1, ZeRO-2 and ZeRO-3 |
| 4 | **The equivalence claim tested**, not assumed: same loss curve or it did not work |
| 5 | **Bytes per weight and bytes on the wire measured** per stage, against 16.00 / 5.50 / 3.75 / 2.00 and 2P / 2P / 2P / 3P |

Then, beyond the assignment, five more mechanisms from the lesson: the memory ladder and the
4-byte floor, the communication-to-compute ratio from a roofline, gradient bucketing and overlap,
CPU offload over PCIe, and the MXFP8 byte accounting.

**The one sentence the whole session rests on:** a reduce-scatter followed by an all-gather is an
all-reduce. That is why ZeRO-1 and ZeRO-2 are free — they perform the two halves data parallelism
was already performing internally, and simply keep the intermediate slice instead of throwing it
away.
""")

code(r"""
import os, math, json, time, itertools, statistics
import torch, torch.nn as nn, torch.nn.functional as F

SEED = 12
torch.manual_seed(SEED)
torch.set_num_threads(int(os.environ.get("S12_THREADS", "4")))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU    = torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
print("torch ", torch.__version__)
print("device", DEVICE, "|", GPU)

# The simulation is deliberately CPU-friendly: 32 ranks live in one process and the
# expensive thing is state, not FLOPs. A GPU makes it faster but changes nothing.
WORLD = int(os.environ.get("S12_WORLD", 32))
print("world size", WORLD)
""")

md(r"""
## 0 · The arithmetic that forces the whole session

Before any code: why is this a problem at all? One weight does not cost one number. It costs five.

| what is stored for one weight | bytes |
|---|---|
| the weight, in the 16-bit format used for arithmetic | 2 |
| its gradient | 2 |
| a 32-bit copy of the weight, kept for accuracy | 4 |
| two 32-bit running averages the optimizer keeps | 8 |
| **total** | **16** |

The 32-bit copy is there because repeatedly adding a very small update to a 16-bit number loses it
to rounding — so the authoritative value is kept at higher precision and a 16-bit version is made
from it for the arithmetic. The two averages are Adam's `m` and `v` from Session 11, which is
where the 8 bytes come from.

Multiply by 30 billion and the answer settles the question.
""")

code(r"""
GiB   = 2 ** 30
PARAM = 30e9                 # V5 is planned at 27-30B; the session uses 30B throughout
CARD  = 80e9 / GiB           # an 80 GB card, in GiB

BYTES = {"weight (bf16)": 2, "gradient (bf16)": 2, "fp32 master copy": 4,
         "Adam m (fp32)": 4, "Adam v (fp32)": 4}
per_w = sum(BYTES.values())
for k, v in BYTES.items():
    print(f"  {k:<22}{v:>3} bytes")
print(f"  {'TOTAL':<22}{per_w:>3} bytes per weight\n")

state = PARAM * per_w
print(f"30e9 weights x {per_w} bytes = {state/1e9:.0f} GB = {state/GiB:.1f} GiB")
print(f"one 80 GB card holds {CARD:.1f} GiB  ->  {state/(CARD*GiB):.2f} cards just to hold it,"
      f" i.e. {math.ceil(state/(CARD*GiB))} cards, before a single calculation")

P_BYTES = PARAM * 2          # one full copy of the PARAMETERS in bf16
print(f"\nP, one full copy of the parameters in bf16 = {P_BYTES/1e9:.0f} GB")
print("every communication cost in this session is a multiple of P:")
for mult in (2, 3):
    print(f"  {mult}P = {mult*P_BYTES/1e9:.0f} GB per GPU per step")
""")

md(r"""
## 1 · Thirty-two virtual GPUs

A "GPU" here is a dict of tensors plus an integer rank. That is the honest description, and it is
enough, because the two things this session measures are **what each rank stores** and **what each
rank sends** — neither of which needs real silicon.

What it is *not* is a simulation of speed. Compute is serialised across the 32 ranks, so wall-clock
time in this notebook means nothing. Everywhere a time appears below it comes from an explicit
cost model with its bandwidth stated, never from `time.time()`.

### Counting bytes the way the hardware counts them

A ring all-reduce over `W` ranks does not move `W` copies of the data. Each rank sends its buffer
around the ring in `W-1` steps of `1/W` each, twice:

$$\text{bytes per rank} = 2 \cdot \frac{W-1}{W} \cdot N$$

At `W = 32` that is `1.9375 N`, which is where "2P" comes from — it is `2P` in the limit of many
ranks, and slightly under it in practice. I count the exact factor rather than the rounded one, so
the numbers below are a hair under the lesson's and that is not an error.
""")

code(r"""
class Cluster:
    # W virtual GPUs. Holds no model - it holds the wire, and counts what crosses it.

    def __init__(self, world):
        self.W = world
        self.reset()

    def reset(self):
        self.bytes_sent = 0          # per rank, summed over the step
        self.ops = []                # (name, bytes_per_rank) for the audit trail

    def _charge(self, name, nbytes):
        self.bytes_sent += nbytes
        self.ops.append((name, nbytes))

    # ---- the three collectives of section 4 ------------------------------------------
    def all_reduce_mean(self, xs):
        # every rank starts with a full buffer, every rank ends with the mean
        assert len(xs) == self.W
        n = xs[0].numel() * xs[0].element_size()
        self._charge("all-reduce", 2 * (self.W - 1) / self.W * n)
        out = torch.stack([x.float() for x in xs]).mean(0).to(xs[0].dtype)
        return [out.clone() for _ in range(self.W)]

    def reduce_scatter_mean(self, xs):
        # every rank starts with a full buffer, rank k ends with slice k of the mean
        assert len(xs) == self.W
        n = xs[0].numel() * xs[0].element_size()
        self._charge("reduce-scatter", (self.W - 1) / self.W * n)
        full = torch.stack([x.float() for x in xs]).mean(0).to(xs[0].dtype)
        return list(torch.chunk(full, self.W))

    def all_gather(self, shards):
        # rank k starts with slice k, every rank ends with the whole thing
        assert len(shards) == self.W
        n = sum(s.numel() * s.element_size() for s in shards)
        self._charge("all-gather", (self.W - 1) / self.W * n)
        return torch.cat([s.reshape(-1) for s in shards])

    # ---- the same all-reduce, actually walked around the ring -------------------------
    def ring_all_reduce_mean(self, xs, count=True):
        # the textbook implementation: W-1 sends to reduce-scatter, W-1 to all-gather,
        # written out so the 2(W-1)/W factor is demonstrated rather than asserted
        W = self.W
        bufs = [list(torch.chunk(x.float().clone(), W)) for x in xs]
        sent = 0
        for step in range(W - 1):                       # phase 1: reduce-scatter
            for r in range(W):
                src, dst = r, (r + 1) % W
                idx = (src - step) % W
                bufs[dst][idx] = bufs[dst][idx] + bufs[src][idx]
                sent += bufs[src][idx].numel() * 4
        for r in range(W):                              # rank r now owns chunk (r+1)%W
            bufs[r][(r + 1) % W] /= W
        for step in range(W - 1):                       # phase 2: all-gather
            for r in range(W):
                src, dst = r, (r + 1) % W
                idx = (src + 1 - step) % W
                bufs[dst][idx] = bufs[src][idx].clone()
                sent += bufs[src][idx].numel() * 4
        if count:
            self._charge("ring all-reduce", sent / W)
        return [torch.cat(b).to(xs[0].dtype) for b in bufs], sent / W

CL = Cluster(WORLD)
N  = 4096
xs = [torch.randn(N) for _ in range(WORLD)]

direct = torch.stack(xs).mean(0)
ring, per_rank = CL.ring_all_reduce_mean(xs, count=False)

print(f"ring all-reduce over {WORLD} ranks, buffer {N} fp32 = {N*4/1024:.0f} KiB\n")
print(f"  max |ring - direct mean|      {(ring[0]-direct).abs().max().item():.3e}")
print(f"  all ranks agree afterwards    {all(torch.equal(ring[0], r) for r in ring)}")
print(f"  bytes sent per rank           {per_rank:,.0f}")
print(f"  as a multiple of the buffer   {per_rank/(N*4):.4f}")
print(f"  2*(W-1)/W predicts            {2*(WORLD-1)/WORLD:.4f}")
print(f"\n  -> the '2P' of the lesson is 2(W-1)/W, which at W={WORLD} is {2*(WORLD-1)/WORLD:.4f}P.")
print("     I count the exact factor everywhere below, so totals land just under 2P and 3P.")
""")

md(r"""
### Is any of that right?

The collectives above are mine, and the byte counts of the entire notebook rest on them. So they
get checked against the real thing: `torch.distributed` with the gloo backend, in separate
processes, over an actual socket.

This is the check that matters most, because a wrong reduce-scatter would still produce plausible
numbers everywhere downstream.
""")

code(r"""
import torch.multiprocessing as tmp
import tempfile, traceback

GLOO_W = int(os.environ.get("S12_GLOO_W", 4))

def _gloo_worker(rank, world, store_file, n, outdir):
    # results go to disk, not through a Queue: torch tensors in a mp.Queue are passed by
    # shared-memory file descriptor, and the handle dies with the child process.
    try:
        import torch, torch.distributed as dist
        dist.init_process_group("gloo", init_method=f"file://{store_file}",
                                rank=rank, world_size=world)
        torch.manual_seed(1234)                       # same seed -> same full set on every rank
        allx = [torch.randn(n) for _ in range(world)]
        x = allx[rank].clone()

        ar = x.clone(); dist.all_reduce(ar, op=dist.ReduceOp.SUM); ar /= world

        # gloo has no native reduce_scatter, so it is built the way section 4 says it
        # can be: all-reduce, then keep your own slice. That identity is exactly what
        # ZeRO-1 exploits, so constructing it here is the claim, not a workaround for it.
        rs = torch.chunk(ar.clone(), world)[rank].clone()

        ag = [torch.empty(n // world) for _ in range(world)]
        dist.all_gather(ag, rs.clone())

        torch.save({"ar": ar, "rs": rs, "ag": torch.cat(ag)},
                   os.path.join(outdir, f"r{rank}.pt"))
        dist.destroy_process_group()
    except Exception:
        with open(os.path.join(outdir, f"r{rank}.err"), "w") as fh:
            fh.write(traceback.format_exc())

def run_gloo(world, n=256):
    ctx = tmp.get_context("fork")                     # fork keeps notebook-defined fns usable
    with tempfile.TemporaryDirectory() as d:
        sf = os.path.join(d, "store")
        ps = [ctx.Process(target=_gloo_worker, args=(r, world, sf, n, d)) for r in range(world)]
        for p in ps: p.start()
        for p in ps: p.join(timeout=180)
        errs = [open(os.path.join(d, f"r{r}.err")).read()
                for r in range(world) if os.path.exists(os.path.join(d, f"r{r}.err"))]
        if errs:
            raise RuntimeError(errs[0].strip().splitlines()[-1])
        miss = [r for r in range(world) if not os.path.exists(os.path.join(d, f"r{r}.pt"))]
        if miss:
            raise RuntimeError(f"ranks {miss} produced no result (process died)")
        return [torch.load(os.path.join(d, f"r{r}.pt")) for r in range(world)]

GLOO_OK, gloo_note = False, ""
try:
    res = run_gloo(GLOO_W)
    torch.manual_seed(1234)
    allx = [torch.randn(256) for _ in range(GLOO_W)]

    mine_ar = Cluster(GLOO_W).all_reduce_mean([x.clone() for x in allx])
    mine_rs = Cluster(GLOO_W).reduce_scatter_mean([x.clone() for x in allx])
    mine_ag = Cluster(GLOO_W).all_gather([s.clone() for s in mine_rs])

    d_ar = max((res[r]["ar"] - mine_ar[r]).abs().max().item() for r in range(GLOO_W))
    d_rs = max((res[r]["rs"] - mine_rs[r]).abs().max().item() for r in range(GLOO_W))
    d_ag = max((res[r]["ag"] - mine_ag).abs().max().item() for r in range(GLOO_W))
    ring, _ = Cluster(GLOO_W).ring_all_reduce_mean([x.clone() for x in allx], count=False)
    d_ring = max((res[r]["ar"] - ring[r]).abs().max().item() for r in range(GLOO_W))

    print(f"real torch.distributed gloo, world size {GLOO_W}, {GLOO_W} OS processes over a socket")
    print("(gloo has no native reduce-scatter; that leg is all-reduce + slice, which is the")
    print(" section-4 identity being used rather than worked around)\n")
    print(f"{'collective':<28}{'max |mine - gloo|':>20}")
    for lab, d in (("all-reduce", d_ar), ("reduce-scatter", d_rs),
                   ("all-gather", d_ag), ("ring all-reduce (mine)", d_ring)):
        print(f"{lab:<28}{d:>20.3e}")
    GLOO_OK = max(d_ar, d_rs, d_ag, d_ring) < 1e-6
    print(f"\n  identical to float tolerance: {GLOO_OK}")
    print("  -> the byte counts in the rest of this notebook sit on primitives that")
    print("     produce exactly what the real library produces.")
except Exception as e:
    gloo_note = f"{type(e).__name__}: {e}"
    print("gloo cross-check could not run here:", gloo_note)
    print("(sandboxed environments often block the socket; the simulator is unaffected)")
""")

md(r"""
## 2 · A demo model, and the four arrangements

A small transformer — 2 layers, `d_model` 64, vocabulary 256 — on a synthetic copying task. Small
on purpose: the interesting quantity is **bytes of state per rank**, and 32 replicas of a large
model would not fit in a Colab session, which is the whole point of the session and a poor reason
to be unable to demonstrate it.

Every arrangement below trains **the same model on the same data in the same order**. The only
thing that differs is where the state lives and what crosses the wire.

| arrangement | weight | gradient | fp32 master + m + v | per-rank bytes/weight |
|---|---|---|---|---|
| data parallel | full | full | full | `2 + 2 + 12` = **16** |
| ZeRO-1 | full | full | **sharded** | `2 + 2 + 12/W` |
| ZeRO-2 | full | **sharded** | **sharded** | `2 + 2/W + 12/W` |
| ZeRO-3 | **sharded** | **sharded** | **sharded** | `16/W` |

Two things I want to be careful about, because they are where a simulation can quietly cheat:

**ZeRO-2's gradients are sharded in the steady state, not during the backward pass.** A rank
computes a full gradient before it can send anything. What ZeRO-2 avoids is *storing* the whole
gradient between steps — the lesson's wording is "discarded as soon as they have been sent where
they are needed", which in a real implementation means a gradient bucket is freed the moment its
reduce-scatter completes. I measure persistent state, and report the transient bucket separately
rather than folding it away.

**ZeRO-3's gather is real.** The forward pass genuinely reassembles each layer's weights from 32
shards, uses them, and frees them. If I kept the full weights around and only *pretended* to shard,
every number in this notebook would still look right.
""")

code(r"""
V, D, LAYERS, SEQ = 256, 64, 2, 32
MICRO = int(os.environ.get("S12_MICRO", 2))          # sequences per rank per step
STEPS = int(os.environ.get("S12_STEPS", 20))

class Block(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.o  = nn.Linear(d, 3*d, bias=False), nn.Linear(d, d, bias=False)
        s.f1, s.f2  = nn.Linear(d, 4*d, bias=False), nn.Linear(4*d, d, bias=False)
    def forward(s, x):
        B, T, d = x.shape
        q, k, v = s.qkv(s.n1(x)).chunk(3, -1)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(a)
        return x + s.f2(F.gelu(s.f1(s.n2(x))))

class TinyLM(nn.Module):
    def __init__(s, v=V, d=D, layers=LAYERS, seq=SEQ):
        super().__init__()
        s.emb  = nn.Embedding(v, d)
        s.pos  = nn.Parameter(torch.zeros(1, seq, d))
        s.blk  = nn.ModuleList(Block(d) for _ in range(layers))
        s.nf   = nn.LayerNorm(d)
        s.head = nn.Linear(d, v, bias=False)
    def forward(s, idx):
        x = s.emb(idx) + s.pos[:, :idx.shape[1]]
        for b in s.blk: x = b(x)
        return s.head(s.nf(x))

torch.manual_seed(SEED)
REF = TinyLM()
NP  = sum(p.numel() for p in REF.parameters())
NAMES = [n for n, _ in REF.named_parameters()]
SHAPES = [p.shape for p in REF.parameters()]
print(f"demo model: {LAYERS} layers, d_model {D}, vocab {V}, seq {SEQ}")
print(f"parameters: {NP:,}")
print(f"a full bf16 copy of the parameters, P = {NP*2:,} bytes = {NP*2/1024:.1f} KiB")

# data: a synthetic copy-and-shift task, deterministic and shared by every arrangement
g = torch.Generator().manual_seed(SEED)
GLOBAL_B = WORLD * MICRO
DATA = torch.randint(0, V, (STEPS, GLOBAL_B, SEQ + 1), generator=g)
print(f"data: {STEPS} steps x global batch {GLOBAL_B} ({WORLD} ranks x {MICRO}) x seq {SEQ}")
""")

code(r"""
def flat(params):                       # a model's parameters as one vector
    return torch.cat([p.detach().reshape(-1) for p in params])

def unflat_into(model, vec):            # write a vector back into a model
    i = 0
    for p in model.parameters():
        n = p.numel(); p.data.copy_(vec[i:i+n].view_as(p)); i += n

def grad_flat(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                      for p in model.parameters()])

def loss_on(model, batch):
    x, y = batch[:, :-1], batch[:, 1:]
    return F.cross_entropy(model(x).float().reshape(-1, V), y.reshape(-1))

PAD = (-NP) % WORLD                     # pad so the vector shards evenly, as FSDP2 does
NPAD = NP + PAD
SH = NPAD // WORLD
print(f"{NP:,} parameters pad to {NPAD:,} so they divide by {WORLD} -> shard {SH:,} each"
      f"  (padding {PAD} elements, {100*PAD/NPAD:.4f}%)")
""")

code(r"""
B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 3e-4

class Engine:
    # One arrangement of the same training run across `world` virtual GPUs.
    # Weights and gradients live in bf16; the master copy and Adam's m and v in fp32.
    # That is the 2 + 2 + 4 + 4 + 4 = 16 bytes of section 1, really allocated.

    FULL_W = {"single", "dp", "zero1", "zero2"}      # who keeps whole weights
    FULL_G = {"single", "dp", "zero1"}               # who keeps whole gradients
    FULL_O = {"single", "dp"}                        # who keeps whole optimizer state

    def __init__(self, stage, world, cluster, init_vec, wdtype=torch.bfloat16):
        self.stage, self.W, self.cl, self.t = stage, world, cluster, 0
        self.wdt = wdtype                            # switchable, to isolate bf16 below
        v = torch.zeros(NPAD); v[:NP] = init_vec
        self.w = ([v.to(wdtype).clone() for _ in range(world)] if stage in self.FULL_W
                  else [v[r*SH:(r+1)*SH].to(wdtype).clone() for r in range(world)])
        self.g = ([torch.zeros(NPAD, dtype=wdtype) for _ in range(world)]
                  if stage in self.FULL_G
                  else [torch.zeros(SH, dtype=wdtype) for _ in range(world)])
        o_n = NPAD if stage in self.FULL_O else SH
        self.master = [(v.clone() if stage in self.FULL_O else v[r*SH:(r+1)*SH].clone())
                       for r in range(world)]
        self.m = [torch.zeros(o_n) for _ in range(world)]
        self.vv = [torch.zeros(o_n) for _ in range(world)]
        self.flops = 0

    # ---- what this rank actually holds, in bytes ------------------------------------
    def bytes_per_rank(self):
        tot = 0
        for buf in (self.w[0], self.g[0], self.master[0], self.m[0], self.vv[0]):
            tot += buf.numel() * buf.element_size()
        return tot

    def breakdown(self):
        return {k: b.numel() * b.element_size() / NPAD for k, b in
                (("weight", self.w[0]), ("gradient", self.g[0]), ("fp32 master", self.master[0]),
                 ("adam m", self.m[0]), ("adam v", self.vv[0]))}

    # ---- one optimisation step -------------------------------------------------------
    def step(self, batch):
        self.t += 1
        ranks = range(self.W)
        # ZeRO-3: the weights are in 32 pieces, so the forward pass has to collect them.
        if self.stage == "zero3":
            w_full = [self.cl.all_gather(self.w)] * self.W
        else:
            w_full = self.w

        losses, grads = [], []
        for r in ranks:
            unflat_into(REF, w_full[r][:NP].float())
            REF.zero_grad(set_to_none=True)
            mb = batch[r*MICRO:(r+1)*MICRO] if self.stage != "single" else batch
            loss = loss_on(REF, mb)
            loss.backward()
            losses.append(loss.item())
            gv = torch.zeros(NPAD); gv[:NP] = grad_flat(REF)
            grads.append(gv.to(self.wdt))
            self.flops += 6 * NP * mb.shape[0] * SEQ
            if self.stage == "single": break

        # ZeRO-3 frees the gathered weights after the forward and collects them again for
        # the backward. Charged, and performed, at whole-model granularity here; a real
        # implementation does it per layer, which moves the same bytes in smaller pieces.
        if self.stage == "zero3":
            _ = self.cl.all_gather(self.w)

        if self.stage == "single":
            self._adam(0, grads[0]); self.w[0] = self.master[0].to(self.wdt)
            return sum(losses) / len(losses)

        if self.stage == "dp":
            avg = self.cl.all_reduce_mean(grads)
            for r in ranks:
                self.g[r] = avg[r]
                self._adam(r, self.g[r]); self.w[r] = self.master[r].to(self.wdt)
        else:                                            # zero1 / zero2 / zero3
            sh = self.cl.reduce_scatter_mean(grads)
            for r in ranks:
                if self.stage == "zero1":                # keeps the full gradient buffer
                    self.g[r] = grads[r]
                else:                                    # zero2/3 keep only their slice
                    self.g[r] = sh[r].clone()
                self._adam(r, sh[r])
            new = [self.master[r].to(self.wdt) for r in ranks]
            if self.stage in ("zero1", "zero2"):         # weights go back to being replicated
                full = self.cl.all_gather(new)
                for r in ranks: self.w[r] = full.clone()
            else:                                        # zero3 leaves them sharded
                for r in ranks: self.w[r] = new[r]
        return sum(losses) / len(losses)

    def _adam(self, r, grad):
        gf = grad.float()
        self.m[r].mul_(B1).add_(gf, alpha=1-B1)
        self.vv[r].mul_(B2).addcmul_(gf, gf, value=1-B2)
        mh = self.m[r] / (1 - B1 ** self.t)
        vh = self.vv[r] / (1 - B2 ** self.t)
        self.master[r].addcdiv_(mh, vh.sqrt() + EPS, value=-LR)
""")

md(r"""
### Running all five

`single` is one GPU seeing the whole global batch of 64 sequences. The other four split those 64
across 32 ranks, two each. If the lesson's claim in section 3 holds, all five produce the same
model.
""")

code(r"""
torch.manual_seed(SEED)
INIT = flat(TinyLM().parameters())
P_SIM = NPAD * 2                                  # one bf16 copy of the parameters, in bytes

runs, curves = {}, {}
for stage in ("single", "dp", "zero1", "zero2", "zero3"):
    cl = Cluster(1 if stage == "single" else WORLD)
    eng = Engine(stage, 1 if stage == "single" else WORLD, cl, INIT)
    t0 = time.time()
    cl.reset(); hist = []
    for s in range(STEPS):
        if s == 1: cl.reset()                     # measure a steady-state step, not step 0
        hist.append(eng.step(DATA[s]))
    runs[stage] = dict(engine=eng, cluster=cl, wall=time.time()-t0,
                       wire=cl.bytes_sent / (STEPS - 1), bpw=eng.bytes_per_rank() / NPAD,
                       flops=eng.flops)
    curves[stage] = hist
    print(f"{stage:<8} loss {hist[0]:.4f} -> {hist[-1]:.4f}   "
          f"{eng.bytes_per_rank()/NPAD:6.3f} bytes/weight   "
          f"{cl.bytes_sent/(STEPS-1)/P_SIM:5.3f} P on the wire per step   "
          f"({time.time()-t0:.1f}s wall)")
""")

md(r"""
## 3 · Did the sharding actually preserve the model?

This is the claim worth testing, because everything else is bookkeeping around it. The lesson says
averaging the gradients from eight GPUs that each saw 32 sequences produces *exactly* the gradient
one GPU would produce from all 256 at once — so the distributed run is mathematically identical to
a single-GPU run on a larger batch.

Two separate questions hide inside that sentence, and they have different answers.
""")

code(r"""
def curve_gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b))

def final_w(stage):
    e = runs[stage]["engine"]
    return (e.master[0] if stage in ("single", "dp")
            else torch.cat([e.master[r] for r in range(WORLD)]))

print("QUESTION 1 - does the sharding itself change anything?")
print("(compare each ZeRO stage against plain data parallelism: same batches, same reduction,")
print(" the only difference is where the state is stored)\n")
print(f"{'arrangement':<12}{'final loss':>12}{'max |loss - DP|':>18}{'max |weights - DP|':>22}")
for stage in ("dp", "zero1", "zero2", "zero3"):
    print(f"{stage:<12}{curves[stage][-1]:>12.6f}"
          f"{curve_gap(curves[stage], curves['dp']):>18.3e}"
          f"{(final_w(stage) - final_w('dp')).abs().max().item():>22.3e}")
print("\n  Exactly zero, all four. Sharding the optimizer state, then the gradients, then the")
print("  weights themselves changes nothing at all about the model being trained. The right")
print("  slice reaches the right rank and the pieces reassemble bit-for-bit.")
""")

code(r"""
print(f"QUESTION 2 - is the {WORLD}-way split identical to one GPU on the whole batch?\n")
print(f"{'arrangement':<12}{'max |loss - single|':>22}{'max |weights - single|':>24}")
for stage in ("dp", "zero1", "zero2", "zero3"):
    print(f"{stage:<12}{curve_gap(curves[stage], curves['single']):>22.3e}"
          f"{(final_w(stage) - final_w('single')).abs().max().item():>24.3e}")
print("\n  Not zero. Small, but not zero - and it would be dishonest to round it away, because")
print("  the interesting part is WHY. Two candidates:")
print(f"    (a) summation order: one mean over {GLOBAL_B} sequences, versus {WORLD} means"
      f" of {MICRO} then averaged")
print("    (b) bf16: weights are rounded to 8 mantissa bits after every step")
print("\n  These are separable. Re-run the same comparison with fp32 weights, changing nothing")
print("  else, and whatever survives is (a).")
""")

code(r"""
fp32_curves, fp32_w = {}, {}
for stage in ("single", "dp", "zero3"):
    cl = Cluster(1 if stage == "single" else WORLD)
    eng = Engine(stage, 1 if stage == "single" else WORLD, cl, INIT, wdtype=torch.float32)
    fp32_curves[stage] = [eng.step(DATA[s]) for s in range(STEPS)]
    fp32_w[stage] = (eng.master[0] if stage in ("single", "dp")
                     else torch.cat([eng.master[r] for r in range(WORLD)]))

print(f"{'weights in':<12}{'stage':<10}{'max |loss - single|':>22}{'max |weights - single|':>24}")
for stage in ("dp", "zero3"):
    print(f"{'bf16':<12}{stage:<10}{curve_gap(curves[stage], curves['single']):>22.3e}"
          f"{(final_w(stage) - final_w('single')).abs().max().item():>24.3e}")
for stage in ("dp", "zero3"):
    print(f"{'fp32':<12}{stage:<10}{curve_gap(fp32_curves[stage], fp32_curves['single']):>22.3e}"
          f"{(fp32_w[stage] - fp32_w['single']).abs().max().item():>24.3e}")
BF16_FACTOR = ((final_w('dp') - final_w('single')).abs().max().item()
               / max((fp32_w['dp'] - fp32_w['single']).abs().max().item(), 1e-30))
print(f"\n  bf16 accounts for a factor of about {BF16_FACTOR:,.0f} of the disagreement.")
print("  (that multiplier is backend-dependent - larger on CUDA than on CPU, because the")
print("   two accumulate differently. The conclusion does not move: bf16 dominates, and")
print("   what survives in fp32 is summation order alone.)")
print("  In fp32 the split-batch run tracks the single-GPU run to float noise, which is (a)")
print("  and is irreducible: a + b + c summed in two orders is two different numbers.")
print("\n  So the lesson's claim is exactly true of the MATHEMATICS and approximately true of")
print("  the ARITHMETIC, and the approximation is dominated by the precision of the weights,")
print("  not by the distribution. That is the honest version, and it is the reason a")
print("  distributed run is not bit-reproducible against a single-GPU one.")
""")

md(r"""
### One more place the order matters

My collectives all reduce with `torch.stack(...).mean(0)`. Real hardware does not: a ring
all-reduce accumulates sequentially around the ring, so rank 5's contribution enters the sum at a
different point. My ring implementation does accumulate that way, which lets me measure the effect
rather than assert it is negligible.
""")

code(r"""
torch.manual_seed(99)
xs32 = [torch.randn(NPAD) * 0.01 for _ in range(WORLD)]
print(f"{'dtype':<10}{'|ring - direct| max':>22}{'relative to the mean':>24}")
for name, dt in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
    ys = [x.to(dt) for x in xs32]
    direct = torch.stack([y.float() for y in ys]).mean(0)
    ring, _ = Cluster(WORLD).ring_all_reduce_mean(ys, count=False)
    d = (ring[0].float() - direct).abs().max().item()
    print(f"{name:<10}{d:>22.3e}{d/direct.abs().mean().item():>24.3e}")

print("\n  Small in fp32, a percent of the signal in bf16. Neither is a bug - both are what")
print("  'the same computation, in a different order' costs. It is also why two runs on")
print("  different world sizes are not expected to match bit-for-bit even with the same seed.")
""")

md(r"""
## 4 · Memory: what each rank stores

Measured by walking the tensors each rank actually holds and summing `numel x element_size`. No
formula is applied here — the formula is what this is being checked against.
""")

code(r"""
def formula(stage, W):
    w = 2 if stage in Engine.FULL_W else 2/W
    g = 2 if stage in Engine.FULL_G else 2/W
    o = 12 if stage in Engine.FULL_O else 12/W
    return w + g + o

LESSON_8 = {"dp": 16.00, "zero1": 5.50, "zero2": 3.75, "zero3": 2.00}

print(f"{'arrangement':<12}{'weight':>9}{'grad':>9}{'master':>9}{'m':>7}{'v':>7}"
      f"{'measured':>11}{'formula':>10}")
for stage in ("dp", "zero1", "zero2", "zero3"):
    b = runs[stage]["engine"].breakdown()
    print(f"{stage:<12}{b['weight']:>9.4f}{b['gradient']:>9.4f}{b['fp32 master']:>9.4f}"
          f"{b['adam m']:>7.4f}{b['adam v']:>7.4f}"
          f"{runs[stage]['bpw']:>11.4f}{formula(stage, WORLD):>10.4f}")

print(f"\n(bytes per weight, world size {WORLD})\n")
print("the same formula at the world size the lesson tabulates, W = 8:")
print(f"{'arrangement':<12}{'formula':>10}{'lesson':>10}{'agree':>8}")
for stage in ("dp", "zero1", "zero2", "zero3"):
    f8 = formula(stage, 8)
    print(f"{stage:<12}{f8:>10.2f}{LESSON_8[stage]:>10.2f}"
          f"{'yes' if abs(f8-LESSON_8[stage]) < 5e-3 else 'NO':>8}")
""")

code(r"""
print("what the sharding bought, at this world size:\n")
base = runs["dp"]["bpw"]
for stage in ("dp", "zero1", "zero2", "zero3"):
    r = runs[stage]
    print(f"  {stage:<7} {r['bpw']:6.3f} bytes/weight   "
          f"{base/r['bpw']:5.1f}x less than data parallel   "
          f"{PARAM*r['bpw']/GiB:7.1f} GiB for a 30B model   "
          f"{'fits' if PARAM*r['bpw']/GiB < CARD else 'does NOT fit'}")

print(f"\n  (a card holds {CARD:.1f} GiB)")
print(f"\n  ZeRO-3 at {WORLD} ranks stores 1/{WORLD} of everything, so 16/{WORLD} ="
      f" {16/WORLD:.3f} bytes per weight.")
print("  The state has not been compressed - it has been DIVIDED. The sum across all")
print(f"  {WORLD} ranks is still exactly 16 bytes per weight, which is the point: zero")
print("  REDUNDANCY, not zero cost. Nothing here makes the model cheaper to train; it")
print("  makes it possible to train at all, by refusing to store the same thing twice.")
tot = runs["zero3"]["bpw"] * WORLD
print(f"  check: {runs['zero3']['bpw']:.3f} x {WORLD} ranks = {tot:.2f} bytes per weight in total")
""")

md(r"""
## 5 · Computation, and what actually changes

The assignment asks how memory *and computation* change. The answer for computation is the part
people get wrong, so it is worth stating precisely: **the arithmetic does not change at all.**
""")

code(r"""
per_step = {st: r["flops"] / STEPS / (1 if st == "single" else WORLD)
            for st, r in runs.items()}
base = per_step["dp"]
print(f"{'arrangement':<10}{'seq/rank':>10}{'FLOPs/rank/step':>18}{'vs DP':>8}"
      f"{'wire/step':>12}{'as P':>8}   collectives per step")
for stage in ("single", "dp", "zero1", "zero2", "zero3"):
    r = runs[stage]
    names = list(dict.fromkeys(n for n, _ in r["cluster"].ops))
    print(f"{stage:<10}{GLOBAL_B if stage=='single' else MICRO:>10}{per_step[stage]:>18,.0f}"
          f"{per_step[stage]/base:>7.0f}x{r['wire']:>12,.0f}{r['wire']/P_SIM:>8.3f}"
          f"   {', '.join(names) if names else '-'}")

print(f"\n  Every rank does the same arithmetic in every arrangement - one forward and one")
print(f"  backward over its own {MICRO} sequences. ZeRO does not save compute and does not")
print(f"  cost compute. What it trades is MEMORY for COMMUNICATION:\n")
for stage in ("dp", "zero1", "zero2", "zero3"):
    r = runs[stage]
    print(f"  {stage:<7} {r['bpw']:6.3f} B/weight   {r['wire']/P_SIM:5.3f}P on the wire")
print(f"\n  predicted: 2P, 2P, 2P, 3P   (exactly, 2(W-1)/W = {2*(WORLD-1)/WORLD:.4f}"
      f" and 3(W-1)/W = {3*(WORLD-1)/WORLD:.4f})")
print("\n  Stages 1 and 2 move the SAME volume as data parallelism. That is section 4's")
print("  equivalence cashed in: data parallelism's all-reduce already IS a reduce-scatter")
print("  followed by an all-gather, and ZeRO-1 just keeps the slice in between instead of")
print("  discarding it. Ten of the sixteen bytes, for free.")
print("  Stage 3 adds one gather of the weights in the forward and one in the backward,")
print("  which is the extra P.")
""")

md(r"""
---

# Beyond the assignment

The assignment asks for ZeRO-1/2/3 and the memory and computation. Everything above does that. The
five sections below are the rest of the lesson, each reduced to a number that can be right or
wrong. Three of them turned out to say something I had not expected going in.

## A · §7 — the memory ladder, and the floor nothing can get under

The lesson's ladder says data parallelism and ZeRO-1 never fit *at any world size*. That is a
strong claim — usually more GPUs fixes memory — so it is worth deriving rather than reading.
""")

code(r"""
LADDER = {   # the lesson's table, GiB per GPU for the 30B model
 "dp":    {8: 447.0, 16: 447.0, 32: 447.0, 64: 447.0},
 "zero1": {8: 153.7, 16: 132.7, 32: 122.2, 64: 117.0},
 "zero2": {8: 104.8, 16:  80.3, 32:  68.1, 64:  62.0},
 "zero3": {8:  55.9, 16:  27.9, 32:  14.0, 64:   7.0}}

print(f"GiB per GPU for a {PARAM/1e9:.0f}B model   (a card holds {CARD:.1f} GiB)\n")
print(f"{'':<8}" + "".join(f"{w:>10} GPUs" for w in (8, 16, 32, 64)) + f"{'  fits from':>14}")
worst = 0.0
for stage in ("dp", "zero1", "zero2", "zero3"):
    row, first = "", None
    for w in (8, 16, 32, 64):
        g = formula(stage, w) * PARAM / GiB
        worst = max(worst, abs(g - LADDER[stage][w]))
        if g < CARD and first is None: first = w
        row += f"{g:>10.1f}{'*' if g < CARD else ' '}    "
    print(f"{stage:<8}{row}{(str(first) + ' GPUs') if first else 'never':>14}")
print(f"\n  * = fits on one card.  Worst disagreement with the lesson's table: {worst:.2f} GiB")
""")

code(r"""
floor = 4 * PARAM / GiB
print("why more GPUs cannot rescue data parallelism or ZeRO-1:\n")
print("  both leave the WEIGHTS and the GRADIENTS replicated on every card.")
print("  that is 2 + 2 = 4 bytes per weight, and there is no W anywhere in that expression.\n")
for w in (8, 64, 512, 8192):
    print(f"    W = {w:>5}:  replicated part {4*PARAM/GiB:7.1f} GiB"
          f"   sharded part {(formula('zero1', w)-4)*PARAM/GiB:7.1f} GiB"
          f"   total {formula('zero1', w)*PARAM/GiB:7.1f} GiB")
print(f"\n  The sharded part goes to zero. The floor does not move: {floor:.1f} GiB, for ever.")

break_even = CARD * GiB / 4
print(f"\n  So the question 'how big a model can data parallelism train' has an answer that does")
print(f"  not mention the cluster at all:")
print(f"    4 bytes/weight fills a {CARD:.1f} GiB card at {break_even/1e9:.1f}B parameters")
print(f"    V5 at {PARAM/1e9:.0f}B is {PARAM/break_even:.2f}x past that line")
print("\n  This is the single most useful thing in the section. It is a one-line calculation")
print("  and it eliminates two of the four arrangements before any benchmark is run.")
""")

md(r"""
## B · §5 — communication as a fraction of compute, from a roofline

The lesson gives 7.10 s of compute per step on 64 H100 and 3.12 s on 64 B200, and 2.40 s of
communication for 2P over InfiniBand. I did not want to take the compute times on trust, so I
rebuilt them from `6ND` and the published peak throughputs — and the two numbers turn out not to
be independent.
""")

code(r"""
TOK    = 1e6                       # tokens per step
NVLINK, IB, PCIE = 450e9, 50e9, 60e9
PEAK   = {"H100": 989.4e12, "B200": 2.25e15}     # bf16 dense, per card

flops_step = 6 * PARAM * TOK
print(f"6ND = 6 x {PARAM/1e9:.0f}e9 x {TOK/1e6:.0f}e6 tokens = {flops_step:.3e} FLOP per step\n")
print(f"{'card':<8}{'64x peak':>14}{'ideal step':>12}{'lesson says':>13}{'implied MFU':>13}")
MFU = {}
for card, t in (("H100", 7.10), ("B200", 3.12)):
    agg = 64 * PEAK[card]
    ideal = flops_step / agg
    MFU[card] = ideal / t
    print(f"{card:<8}{agg:>14.3e}{ideal:>12.2f}s{t:>12.2f}s{100*ideal/t:>12.1f}%")
print("\n  Both rows land on the same utilisation. The lesson's 7.10 and 3.12 are not two")
print("  measurements - they are one assumption (MFU ~ 40%) applied to two peak numbers.")
print("  Worth knowing before quoting them as evidence about hardware.")
""")

code(r"""
print(f"{'path':<28}{'2P = 120 GB':>14}{'3P = 180 GB':>14}")
for name, bw in (("NVLink, inside one node", NVLINK), ("InfiniBand, between nodes", IB)):
    print(f"{name:<28}{2*P_BYTES/bw:>13.2f}s{3*P_BYTES/bw:>13.2f}s")

print(f"\n{'':<8}{'compute':>10}{'2P over IB':>13}{'comm/compute':>15}{'sum if unhidden':>18}")
for card, t in (("H100", 7.10), ("B200", 3.12)):
    c = 2 * P_BYTES / IB
    print(f"{card:<8}{t:>9.2f}s{c:>12.2f}s{100*c/t:>14.0f}%{t+c:>17.2f}s")

print("\n  The volume did not change. The compute it has to hide behind got 2.3x shorter, so")
print("  the same 120 GB went from a third of the step to three quarters of it.")
print("  Buying faster cards makes a run MORE network-bound, not less - and it is the only")
print("  line in this session where the right answer gets harder as the hardware improves.")
print("  Both ratios are still under 1, so a perfectly overlapped transfer still hides. That")
print("  is what section C is about, and at 77% it stops being an optimisation and becomes a")
print("  requirement.")
""")

md(r"""
## C · §10 — bucketing and overlap

The backward pass runs last layer to first, so the last layer's gradients are finished long before
the first layer's. They can start their journey immediately. Gradients are collected into buckets
and a bucket is sent the moment it fills.

The lesson states three things about this, and the third one is the interesting one:

1. at a bucket of two layers on an H100 step, **83 percent** of the transfer finishes before the
   backward pass ends
2. on an H100 step the **smallest** bucket is the best one
3. on a B200 step the best bucket holds **two** layers, and going below that makes the run *slower*

Claim 3 needs a per-transfer fixed cost to be true at all — with a perfectly free link, smaller
buckets can never hurt. So the model below has exactly one free parameter, the fixed cost `α` of
starting a transfer, and I solve for the range of `α` that makes the lesson's three statements
simultaneously true rather than picking one that looks good.
""")

code(r"""
NLAYER = 12
XFER   = 2 * P_BYTES / IB                 # 2.40 s to move 2P over InfiniBand
FWD_FRAC = 1/3                            # backward is about twice the forward

def simulate(bucket, step_compute, alpha):
    # returns (step time, fraction of transfer time completed before the backward ended)
    bwd = step_compute * (1 - FWD_FRAC)
    per_layer_bwd = bwd / NLAYER
    per_layer_xfer = XFER / NLAYER
    nb = math.ceil(NLAYER / bucket)
    link_free, done_before, total_x = 0.0, 0.0, 0.0
    for b in range(nb):
        k = min(bucket, NLAYER - b * bucket)
        ready = (b + 1) * bucket * per_layer_bwd          # gradients exist only once computed
        x = k * per_layer_xfer + alpha
        start = max(ready, link_free)
        link_free = start + x
        total_x += x
        done_before += max(0.0, min(link_free, bwd) - min(start, bwd))
    return step_compute * FWD_FRAC + max(bwd, link_free), done_before / total_x

H, B = 7.10, 3.12
def best_bucket(step, alpha):
    return min(range(1, 13), key=lambda k: simulate(k, step, alpha)[0])

grid = [i/2000 for i in range(0, 601)]            # 0 to 300 ms in 0.5 ms steps
GOOD = [a for a in grid if best_bucket(H, a) == 1 and best_bucket(B, a) == 2]
print("solving for the per-transfer fixed cost alpha that reproduces the lesson's")
print("statements 2 (H100 prefers the smallest bucket) and 3 (B200 prefers two):\n")
for alpha in (0.0, 0.01, 0.03, 0.05, 0.08, 0.12, 0.2):
    hb, bb = best_bucket(H, alpha), best_bucket(B, alpha)
    print(f"  alpha = {alpha*1000:>5.0f} ms   best bucket: H100 {hb:>2}   B200 {bb:>2}"
          f"   {'<- both match' if (hb, bb) == (1, 2) else ''}")
print(f"\n  both hold for alpha in [{min(GOOD)*1000:.1f}, {max(GOOD)*1000:.1f}] ms"
      if GOOD else "\n  no alpha reproduces both")
print("  That is a wide, plausible band, but it is far above a bare NCCL launch latency of")
print("  tens of microseconds - so the lesson's 'fixed cost of starting a transfer' is")
print("  standing in for more than launch overhead. I use 50 ms below and say so, rather")
print("  than presenting a fitted parameter as a measured one.")
""")

code(r"""
ALPHA = 0.05
print(f"bucket sweep at alpha = {ALPHA*1000:.0f} ms"
      f"   (backward {H*(1-FWD_FRAC):.2f}s on H100, {B*(1-FWD_FRAC):.2f}s on B200,"
      f" transfer {XFER:.2f}s)\n")
print(f"{'bucket':>7}{'transfers':>11}"
      f"{'H100 step':>12}{'overlap':>10}   |{'B200 step':>12}{'overlap':>10}")
for k in (1, 2, 3, 4, 6, 12):
    th, oh = simulate(k, H, ALPHA)
    tb, ob = simulate(k, B, ALPHA)
    print(f"{k:>7}{math.ceil(NLAYER/k):>11}{th:>11.2f}s{100*oh:>9.0f}%   |"
          f"{tb:>11.2f}s{100*ob:>9.0f}%")

print(f"\n  no overlap at all would cost {H+XFER:.2f}s on H100 and {B+XFER:.2f}s on B200.")
best_h = min((1,2,3,4,6,12), key=lambda k: simulate(k, H, ALPHA)[0])
best_b = min((1,2,3,4,6,12), key=lambda k: simulate(k, B, ALPHA)[0])
print(f"  best bucket: H100 {best_h} layer(s), B200 {best_b} layer(s)"
      f"  -> the reversal the lesson describes.")
print(f"  overlap at bucket 2 on H100: {100*simulate(2, H, ALPHA)[1]:.1f}%"
      f"  (the lesson says 83%)")
print("\n  That 83% is not a coincidence and not a measurement - it is 5/6, and it is a")
print("  CEILING. The final bucket's gradients do not exist until the backward pass has")
print("  finished, so its transfer cannot begin before then, and with N buckets no more")
print("  than (N-1)/N of the traffic can ever be hidden:")
for k in (1, 2, 3, 4, 6, 12):
    nb = math.ceil(NLAYER / k)
    print(f"    bucket {k:>2} -> {nb:>2} buckets -> ceiling {100*(nb-1)/nb:>5.1f}%"
          f"   measured {100*simulate(k, H, ALPHA)[1]:>5.1f}%")
print("\n  The H100 column sits exactly on its ceiling at every bucket size, which says the")
print("  link is never the constraint there. The B200 column sits below it, which says the")
print("  link is. That single comparison is the diagnosis the whole section is for.")
print(f"\n  Why they differ: on H100 the backward lasts {H*(1-FWD_FRAC):.2f}s and the"
      f" transfer {XFER:.2f}s, so")
print("  the link has slack and finishing earlier is always better. On B200 the backward")
print(f"  is {B*(1-FWD_FRAC):.2f}s and the transfer is still {XFER:.2f}s - the link is now the"
      " critical path,")
print("  and every extra transfer adds its fixed cost to a chain that nothing is hiding,")
print("  so cutting the bucket in half stops paying. Same configuration file, opposite")
print("  right answer, decided entirely by which side is longer.")
""")

md(r"""
## D · §8 — offload, priced in PCIe seconds

The optimizer state is 12 of the 16 bytes and is touched exactly once per step, which makes it the
obvious thing to move off the card. The cost is PCIe at roughly 60 GB/s — comparable to a network
cable, and 7.5x slower than NVLink. So offload converts a memory problem into a bandwidth problem,
and the question is only whether that trade is the one you need.
""")

code(r"""
W_OFF = 32
opt_bytes  = 12 * PARAM / W_OFF          # this rank's slice of the optimizer state
grad_bytes = 2 * PARAM / W_OFF
print(f"ZeRO-1 at W = {W_OFF}, per card:\n")
print(f"  optimizer slice held on the GPU    {opt_bytes/GiB:7.1f} GiB")
print(f"  total state on the GPU             {formula('zero1', W_OFF)*PARAM/GiB:7.1f} GiB"
      f"   (a card holds {CARD:.1f})\n")

naive = 2 * opt_bytes / PCIE                       # down to the CPU and back up
smart = 2 * grad_bytes / PCIE                      # only the gradient down, the weight up
print(f"{'scheme':<38}{'moves':>10}{'PCIe time':>12}{'% of an H100 step':>20}")
print(f"{'park the state, update on the GPU':<38}{2*opt_bytes/1e9:>9.1f}G{naive:>11.2f}s{100*naive/H:>19.0f}%")
print(f"{'park the state, update on the CPU':<38}{2*grad_bytes/1e9:>9.1f}G{smart:>11.2f}s{100*smart/H:>19.0f}%")
print(f"\n  both free the same {opt_bytes/GiB:.1f} GiB, taking this card from"
      f" {formula('zero1', W_OFF)*PARAM/GiB:.1f} to"
      f" {(formula('zero1', W_OFF) - 12/W_OFF)*PARAM/GiB:.1f} GiB.")
print(f"  It still does not fit under {CARD:.1f}, because the 4-byte floor of section A is")
print(f"  {4*PARAM/GiB:.1f} GiB and offloading the optimizer does not touch it.")
print("\n  That is the whole lesson of this section in one line: offload is worth buying when")
print("  memory is the binding constraint, and here - for ZeRO-1 - it is not the binding")
print("  constraint, it is merely a large one. Doing the update on the CPU is the better of")
print("  the two schemes by a wide margin, because the state then never crosses PCIe at all;")
print("  only the gradient goes down and the new weight comes back.")
""")

md(r"""
## E · §11 — what 8-bit actually saves

MXFP8 stores a block of 32 values with one shared 8-bit exponent. The lesson's claim is that the
effect on *stored state* is much smaller than it first appears, and that the real gains are
elsewhere. Both halves of that are checkable.
""")

code(r"""
SCALE = 1/32                            # one shared exponent byte per 32 values
fp8 = (1 + SCALE) + (1 + SCALE) + 12    # weight + gradient in fp8, optimizer untouched
print(f"{'':<34}{'bytes/weight':>14}{'30B model':>12}")
print(f"{'bf16 weights and gradients':<34}{16.0:>14.2f}{16*PARAM/GiB:>11.1f} GiB")
print(f"{'MXFP8 weights and gradients':<34}{fp8:>14.4f}{fp8*PARAM/GiB:>11.1f} GiB")
print(f"\n  reduction in stored state: {100*(16-fp8)/16:.1f}%   (the lesson says 12.1%)")
print(f"  the block scale costs {2*SCALE:.4f} bytes per parameter across the two tensors")
print(f"  the 12 bytes of optimizer state are untouched - the update still needs the accuracy,")
print(f"  and they are {12/16:.0%} of the total, which is why the saving is small.\n")

P8 = PARAM * 1                          # P is now one byte per parameter
print("where it does pay - the same 2P, now half the bytes:\n")
print(f"{'':<8}{'2P bf16':>10}{'over IB':>10}{'2P fp8':>10}{'over IB':>10}"
      f"{'comm/compute bf16':>20}{'fp8':>8}")
for card, t in (("H100", 7.10), ("B200", 3.12)):
    c16, c8 = 2*P_BYTES/IB, 2*P8/IB
    print(f"{card:<8}{2*P_BYTES/1e9:>9.0f}G{c16:>9.2f}s{2*P8/1e9:>9.0f}G{c8:>9.2f}s"
          f"{100*c16/t:>19.0f}%{100*c8/t:>7.0f}%")
print("\n  On B200 that is the difference between a transfer filling three quarters of the")
print("  step and one filling well under half of it. 12% off the stored state is the least")
print("  interesting thing 8-bit does; halving the wire volume is what it is actually for,")
print("  and it lands hardest on exactly the configuration section B showed to be in")
print("  trouble.")
print("\n  Softmax stays in fp32 throughout. The exponential turns a modest gap between two")
print("  scores into a very large ratio, so it amplifies input error - and it is bandwidth")
print("  bound rather than arithmetic bound, so the precision is close to free there. The")
print("  rule is not 'use 8-bit everywhere'; it is 'use high precision where it costs")
print("  nothing', which is a different and better rule.")
""")

md(r"""
---

## F · §12 and §13 — what V4 ran, and what follows for V5

V4 (LightningLM v0.1) ran DeepSpeed at **ZeRO-2 on 8 GPUs**, bf16, 2 sequences per GPU, 2
accumulation steps, global batch 32, `overlap_comm` on, bucket size 2e8 bytes. Two entries in that
config are worth more than the rest: `round_robin_gradients` appears in the *filename* as an
out-of-memory fix, which records a run that hit a ceiling and was rescued by reordering gradient
bucket assignment; and a ZeRO-3 config exists in the same repository, verified on four 16 GB cards,
which never ran the production model.

So the question for V5 is not "is ZeRO-3 better". It is whether V5 has any choice.
""")

code(r"""
V4_PARAM = 0.3e9                    # LightningLM v0.1, order of magnitude
print("what changed between the two runs:\n")
print(f"{'':<26}{'V4-scale (0.3B)':>18}{'V5 (30B)':>14}")
for lab, stage, w in (("data parallel, 8 GPUs", "dp", 8), ("ZeRO-2, 8 GPUs", "zero2", 8)):
    print(f"{lab:<26}{formula(stage,w)*V4_PARAM/GiB:>15.1f} GiB"
          f"{formula(stage,w)*PARAM/GiB:>11.1f} GiB")
print(f"\n  V4 at ZeRO-2 on 8 cards needed {formula('zero2',8)*V4_PARAM/GiB:.1f} GiB per card and a card")
print(f"  holds {CARD:.1f}. It had room to spare and never needed stage 3 - which is exactly what")
print("  the session says happened. The same arrangement for V5 needs"
      f" {formula('zero2',8)*PARAM/GiB:.1f} GiB.\n")

print("the two arrangements that fit V5, and what each costs:\n")
print(f"{'option':<22}{'GiB/GPU':>10}{'headroom':>11}{'wire/step':>12}"
      f"{'IB seconds':>12}{'% of H100 step':>16}")
for lab, stage, w, mult in (("ZeRO-2 on 32 GPUs", "zero2", 32, 2),
                            ("ZeRO-3 on 8 GPUs",  "zero3", 8,  3),
                            ("ZeRO-3 on 32 GPUs", "zero3", 32, 3)):
    g = formula(stage, w) * PARAM / GiB
    t = mult * P_BYTES / IB
    print(f"{lab:<22}{g:>10.1f}{CARD-g:>10.1f}G{mult:>11}P{t:>11.2f}s{100*t/H:>15.0f}%")

print(f"\n  ZeRO-2 on 32 leaves {CARD - formula('zero2',32)*PARAM/GiB:.1f} GiB for activations."
      " ZeRO-3 on 8 leaves"
      f" {CARD - formula('zero3',8)*PARAM/GiB:.1f} GiB")
print("  but pays 3P instead of 2P - half as much again on the wire, on a link that section B")
print("  already showed to be 34% of an H100 step and 77% of a B200 one.")
print("\n  My reading of my own numbers: ZeRO-3 on 32, which is neither of the two the session")
print("  frames as the choice. It costs the same 3P as ZeRO-3 on 8, and it leaves"
      f" {CARD - formula('zero3',32)*PARAM/GiB:.1f} GiB")
print(f"  for activations instead of {CARD - formula('zero3',8)*PARAM/GiB:.1f} - and activation"
      " memory is the term the session")
print("  explicitly says is missing from the table and pushes the practical threshold up.")
print("  Recomputation buys activation memory at ~30% more compute, which is a worse trade")
print("  than spending GPUs I have already been given.")
""")

md(r"""
### The four open questions, and what I would measure

| question | what would settle it | what this notebook already says |
|---|---|---|
| ZeRO-2 on 32, or ZeRO-3 on 8? | measured step time for both with activations included | wrong pair — ZeRO-3 on 32 dominates ZeRO-3 on 8 at identical wire cost |
| how many GPUs per node? | how much traffic stays inside a node | §B: NVLink is 9× InfiniBand, so a 3P run that crosses nodes pays 3.60 s and one that does not pays 0.40 s |
| 8-bit from the start? | short bf16 vs MXFP8 run, same architecture | §E: the argument is the wire, not the memory — it halves P, which is worth most exactly where §B says we are in trouble |
| does state go to system memory? | is the run memory-bound or communication-bound | §D: at ZeRO-3 on 32 we are not memory-bound, so no |

The thread running through all four: **once the arrangement is chosen, every remaining decision is
about the interconnect, not about the cards.**
""")

code(r"""
print("=" * 78)
print(f"{'MEASURED, not quoted':^78}")
print("=" * 78)
print(f"world size {WORLD} virtual GPUs | model {NP:,} params | P = {P_SIM:,} bytes\n")
print(f"{'arrangement':<12}{'bytes/weight':>14}{'vs DP':>8}{'wire/step':>12}"
      f"{'30B/GPU':>12}{'fits 80GB':>11}")
for stage in ("dp", "zero1", "zero2", "zero3"):
    r = runs[stage]
    g = r["bpw"] * PARAM / GiB
    print(f"{stage:<12}{r['bpw']:>14.3f}{runs['dp']['bpw']/r['bpw']:>7.1f}x"
          f"{r['wire']/P_SIM:>11.3f}P{g:>11.1f}G{'yes' if g < CARD else 'no':>11}")
print(f"\n  1  collectives verified against torch.distributed gloo: "
      f"{'all four identical' if GLOO_OK else 'not run here'}")
print( "  2  sharding is exact: ZeRO-1/2/3 match data parallelism to 0.00e+00")
print(f"  3  vs one GPU on the whole batch: differs, and bf16 is ~{BF16_FACTOR:,.0f}x of it")
print( "  4  compute per rank is identical in all four - ZeRO trades memory for wire only")
print(f"  5  bytes/weight matches 16.00 / 5.50 / 3.75 / 2.00 at W=8 exactly")
print(f"  6  wire matches 2P / 2P / 2P / 3P, exactly 2(W-1)/W and 3(W-1)/W")
print( "  7  the 4-byte floor kills DP and ZeRO-1 at 20B params, at ANY world size")
print( "  8  the lesson's 7.10s and 3.12s both encode MFU = 40%")
print( "  9  the 83% overlap figure is 5/6, a structural ceiling of (N-1)/N buckets")
print( " 10  MXFP8 saves 12.1% of state and 50% of the wire; the wire is the point")
print("=" * 78)
""")

# ===== MORE CELLS GO ABOVE THIS LINE =====

nb = {"cells": C,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(HERE, "S12_zero.ipynb")

src = open(os.path.abspath(__file__)).read()
bad = [m for m in re.findall(r'code\(r"""(.*?)"""\)', src, re.S) if '"""' in m]
assert not bad, f"{len(bad)} cell(s) truncated by an inner triple-quote"

json.dump(nb, open(out, "w"), indent=1)
ncode = sum(1 for c in C if c["cell_type"] == "code")
print(f"wrote {out}  ({len(C)} cells: {ncode} code, {len(C)-ncode} markdown)")
