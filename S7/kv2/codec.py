"""Kronecker byte codecs.

KroneckerV1  - faithful reimplementation of the shipped scheme (Session 7 §7):
                 kappa(b) = (1/sqrt(L)) * vec( sum_p  c[byte_p] (x) p[pos_p] )
               with a 256 x 32 grid, L = min(len(bytes), pos_dim). Bytes past
               position 32 are silently dropped.

ElasticKronecker (this work) - same Kronecker idea (one-hot value (x) one-hot
               position, frozen, never trained) but the POSITION factor becomes
               elastic, so a token of any length is represented without
               truncation and the code is smaller.

Both expose .encode(token_text) -> np.ndarray of fixed dim, and are pure
functions: same text always gives the same vector.
"""
from __future__ import annotations
import numpy as np

BYTE_VALUES = 256


def _znorm(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-token z-normalisation, matching the reference implementation exactly:
    torch's .std() is Bessel-corrected (ddof=1) and eps is ADDED to the std
    (codec.py: `std = (out - mean).std(dim=-1, keepdim=True) + eps`)."""
    mean = v.mean()
    std = v.std(ddof=1) + eps
    return (v - mean) / std


def utf8_safe_len(b: bytes, cap: int) -> int:
    """Paper §3.2: 'if d_p falls in the middle of a multi-byte codepoint, we back
    off to the previous codepoint boundary'."""
    if len(b) <= cap:
        return len(b)
    L = cap
    while L > 0 and (b[L] & 0xC0) == 0x80:          # 0b10xxxxxx = continuation byte
        L -= 1
    return L


class KroneckerV1:
    """Faithful reimplementation of Kronecker Embeddings V1 (Shravan, 2026), §3.2-3.3:

        kappa(b) = (1/sqrt(L)) * sum_p  c[b_p] (x) p[p],     L <= d_p
        coordinate of (b_p, p) is  b_p * d_p + p
        d_c = 256 (full byte alphabet), production d_p = 32
        then per-token z-normalisation to mean 0, std 1
        tokens longer than d_p are truncated with UTF-8-safe back-off
    """

    name = "kronecker-v1"

    def __init__(self, pos_dim: int = 32, char_dim: int = BYTE_VALUES):
        self.pos_dim, self.char_dim = pos_dim, char_dim
        self.dim = char_dim * pos_dim

    def grid(self, token: str) -> np.ndarray:
        b = token.encode("utf-8")
        L = utf8_safe_len(b, self.pos_dim)          # <-- the truncation, per paper
        g = np.zeros((self.char_dim, self.pos_dim), dtype=np.float32)
        for p in range(L):
            g[b[p], p] += 1.0
        if L > 0:
            g /= np.sqrt(L)
        return g

    def encode(self, token: str) -> np.ndarray:
        return _znorm(self.grid(token).reshape(-1))

    def dropped_bytes(self, token: str) -> int:
        b = token.encode("utf-8")
        return len(b) - utf8_safe_len(b, self.pos_dim)

    # ---- reference-API aliases (embedding.KroneckerEmbedding) ----
    @property
    def D(self) -> int:
        return self.dim

    def extra_repr(self) -> str:
        return (f"char_dim={self.char_dim}, pos_dim={self.pos_dim}, D={self.D}")


class ElasticKronecker:
    """Elastic position axis: HEAD (absolute prefix) + TAIL (absolute suffix)
    + MID (relative, unbounded) + LEN (explicit length signature).

    Why each piece exists
      HEAD  keeps the property that made Kronecker attractive: tokens sharing a
            prefix (train / training / trainer) start out near each other.
      TAIL  anchors the suffix. Agglutinative Indic morphology lives at the end
            of the word, and it is exactly what V1 truncates away first.
      MID   absorbs everything between head and tail at *relative* resolution,
            so a token of any byte length lands somewhere - nothing is dropped.
            Long tokens degrade gracefully (several bytes superpose in a bin)
            instead of failing silently (bytes deleted).
      LEN   a log-scaled length signature, so two tokens that agree on head,
            tail and middle distribution but differ in length still differ.

    Cost: 256*(H+T+M) + len_dim. With H=T=M=8 that is 6,144 + 32 = 6,176 dims
    against V1's 8,192 - a 24.6% smaller code AND no truncation at all.
    """

    name = "elastic-kronecker"

    def __init__(self, head: int = 8, tail: int = 8, mid: int = 8,
                 len_dim: int = 32, char_dim: int = BYTE_VALUES,
                 len_weight: float = 0.35):
        self.head, self.tail, self.mid = head, tail, mid
        self.len_dim, self.char_dim = len_dim, char_dim
        self.pos_bins = head + tail + mid
        # +char_dim for the ORDER MOMENT plane (see _order_moment)
        self.dim = char_dim * self.pos_bins + char_dim + len_dim
        # The length signature is dense (|values| ~ 1) while the grid is sparse
        # (std ~ 0.01). Concatenating them raw and z-norming globally lets 32
        # length dims outweigh 6,144 grid dims and destroys the spelling
        # geometry. So: normalise the grid on its own, then add the length
        # channel at a controlled weight - big enough to break ties, small
        # enough to leave the spelling signal in charge.
        self.len_weight = len_weight

    # ---- position assignment: which bin(s) does byte p of a length-N token hit ----
    def _bins(self, p: int, N: int) -> int:
        """Absolute for the first `head` and last `tail` bytes, relative in between."""
        if p < self.head:
            return p                                        # HEAD block
        if p >= N - self.tail:
            return self.head + (self.tail - (N - p))         # TAIL block
        span = N - self.head - self.tail                     # middle length (>0 here)
        frac = (p - self.head) / max(span, 1)
        k = min(self.mid - 1, int(frac * self.mid))
        return self.head + self.tail + k                     # MID block

    def grid(self, token: str) -> np.ndarray:
        b = token.encode("utf-8")
        N = len(b)
        g = np.zeros((self.char_dim, self.pos_bins), dtype=np.float32)
        for p in range(N):                                   # <-- every byte, no cap
            g[b[p], self._bins(p, N)] += 1.0
        if N > 0:
            g /= np.sqrt(N)
        return g

    def _length_signature(self, N: int) -> np.ndarray:
        """Smooth log-length code: unbounded, and distinguishes 'a' from 'aa'."""
        v = np.zeros(self.len_dim, dtype=np.float32)
        if N == 0:
            return v
        x = np.log1p(N)
        for k in range(self.len_dim // 2):
            w = (k + 1) * 0.7
            v[2 * k] = np.sin(x * w)
            v[2 * k + 1] = np.cos(x * w)
        return v

    def _order_moment(self, token: str) -> np.ndarray:
        """Mean relative position of each byte value.

        Binning the middle at relative resolution means several bytes can share
        one bin, and a plain count grid is blind to their ORDER: swapping two
        bytes inside one bin leaves the grid identical. This plane records, per
        byte value, the mean position at which it occurred, so a transposition
        changes the code. It costs 256 dims and removes the only collision mode
        the elastic scheme has.
        """
        b = token.encode("utf-8")
        N = len(b)
        m = np.zeros(self.char_dim, dtype=np.float32)
        if N == 0:
            return m
        cnt = np.zeros(self.char_dim, dtype=np.float32)
        for p, byte in enumerate(b):
            m[byte] += (p + 0.5) / N
            cnt[byte] += 1.0
        nz = cnt > 0
        m[nz] /= cnt[nz]
        return m

    def encode(self, token: str) -> np.ndarray:
        gz = _znorm(self.grid(token).reshape(-1))
        om = self._order_moment(token) * self.len_weight
        ls = self._length_signature(len(token.encode("utf-8"))) * self.len_weight
        return np.concatenate([gz, om, ls])

    def dropped_bytes(self, token: str) -> int:
        return 0                                             # by construction

    # ---- reference-API aliases: ElasticKronecker is a drop-in sibling of
    # KroneckerEmbedding, so it can be swapped in the reference package by
    # changing one constructor call.
    @property
    def D(self) -> int:
        return self.dim

    def extra_repr(self) -> str:
        return (f"char_dim={self.char_dim}, head={self.head}, tail={self.tail}, "
                f"mid={self.mid}, order={self.char_dim}, len_dim={self.len_dim}, "
                f"D={self.D}")


def code_key(vec: np.ndarray, places: int = 6) -> bytes:
    """Hashable identity of a code, for exact collision counting."""
    return np.round(vec, places).tobytes()
