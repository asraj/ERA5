"""Literal numpy transcription of the reference implementation.

Source: github.com/theschoolofai/kronecker-embeddings
        src/kronecker_embeddings/codec.py           (kronecker_codec, encode_single)
        src/kronecker_embeddings/tokenizer_utils.py (utf8_safe_truncate)
        src/kronecker_embeddings/embedding.py       (KroneckerEmbedding)

torch is not available in this environment, so this is a line-by-line port of the
reference math to numpy. It exists for ONE purpose: to be an independent oracle
that `kv2.codec.KroneckerV1` is checked against, so our baseline is provably the
published method and not our reading of it.

Reference details that are easy to get wrong and are reproduced exactly here:
  * linear index = byte_value * pos_dim + pos          ("Storage convention matches production")
  * scales = rsqrt(clamp_min(L, 1))                    (guards L=0)
  * z-norm uses torch .std(), which is UNBIASED (ddof=1), and eps is ADDED
    to the std, not folded under a sqrt:  (x-mean)/(std_ddof1 + 1e-6)
  * projection: Linear(D, d_model, bias=False), init normal(0, 1/sqrt(D))
"""
from __future__ import annotations
import numpy as np


def utf8_safe_truncate(b: bytes, cap: int) -> bytes:
    """tokenizer_utils.utf8_safe_truncate: never cut a multi-byte codepoint."""
    if len(b) <= cap:
        return b
    L = cap
    while L > 0 and (b[L] & 0xC0) == 0x80:      # 0b10xxxxxx continuation byte
        L -= 1
    return b[:L]


def codec_output_dim(char_dim: int = 256, pos_dim: int = 32) -> int:
    return char_dim * pos_dim


def kronecker_codec(byte_sequences, lengths, char_dim: int = 256, pos_dim: int = 32,
                    length_normalize: bool = True, z_normalize: bool = True,
                    eps: float = 1e-6) -> np.ndarray:
    """Port of codec.kronecker_codec. byte_sequences: (B, pos_dim) uint8."""
    byte_sequences = np.asarray(byte_sequences)
    if byte_sequences.ndim != 2:
        raise ValueError("byte_sequences must be (B, pos_dim)")
    if byte_sequences.shape[1] != pos_dim:
        raise ValueError("byte_sequences.shape[1] != pos_dim")
    B = byte_sequences.shape[0]
    D = char_dim * pos_dim

    bytes_long = byte_sequences.astype(np.int64)
    lens_long = np.asarray(lengths).astype(np.int64)
    pos = np.broadcast_to(np.arange(pos_dim), (B, pos_dim))

    lin_idx = bytes_long * pos_dim + pos                     # byte_value * pos_dim + pos
    valid = pos < lens_long[:, None]

    if length_normalize:
        scales = 1.0 / np.sqrt(np.maximum(lens_long, 1).astype(np.float32))
        src = valid.astype(np.float32) * scales[:, None]
    else:
        src = valid.astype(np.float32)

    out = np.zeros((B, D), dtype=np.float32)
    for i in range(B):                                        # == Tensor.scatter_add_
        np.add.at(out[i], lin_idx[i], src[i])

    if z_normalize:
        mean = out.mean(axis=-1, keepdims=True)
        # torch's .std() is Bessel-corrected (ddof=1); eps is ADDED to it.
        std = (out - mean).std(axis=-1, keepdims=True, ddof=1) + eps
        out = (out - mean) / std
    return out


def encode_single(byte_seq: bytes, char_dim: int = 256, pos_dim: int = 32,
                  length_normalize: bool = True, z_normalize: bool = True,
                  eps: float = 1e-6) -> np.ndarray:
    """Port of codec.encode_single. NOTE: like the reference, this does NOT
    UTF-8-safe-truncate; the production path truncates when building the byte
    buffer, which is why our KroneckerV1 truncates safely."""
    L = min(len(byte_seq), pos_dim)
    buf = np.zeros(pos_dim, dtype=np.uint8)
    if L > 0:
        buf[:L] = np.frombuffer(bytearray(byte_seq[:L]), dtype=np.uint8)
    return kronecker_codec(buf[None, :], np.array([L]), char_dim, pos_dim,
                           length_normalize, z_normalize, eps)[0]


def build_byte_buffer(tokens, pos_dim: int = 32):
    """Port of tokenizer_utils.build_byte_buffer: (V, pos_dim) uint8 + (V,) int16."""
    V = len(tokens)
    bb = np.zeros((V, pos_dim), dtype=np.uint8)
    lb = np.zeros(V, dtype=np.int16)
    for i, t in enumerate(tokens):
        raw = utf8_safe_truncate(t.encode("utf-8"), pos_dim)
        bb[i, :len(raw)] = np.frombuffer(bytearray(raw), dtype=np.uint8)
        lb[i] = len(raw)
    return bb, lb


class KroneckerEmbedding:
    """Port of embedding.KroneckerEmbedding (numpy). Same constructor surface,
    same forward contract: (..., L) ids -> (..., L, d_model)."""

    def __init__(self, vocab_size: int, d_model: int, tokens=None, char_dim: int = 256,
                 pos_dim: int = 32, mode: str = "dynamic", byte_buffer=None,
                 length_buffer=None, projection_init: str = "normal",
                 length_normalize: bool = True, z_normalize: bool = True, seed: int = 0):
        if mode not in ("dynamic", "cached"):
            raise ValueError("mode must be 'dynamic' or 'cached'")
        self.vocab_size, self.d_model = vocab_size, d_model
        self.char_dim, self.pos_dim = char_dim, pos_dim
        self.D = codec_output_dim(char_dim, pos_dim)
        self.mode = mode
        self.length_normalize, self.z_normalize = length_normalize, z_normalize
        if byte_buffer is None or length_buffer is None:
            if tokens is None:
                raise ValueError("pass tokens or both byte_buffer and length_buffer")
            byte_buffer, length_buffer = build_byte_buffer(tokens, pos_dim)
        self._byte_buffer, self._length_buffer = byte_buffer, length_buffer
        rng = np.random.default_rng(seed)
        if projection_init == "normal":
            self.projection = rng.normal(0.0, 1.0 / np.sqrt(self.D), (self.D, d_model))
        else:
            lim = np.sqrt(6.0 / (self.D + d_model))
            self.projection = rng.uniform(-lim, lim, (self.D, d_model))
        if mode == "cached":
            self._codec_table = kronecker_codec(
                self._byte_buffer, self._length_buffer, char_dim, pos_dim,
                length_normalize, z_normalize)

    @property
    def num_embeddings(self): return self.vocab_size

    @property
    def embedding_dim(self): return self.d_model

    def _codec_lookup(self, input_ids: np.ndarray) -> np.ndarray:
        flat = np.asarray(input_ids).reshape(-1)
        if self.mode == "cached":
            out = self._codec_table[flat]
        else:
            out = kronecker_codec(self._byte_buffer[flat], self._length_buffer[flat],
                                  self.char_dim, self.pos_dim,
                                  self.length_normalize, self.z_normalize)
        return out.reshape(*np.asarray(input_ids).shape, self.D)

    def forward(self, input_ids) -> np.ndarray:
        return self._codec_lookup(input_ids) @ self.projection

    __call__ = forward

    def extra_repr(self) -> str:
        return (f"vocab_size={self.vocab_size}, d_model={self.d_model}, "
                f"char_dim={self.char_dim}, pos_dim={self.pos_dim}, "
                f"D={self.D}, mode={self.mode!r}")
