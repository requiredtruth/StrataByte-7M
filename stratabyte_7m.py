#!/usr/bin/env python3
"""
StrataByte-7M
===========

A single-file, CPU-first byte-level language-model trainer and PySide6
playground.  Its only non-standard dependencies are PyTorch and PySide6.
There is no tokenizer: every UTF-8 byte is a token in a fixed vocabulary of
256 values.

Run:
    python -m pip install torch PySide6
    python stratabyte_7m.py

Optional verification:
    python stratabyte_7m.py --self-test

The default model is intentionally small enough for local CPU experiments.
Edit ModelConfig (or load a checkpoint carrying another ModelConfig) to scale
between roughly 3M and 10M parameters.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import copy
import io
import math
import mmap
import os
import queue
import random
import sys
import threading
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - depends on the local machine
    raise SystemExit(
        "PySide6 is required for the desktop interface. Install it with: "
        "python -m pip install PySide6"
    ) from exc

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - depends on the local machine
    raise SystemExit(
        "PyTorch is required. Install a CPU build from https://pytorch.org/ "
        "and run this file again."
    ) from exc


APP_NAME = "StrataByte-7M"
FORMAT_VERSION = 1
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)

# A deliberately small public-domain fallback keeps the application runnable
# when the first-run download is blocked. The downloader is always attempted
# first; this text is not used when TinyShakespeare is available.
FALLBACK_CORPUS = ("""
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

HAMLET:
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name;
Or, if thou wilt not, be but sworn my love.

MACBETH:
Tomorrow, and tomorrow, and tomorrow,
Creeps in this petty pace from day to day.

PUCK:
If we shadows have offended,
Think but this, and all is mended.
""" * 512).encode("utf-8")


def application_data_dir() -> Path:
    """Return a writable, platform-appropriate application data directory."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
        path = base / "StrataByte-7M"
    elif sys.platform == "darwin":
        path = Path.home() / "Library" / "Application Support" / "StrataByte-7M"
    else:
        path = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        path = path / "stratabyte-7m"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dataset(status: Optional[Callable[[str], None]] = None) -> Path:
    """Download TinyShakespeare atomically on first use, with an offline fallback."""
    data_dir = application_data_dir()
    destination = data_dir / "tinyshakespeare.txt"
    if destination.exists() and destination.stat().st_size >= 10_000:
        if status:
            status(f"Dataset ready: {destination.name}")
        return destination

    partial = destination.with_suffix(".download")
    if status:
        status("Downloading TinyShakespeare…")
    try:
        request = urllib.request.Request(
            TINY_SHAKESPEARE_URL,
            headers={"User-Agent": "StrataByte-7M/1.0"},
        )
        with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or 0)
            received = 0
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                out.write(block)
                received += len(block)
                if status and total:
                    status(f"Downloading TinyShakespeare… {100.0 * received / total:.0f}%")
        if partial.stat().st_size < 10_000:
            raise OSError("downloaded corpus is unexpectedly small")
        os.replace(partial, destination)
        if status:
            status(f"Downloaded {destination.stat().st_size / 1_000_000:.2f} MB")
    except Exception as exc:
        try:
            partial.unlink(missing_ok=True)
        except OSError:
            pass
        destination.write_bytes(FALLBACK_CORPUS)
        if status:
            status(
                "Dataset download unavailable; using the bundled public-domain "
                f"fallback ({exc.__class__.__name__})."
            )
    return destination


class StreamingByteLoader:
    """File-backed random byte-stream batches with no tokenization or vocabulary build."""

    def __init__(self, path: Path, batch_size: int, sequence_length: int, seed: int = 1337):
        if sequence_length < 8:
            raise ValueError("sequence_length must be at least 8")
        self.path = path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self._rng = random.Random(seed)
        self._file = path.open("rb")
        self._size = path.stat().st_size
        if self._size < sequence_length + 2:
            self._file.close()
            raise ValueError("dataset is shorter than one requested training sequence")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def next_batch(self) -> Tuple[Tensor, Tensor]:
        span = self.sequence_length + 1
        last_start = self._size - span
        chunks: List[bytes] = []
        for _ in range(self.batch_size):
            start = self._rng.randint(0, last_start)
            chunks.append(self._mmap[start:start + span])
        # bytearray provides writable storage, avoiding PyTorch's non-writable
        # buffer warning while retaining a single compact conversion.
        flat = torch.frombuffer(bytearray(b"".join(chunks)), dtype=torch.uint8)
        batch = flat.reshape(self.batch_size, span).to(dtype=torch.long)
        return batch[:, :-1], batch[:, 1:]

    def close(self) -> None:
        self._mmap.close()
        self._file.close()

    def __enter__(self) -> "StreamingByteLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    patch_size: int = 4
    d_model: int = 192
    n_heads: int = 8
    n_layers: int = 3
    q_rank: int = 64
    kv_rank: int = 64
    expert_hidden: int = 256
    n_experts: int = 8
    default_top_k: int = 2
    dropout: float = 0.05
    max_recurrence: int = 8

    def validate(self) -> None:
        if self.vocab_size != 256:
            raise ValueError("byte vocabulary must contain exactly 256 values")
        if self.patch_size != 4:
            raise ValueError("this byte sandwich uses a fixed patch factor K=4")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads < 8:
            raise ValueError("at least 8 heads are required for all 1/2/4 staggered phases")
        if (self.d_model // self.n_heads) % 2:
            raise ValueError("attention head dimension must be even for rotary positions")
        if not (0 < self.q_rank < self.d_model and 0 < self.kv_rank < self.d_model):
            raise ValueError("Q and KV ranks must be positive low-rank bottlenecks")
        if self.n_layers < 1 or self.expert_hidden < 1 or self.max_recurrence < 1:
            raise ValueError("layer, expert-hidden, and recurrence sizes must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not (1 <= self.default_top_k <= self.n_experts):
            raise ValueError("default_top_k must be within the expert count")
        if self.n_experts != 8:
            raise ValueError("the requested architecture uses exactly 8 routed experts")


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(dtype=x.dtype) * self.weight.to(dtype=x.dtype)


def apply_rotary(x: Tensor) -> Tensor:
    """Apply RoPE to [batch, heads, time, head_dim] without a length limit."""
    _, _, length, width = x.shape
    half = width // 2
    positions = torch.arange(length, device=x.device, dtype=torch.float32)
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=x.device, dtype=torch.float32)
        / max(1, half)
    )
    angles = positions[:, None] * frequencies[None, :]
    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]
    first = x[..., :half].float()
    second = x[..., half:2 * half].float()
    rotated = torch.cat(
        (first * cosine - second * sine, first * sine + second * cosine), dim=-1
    )
    if width > 2 * half:
        rotated = torch.cat((rotated, x[..., 2 * half:].float()), dim=-1)
    return rotated.to(dtype=x.dtype)


class CompressedDifferentialAttention(nn.Module):
    """Low-rank MLA projections, differential cancellation, and head gating."""

    EPSILON = 1e-6

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dimension = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout

        self.q_down = nn.Linear(config.d_model, config.q_rank, bias=False)
        self.q1_up = nn.Linear(config.q_rank, config.d_model, bias=False)
        self.q2_up = nn.Linear(config.q_rank, config.d_model, bias=False)

        self.kv_down = nn.Linear(config.d_model, config.kv_rank, bias=False)
        self.k1_up = nn.Linear(config.kv_rank, config.d_model, bias=False)
        self.k2_up = nn.Linear(config.kv_rank, config.d_model, bias=False)
        self.v_up = nn.Linear(config.kv_rank, config.d_model, bias=False)

        self.gate = nn.Linear(config.d_model, config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        initial_lambda = 0.8
        raw = math.log(initial_lambda / (1.0 - initial_lambda))
        self.lambda_raw = nn.Parameter(torch.full((config.n_heads,), raw))
        self._mask_cache: Dict[Tuple[int, str], Tensor] = {}

        patterns = [(1, 0), (1, 0), (2, 0), (2, 1),
                    (4, 0), (4, 1), (4, 2), (4, 3)]
        if self.n_heads <= len(patterns):
            selected = patterns[:self.n_heads]
        else:
            selected = [patterns[index % len(patterns)] for index in range(self.n_heads)]
        self.register_buffer(
            "head_strides", torch.tensor([item[0] for item in selected], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "head_offsets", torch.tensor([item[1] for item in selected], dtype=torch.long),
            persistent=False,
        )

    def _causal_multiresolution_mask(self, length: int, device: torch.device) -> Tensor:
        cache_key = (length, str(device))
        cached = self._mask_cache.get(cache_key)
        if cached is not None:
            return cached
        query_index = torch.arange(length, device=device)[:, None]
        key_index = torch.arange(length, device=device)[None, :]
        # Strict block-causality: attention itself excludes both the diagonal
        # and every future patch. The residual stream still carries patch m's
        # own encoded state, so excluding self-attention loses no local signal.
        causal = key_index < query_index
        per_head: List[Tensor] = []
        for stride_tensor, offset_tensor in zip(self.head_strides, self.head_offsets):
            stride = int(stride_tensor.item())
            offset = int(offset_tensor.item())
            staggered = key_index.remainder(stride) == offset
            per_head.append(causal & staggered)
        mask = torch.stack(per_head, dim=0)
        if len(self._mask_cache) >= 12:
            self._mask_cache.clear()
        self._mask_cache[cache_key] = mask
        return mask

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        q_compressed = self.q_down(x)
        kv_compressed = self.kv_down(x)
        q1 = apply_rotary(self._split_heads(self.q1_up(q_compressed)))
        q2 = apply_rotary(self._split_heads(self.q2_up(q_compressed)))
        k1 = apply_rotary(self._split_heads(self.k1_up(kv_compressed)))
        k2 = apply_rotary(self._split_heads(self.k2_up(kv_compressed)))
        value = self._split_heads(self.v_up(kv_compressed))

        scale = self.head_dim ** -0.5
        scores1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1)) * scale
        scores2 = torch.matmul(q2.float(), k2.float().transpose(-2, -1)) * scale
        allowed = self._causal_multiresolution_mask(x.shape[1], x.device)[None, :, :, :]
        probability1 = F.softmax(scores1.masked_fill(~allowed, -1.0e9), dim=-1)
        probability2 = F.softmax(scores2.masked_fill(~allowed, -1.0e9), dim=-1)
        allowed_float = allowed.to(dtype=probability1.dtype)
        probability1 = probability1 * allowed_float
        probability2 = probability2 * allowed_float
        probability1 = probability1 / (
            probability1.sum(dim=-1, keepdim=True) + self.EPSILON
        )
        probability2 = probability2 / (
            probability2.sum(dim=-1, keepdim=True) + self.EPSILON
        )

        # sigmoid keeps 0 < lambda < 1 and starts at exactly 0.8.  Normalizing
        # by (1-lambda+eps) keeps the signed differential weights centered at a
        # unit sum while EPSILON prevents a singular denominator.
        differential_lambda = torch.sigmoid(self.lambda_raw)[None, :, None, None]
        denominator = 1.0 - differential_lambda + self.EPSILON
        weights = (probability1 - differential_lambda * probability2) / denominator
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        attended = torch.matmul(weights, value.float()).to(dtype=x.dtype)
        head_gate = torch.sigmoid(self._split_heads(self.gate(x)))
        attended = attended * head_gate
        merged = attended.transpose(1, 2).contiguous().reshape(x.shape)
        return self.output(merged)


class SwiGLUExpert(nn.Module):
    def __init__(self, dimension: int, hidden: int):
        super().__init__()
        self.input = nn.Linear(dimension, hidden * 2)
        self.output = nn.Linear(hidden, dimension)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class SparseMixtureOfExperts(nn.Module):
    """Eight token-routed experts (Top-2 by default) plus one shared expert."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.default_top_k = config.default_top_k
        self.router = nn.Linear(config.d_model, config.n_experts)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config.d_model, config.expert_hidden)
             for _ in range(config.n_experts)]
        )
        self.shared_expert = SwiGLUExpert(config.d_model, config.expert_hidden)

    def reset_router(self) -> None:
        # Near-equal probabilities plus tiny symmetry-breaking noise distribute
        # the first batch instead of sending every tied token to the same pair.
        nn.init.normal_(self.router.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.router.bias)

    def forward(self, x: Tensor, top_k: Optional[int] = None) -> Tuple[Tensor, Tensor, Tensor]:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        routed_k = self.default_top_k if top_k is None else int(top_k)
        routed_k = max(1, min(routed_k, self.n_experts))

        router_logits = self.router(flat).float()
        full_probabilities = F.softmax(router_logits, dim=-1)
        top_values, top_indices = torch.topk(router_logits, k=routed_k, dim=-1)
        top_weights = F.softmax(top_values, dim=-1)

        # Cast routed results back to the residual-stream dtype. This keeps CPU
        # bfloat16 autocast compatible with index_add, which requires identical
        # source and destination dtypes.
        output = self.shared_expert(flat).to(dtype=flat.dtype)
        for expert_index, expert in enumerate(self.experts):
            selected = (top_indices == expert_index).nonzero(as_tuple=False)
            if selected.numel() == 0:
                continue
            token_indices = selected[:, 0]
            route_slots = selected[:, 1]
            expert_input = flat.index_select(0, token_indices)
            contribution = expert(expert_input).to(dtype=flat.dtype)
            route_weight = top_weights[token_indices, route_slots, None].to(
                dtype=contribution.dtype
            )
            contribution = contribution * route_weight
            output = output.index_add(0, token_indices, contribution)

        # f_i is the actual fraction of Top-k dispatches; P_i is the average
        # full router probability. f is non-differentiable and intentionally
        # detached, while P carries the balancing gradient into the router.
        dispatch = F.one_hot(top_indices, num_classes=self.n_experts).float()
        fraction = dispatch.sum(dim=(0, 1)) / float(flat.shape[0] * routed_k)
        mean_probability = full_probabilities.mean(dim=0)
        balance_core = self.n_experts * torch.sum(fraction.detach() * mean_probability)

        # The training objective applies the requested coefficient 1e-4 once.
        router_z_core = torch.logsumexp(router_logits, dim=-1).square().mean()
        return output.reshape(original_shape), balance_core, router_z_core


class CoreBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model)
        self.attention = CompressedDifferentialAttention(config)
        self.expert_norm = RMSNorm(config.d_model)
        self.experts = SparseMixtureOfExperts(config)
        self.dropout = config.dropout

    def forward(self, x: Tensor, top_k: Optional[int]) -> Tuple[Tensor, Tensor, Tensor]:
        x = x + F.dropout(
            self.attention(self.attention_norm(x)), self.dropout, self.training
        )
        routed, balance, router_z = self.experts(self.expert_norm(x), top_k)
        x = x + F.dropout(routed, self.dropout, self.training)
        return x, balance, router_z


class LocalPatchEncoder(nn.Module):
    """Non-overlapping learned local compressor with downsampling factor K=4."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.convolution = nn.Conv1d(
            config.d_model,
            config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.mean_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)

    def forward(self, byte_states: Tensor) -> Tuple[Tensor, Tensor, int]:
        batch, length, dimension = byte_states.shape
        padding = (-length) % self.patch_size
        if padding:
            padded = F.pad(byte_states, (0, 0, 0, padding))
        else:
            padded = byte_states
        convolution = self.convolution(padded.transpose(1, 2)).transpose(1, 2)
        patches = padded.reshape(batch, -1, self.patch_size, dimension)
        pooled = patches.mean(dim=2)
        encoded = self.norm(convolution + self.mean_projection(pooled))
        return encoded, padded, padding


class StrictLocalCausalAttention(nn.Module):
    """Within-patch self-attention where position k sees only positions < k."""

    def __init__(self, dimension: int, n_heads: int, dropout: float):
        super().__init__()
        if dimension % n_heads:
            raise ValueError("local attention dimension must divide n_heads")
        self.n_heads = n_heads
        self.head_dim = dimension // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, dimension = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def split(value: Tensor) -> Tensor:
            return value.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split(q), split(k), split(v)
        scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
        scores = scores * (self.head_dim ** -0.5)
        row = torch.arange(length, device=x.device)[:, None]
        column = torch.arange(length, device=x.device)[None, :]
        allowed = (column < row)[None, None, :, :]  # strict: diagonal is excluded
        scores = scores.masked_fill(~allowed, -1.0e9)
        probability = F.softmax(scores, dim=-1) * allowed.to(dtype=scores.dtype)
        probability = probability / (probability.sum(dim=-1, keepdim=True) + 1e-6)
        probability = F.dropout(probability, self.dropout, self.training)
        attended = torch.matmul(probability, v.float()).to(dtype=x.dtype)
        merged = attended.transpose(1, 2).contiguous().reshape(batch, length, dimension)
        return self.output(merged)


class LocalPatchDecoder(nn.Module):
    """Strict local byte decoder cross-attending to core latent + encoder skip."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.dimension = config.d_model
        self.offset_embedding = nn.Parameter(
            torch.empty(config.patch_size, config.d_model)
        )
        self.bos_latent = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.bos_skip = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.local_norm = RMSNorm(config.d_model)
        self.local_attention = StrictLocalCausalAttention(
            config.d_model, config.n_heads, config.dropout
        )
        self.cross_norm = RMSNorm(config.d_model)
        self.memory_norm = RMSNorm(config.d_model)
        self.cross_attention = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUExpert(config.d_model, config.d_model * 2)
        self.output_norm = RMSNorm(config.d_model)
        self.dropout = config.dropout
        nn.init.normal_(self.offset_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        padded_byte_states: Tensor,
        encoded_patches: Tensor,
        core_patches: Tensor,
        original_length: int,
    ) -> Tensor:
        batch, padded_length, dimension = padded_byte_states.shape
        patch_count = padded_length // self.patch_size
        local = padded_byte_states.reshape(batch, patch_count, self.patch_size, dimension)
        local = local + self.offset_embedding[None, None, :, :]
        local = local.reshape(batch * patch_count, self.patch_size, dimension)

        # The residual query at byte k contains the preceding observed language
        # byte for next-byte prediction; its self-attention context is strictly
        # positions < k. Thus no same-position or future-byte attention exists.
        local = local + F.dropout(
            self.local_attention(self.local_norm(local)), self.dropout, self.training
        )

        # Decoder patch m receives z_m formed from completed patch m-1. This
        # one-patch shift prevents the patch compressor from leaking any byte
        # that the decoder is currently predicting. The second memory token is
        # the requested long encoder-to-decompressor precision skip.
        latent_memory = torch.cat(
            (self.bos_latent.expand(batch, -1, -1), core_patches[:, :-1, :]), dim=1
        )
        skip_memory = torch.cat(
            (self.bos_skip.expand(batch, -1, -1), encoded_patches[:, :-1, :]), dim=1
        )
        memory = torch.stack((latent_memory, skip_memory), dim=2)
        memory = memory.reshape(batch * patch_count, 2, dimension)
        normalized_memory = self.memory_norm(memory)
        cross, _ = self.cross_attention(
            self.cross_norm(local), normalized_memory, normalized_memory,
            need_weights=False,
        )
        local = local + F.dropout(cross, self.dropout, self.training)
        local = local + F.dropout(self.ffn(self.ffn_norm(local)), self.dropout, self.training)
        decoded = self.output_norm(local)
        decoded = decoded.reshape(batch, padded_length, dimension)
        return decoded[:, :original_length, :]


class HierarchicalByteModel(nn.Module):
    """Byte encoder -> causal patch core -> autoregressive byte decompressor."""

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        config.validate()
        self.config = config
        self.byte_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = LocalPatchEncoder(config)
        self.core_blocks = nn.ModuleList([CoreBlock(config) for _ in range(config.n_layers)])
        self.recurrent_block = CoreBlock(config)  # shared weights at every recurrence step
        self.recurrence_embedding = nn.Parameter(
            torch.zeros(config.max_recurrence, config.d_model)
        )
        self.core_norm = RMSNorm(config.d_model)
        self.decoder = LocalPatchDecoder(config)
        self.byte_output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(
            [nn.Linear(config.d_model, config.patch_size * config.vocab_size)
             for _ in range(2)]
        )
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, SparseMixtureOfExperts):
                module.reset_router()
        # Weight tying improves byte precision and removes an unnecessary matrix.
        self.byte_output.weight = self.byte_embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        byte_ids: Tensor,
        recurrent_steps: int = 1,
        top_k_experts: Optional[int] = None,
        return_mtp: bool = True,
    ) -> Dict[str, Any]:
        if byte_ids.ndim != 2 or byte_ids.shape[1] < 1:
            raise ValueError("byte_ids must have shape [batch, sequence>=1]")
        if byte_ids.dtype != torch.long:
            byte_ids = byte_ids.long()
        if torch.any((byte_ids < 0) | (byte_ids >= self.config.vocab_size)):
            raise ValueError("byte_ids contains values outside 0..255")

        byte_states = self.byte_embedding(byte_ids)
        encoded, padded, _ = self.encoder(byte_states)
        core = encoded
        balance_losses: List[Tensor] = []
        router_z_losses: List[Tensor] = []

        for block in self.core_blocks:
            core, balance, router_z = block(core, top_k_experts)
            balance_losses.append(balance)
            router_z_losses.append(router_z)

        repeats = max(0, min(int(recurrent_steps), self.config.max_recurrence))
        for repeat in range(repeats):
            core = core + self.recurrence_embedding[repeat][None, None, :]
            core, balance, router_z = self.recurrent_block(core, top_k_experts)
            balance_losses.append(balance)
            router_z_losses.append(router_z)

        core = self.core_norm(core)
        decoded = self.decoder(padded, encoded, core, byte_ids.shape[1])
        logits = self.byte_output(decoded)
        zero = logits.new_zeros((), dtype=torch.float32)
        balance_loss = torch.stack(balance_losses).mean() if balance_losses else zero
        router_z_loss = torch.stack(router_z_losses).mean() if router_z_losses else zero

        mtp_logits: Optional[List[Tensor]] = None
        if return_mtp:
            batch, patches, _ = core.shape
            mtp_logits = [
                head(core).reshape(
                    batch, patches, self.config.patch_size, self.config.vocab_size
                )
                for head in self.mtp_heads
            ]
        return {
            "logits": logits,
            "mtp_logits": mtp_logits,
            "balance_loss": balance_loss,
            "router_z_loss": router_z_loss,
        }


@dataclass
class LossBundle:
    total: Tensor
    byte_ce: Tensor
    mtp: Tensor
    balance_core: Tensor
    router_z_core: Tensor


def compute_joint_loss(
    model: HierarchicalByteModel,
    inputs: Tensor,
    targets: Tensor,
    recurrent_steps: int,
    top_k_experts: int,
) -> LossBundle:
    """Compute the exact requested scalar objective with each coefficient once."""
    output = model(
        inputs,
        recurrent_steps=recurrent_steps,
        top_k_experts=top_k_experts,
        return_mtp=True,
    )
    byte_ce = F.cross_entropy(
        output["logits"].reshape(-1, model.config.vocab_size), targets.reshape(-1)
    )

    patch_size = model.config.patch_size
    if inputs.shape[1] % patch_size:
        raise ValueError("training sequence length must be divisible by patch size K=4")
    patch_targets = inputs.reshape(inputs.shape[0], -1, patch_size)
    mtp_terms: List[Tensor] = []
    assert output["mtp_logits"] is not None
    for future_offset, prediction in enumerate(output["mtp_logits"], start=1):
        if patch_targets.shape[1] <= future_offset:
            continue
        valid_prediction = prediction[:, :-future_offset, :, :]
        valid_target = patch_targets[:, future_offset:, :]
        mtp_terms.append(
            F.cross_entropy(
                valid_prediction.reshape(-1, model.config.vocab_size),
                valid_target.reshape(-1),
            )
        )
    mtp = torch.stack(mtp_terms).mean() if mtp_terms else byte_ce.new_zeros(())

    balance_core = output["balance_loss"]
    router_z_core = output["router_z_loss"]
    # L_aux = 0.01 * E * sum_i(f_i P_i); L_z = 1e-4 * E[logsumexp(x)^2].
    # Writing the raw cores above and coefficients here prevents double-scaling.
    total = byte_ce + 0.5 * mtp + 0.01 * balance_core + 0.0001 * router_z_core
    return LossBundle(total, byte_ce, mtp, balance_core, router_z_core)


class WarmupCosineSchedule:
    """200-step linear warmup followed by cosine decay to 10% of peak LR."""

    def __init__(self, optimizer: torch.optim.Optimizer, peak_lr: float, total_steps: int):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.total_steps = max(201, int(total_steps))
        self.warmup_steps = 200
        self.last_step = -1

    def learning_rate(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps - 1
        )
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.peak_lr * (0.1 + 0.9 * cosine)

    def step(self, step: int) -> float:
        lr = self.learning_rate(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_step = step
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {"last_step": self.last_step, "total_steps": self.total_steps,
                "peak_lr": self.peak_lr}


def precision_context(precision: str):
    if precision == "bfloat16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sample_from_logits(logits: Tensor, temperature: float, top_p: float) -> int:
    """Numerically stable temperature and nucleus sampling for one byte."""
    temperature = max(0.05, float(temperature))
    top_p = min(1.0, max(0.01, float(top_p)))
    scaled = logits.detach().float() / temperature
    probabilities = F.softmax(scaled, dim=-1)
    sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative = torch.cumsum(sorted_probabilities, dim=-1)
    remove = cumulative - sorted_probabilities >= top_p
    sorted_probabilities = sorted_probabilities.masked_fill(remove, 0.0)
    total = sorted_probabilities.sum()
    if not torch.isfinite(total) or total <= 0:
        return int(torch.argmax(scaled).item())
    sorted_probabilities = sorted_probabilities / total
    selected = torch.multinomial(sorted_probabilities, num_samples=1)
    return int(sorted_indices[selected].item())


def speculative_bytes(
    model: HierarchicalByteModel,
    context_bytes: Sequence[int],
    recurrent_steps: int,
    top_k_experts: int,
    temperature: float,
    top_p: float,
    use_mtp: bool,
    maximum_context: int,
) -> Tuple[List[int], int]:
    """Return verified bytes and the number accepted from the two MTP draft heads."""
    patch_size = model.config.patch_size
    maximum_context = max(patch_size * 4, maximum_context)
    maximum_context -= maximum_context % patch_size
    cropped = list(context_bytes[-maximum_context:])
    if not cropped:
        cropped = [10]  # newline seed for an empty prompt
    tensor = torch.tensor(cropped, dtype=torch.long).unsqueeze(0)
    aligned = len(cropped) % patch_size == 0
    first = model(
        tensor,
        recurrent_steps=recurrent_steps,
        top_k_experts=top_k_experts,
        return_mtp=use_mtp and aligned,
    )
    main_byte = sample_from_logits(first["logits"][0, -1], temperature, top_p)
    if not use_mtp or not aligned or first["mtp_logits"] is None:
        return [main_byte], 0

    # Both heads are used: +1 patch then +2 patch. Greedy drafts maximize the
    # chance that the main autoregressive distribution verifies several bytes.
    drafts: List[int] = []
    for head_logits in first["mtp_logits"]:
        drafts.extend(
            int(value) for value in torch.argmax(head_logits[0, -1], dim=-1).tolist()
        )
    if not drafts or main_byte != drafts[0]:
        return [main_byte], 0

    trial = cropped + drafts
    trial_tensor = torch.tensor(trial, dtype=torch.long).unsqueeze(0)
    verifier = model(
        trial_tensor,
        recurrent_steps=recurrent_steps,
        top_k_experts=top_k_experts,
        return_mtp=False,
    )["logits"]
    verified = [main_byte]
    accepted = 1
    context_length = len(cropped)
    for draft_index in range(1, len(drafts)):
        logit_index = context_length + draft_index - 1
        sampled = sample_from_logits(verifier[0, logit_index], temperature, top_p)
        verified.append(sampled)
        if sampled != drafts[draft_index]:
            break
        accepted += 1
    return verified, accepted


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    sequence_length: int
    learning_rate: float
    recurrent_steps: int
    top_k_experts: int
    max_steps: int
    precision: str


@dataclass(frozen=True)
class GenerationSettings:
    temperature: float
    top_p: float
    recurrent_steps: int
    use_mtp: bool
    max_new_bytes: int
    top_k_experts: int
    maximum_context: int
    precision: str


def safe_torch_load(source: Any) -> Any:
    """Prefer PyTorch's restricted weights-only loader when the version supports it."""
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added
        return torch.load(source, map_location="cpu")


class LossPlotWidget(QtWidgets.QWidget):
    """Dependency-free loss chart painted directly with Qt."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.values: List[float] = []
        self.setMinimumHeight(250)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

    def add_value(self, value: float) -> None:
        if math.isfinite(value):
            self.values.append(float(value))
            if len(self.values) > 2_000:
                self.values = self.values[-2_000:]
            self.update()

    def clear_values(self) -> None:
        self.values.clear()
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt API
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        background = QtGui.QColor("#10151f")
        grid = QtGui.QColor("#293244")
        text = QtGui.QColor("#aebbd0")
        accent = QtGui.QColor("#55d6be")
        painter.fillRect(self.rect(), background)

        bounds = self.rect().adjusted(55, 18, -18, -35)
        painter.setPen(text)
        painter.drawText(12, 19, "Total loss")
        if len(self.values) < 2 or bounds.width() <= 0 or bounds.height() <= 0:
            painter.drawText(bounds, QtCore.Qt.AlignmentFlag.AlignCenter, "Waiting for training data")
            return

        visible = self.values[-600:]
        low = min(visible)
        high = max(visible)
        if math.isclose(low, high):
            low -= 0.5
            high += 0.5
        padding = (high - low) * 0.08
        low -= padding
        high += padding

        painter.setPen(QtGui.QPen(grid, 1))
        for line in range(5):
            y = bounds.top() + line * bounds.height() / 4.0
            painter.drawLine(
                QtCore.QPointF(bounds.left(), y), QtCore.QPointF(bounds.right(), y)
            )
            label_value = high - line * (high - low) / 4.0
            painter.setPen(text)
            painter.drawText(4, int(y + 4), f"{label_value:.3f}")
            painter.setPen(QtGui.QPen(grid, 1))

        path = QtGui.QPainterPath()
        for index, value in enumerate(visible):
            x = bounds.left() + index * bounds.width() / max(1, len(visible) - 1)
            y = bounds.bottom() - (value - low) * bounds.height() / (high - low)
            point = QtCore.QPointF(x, y)
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.setPen(QtGui.QPen(accent, 2.0))
        painter.drawPath(path)
        painter.setPen(text)
        painter.drawText(
            bounds.left(), self.height() - 10,
            f"last {len(visible)} steps  •  current {visible[-1]:.4f}",
        )


class ByteLabWindow(QtWidgets.QMainWindow):
    """PySide6 desktop shell; worker threads communicate only through ui_queue."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1120, 780)
        self.setMinimumSize(860, 620)

        requested_threads = os.environ.get("BYTE_LAB_CPU_THREADS")
        if requested_threads:
            try:
                cpu_threads = max(1, int(requested_threads))
            except ValueError:
                cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        else:
            cpu_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

        self.ui_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.model_lock = threading.RLock()
        self.model_config = ModelConfig()
        self.model = HierarchicalByteModel(self.model_config).cpu()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.pending_optimizer_state: Optional[Dict[str, Any]] = None
        self.global_step = 0

        self.training_thread: Optional[threading.Thread] = None
        self.generation_thread: Optional[threading.Thread] = None
        self.checkpoint_thread: Optional[threading.Thread] = None
        self.training_stop = threading.Event()
        self.training_run = threading.Event()
        self.generation_stop = threading.Event()
        self.dataset_ready = threading.Event()
        self.dataset_path: Optional[Path] = None
        self.dataset_error: Optional[BaseException] = None

        self._build_interface(cpu_threads)
        self._apply_style()
        self._set_training_buttons(False, False)
        self._set_generation_buttons(False)

        # The 50 ms timer drains the thread-safe queue. All widget mutations
        # happen here on the main GUI thread, never in a Python worker.
        self.queue_timer = QtCore.QTimer(self)
        self.queue_timer.setInterval(50)
        self.queue_timer.timeout.connect(self._poll_ui_queue)
        self.queue_timer.start()
        self.statusBar().showMessage("Preparing the byte dataset in the background…", 0)
        self.dataset_thread = threading.Thread(
            target=self._dataset_worker,
            name="dataset-download",
            daemon=True,
        )
        self.dataset_thread.start()

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #151b26; color: #e7edf7; }
            QTabWidget::pane { border: 1px solid #303b50; top: -1px; }
            QTabBar::tab { background: #1d2634; padding: 10px 20px; margin-right: 2px; }
            QTabBar::tab:selected { background: #2a3850; color: #6ee7d2; }
            QGroupBox { border: 1px solid #344157; border-radius: 7px; margin-top: 12px;
                        padding: 12px 8px 8px 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {
                background: #0f1520; border: 1px solid #3b4961; border-radius: 4px;
                padding: 5px; selection-background-color: #2e806f;
            }
            QPushButton { background: #2a3850; border: 1px solid #465976;
                          border-radius: 5px; padding: 7px 14px; }
            QPushButton:hover { background: #344967; }
            QPushButton:disabled { color: #69768a; background: #1b2230; }
            QPushButton#primary { background: #197965; border-color: #49cdb2; }
            QPushButton#danger { background: #71333d; border-color: #ae5665; }
            QLabel#metricValue { font-size: 17px; font-weight: 700; color: #6ee7d2; }
            QStatusBar { background: #0f1520; }
        """)

    @staticmethod
    def _metric_label(text: str = "—") -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(text)
        label.setObjectName("metricValue")
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        return label

    def _build_interface(self, cpu_threads: int) -> None:
        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        training_tab = QtWidgets.QWidget()
        inference_tab = QtWidgets.QWidget()
        tabs.addTab(training_tab, "Training")
        tabs.addTab(inference_tab, "Inference Playground")

        training_layout = QtWidgets.QVBoxLayout(training_tab)
        upper = QtWidgets.QHBoxLayout()
        training_layout.addLayout(upper)

        settings_box = QtWidgets.QGroupBox("Training controls")
        settings_form = QtWidgets.QFormLayout(settings_box)
        self.batch_input = QtWidgets.QSpinBox()
        self.batch_input.setRange(1, 64)
        self.batch_input.setValue(2)
        self.sequence_input = QtWidgets.QSpinBox()
        self.sequence_input.setRange(16, 2048)
        self.sequence_input.setSingleStep(4)
        self.sequence_input.setValue(128)
        self.learning_rate_input = QtWidgets.QDoubleSpinBox()
        self.learning_rate_input.setDecimals(7)
        self.learning_rate_input.setRange(0.0000001, 0.1)
        self.learning_rate_input.setSingleStep(0.00005)
        self.learning_rate_input.setValue(0.0003)
        self.training_recurrence_input = QtWidgets.QSpinBox()
        self.training_recurrence_input.setRange(0, self.model_config.max_recurrence)
        self.training_recurrence_input.setValue(1)
        self.top_k_input = QtWidgets.QSpinBox()
        self.top_k_input.setRange(1, self.model_config.n_experts)
        self.top_k_input.setValue(2)
        self.max_steps_input = QtWidgets.QSpinBox()
        self.max_steps_input.setRange(201, 10_000_000)
        self.max_steps_input.setValue(5_000)
        self.precision_input = QtWidgets.QComboBox()
        self.precision_input.addItems(["float32", "bfloat16"])

        settings_form.addRow("Batch size", self.batch_input)
        settings_form.addRow("Sequence length", self.sequence_input)
        settings_form.addRow("Peak learning rate", self.learning_rate_input)
        settings_form.addRow("Latent recurrence steps", self.training_recurrence_input)
        settings_form.addRow("Top-k routed experts", self.top_k_input)
        settings_form.addRow("Steps this run", self.max_steps_input)
        settings_form.addRow("CPU precision", self.precision_input)

        control_row = QtWidgets.QHBoxLayout()
        self.start_training_button = QtWidgets.QPushButton("Start")
        self.start_training_button.setObjectName("primary")
        self.pause_training_button = QtWidgets.QPushButton("Pause")
        self.resume_training_button = QtWidgets.QPushButton("Resume")
        self.stop_training_button = QtWidgets.QPushButton("Stop")
        self.stop_training_button.setObjectName("danger")
        for button in (
            self.start_training_button,
            self.pause_training_button,
            self.resume_training_button,
            self.stop_training_button,
        ):
            control_row.addWidget(button)
        settings_form.addRow(control_row)
        upper.addWidget(settings_box, 0)

        metrics_box = QtWidgets.QGroupBox("Live metrics")
        metrics_grid = QtWidgets.QGridLayout(metrics_box)
        self.step_metric = self._metric_label("0")
        self.loss_metric = self._metric_label("—")
        self.byte_loss_metric = self._metric_label("—")
        self.mtp_loss_metric = self._metric_label("—")
        self.speed_metric = self._metric_label("—")
        self.lr_metric = self._metric_label("—")
        self.grad_metric = self._metric_label("—")
        self.route_metric = self._metric_label("—")
        metric_items = [
            ("Step", self.step_metric),
            ("Total loss", self.loss_metric),
            ("Byte CE", self.byte_loss_metric),
            ("MTP loss", self.mtp_loss_metric),
            ("Tokens/s (bytes/s)", self.speed_metric),
            ("Learning rate", self.lr_metric),
            ("Gradient norm", self.grad_metric),
            ("Balance / Router-Z", self.route_metric),
        ]
        for index, (name, widget) in enumerate(metric_items):
            row, column = divmod(index, 4)
            card = QtWidgets.QVBoxLayout()
            caption = QtWidgets.QLabel(name)
            caption.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            card.addWidget(caption)
            card.addWidget(widget)
            metrics_grid.addLayout(card, row, column)
        upper.addWidget(metrics_box, 1)

        model_row = QtWidgets.QHBoxLayout()
        self.model_summary_label = QtWidgets.QLabel()
        self._refresh_model_summary(cpu_threads)
        model_row.addWidget(self.model_summary_label, 1)
        self.save_button = QtWidgets.QPushButton("Save checkpoint…")
        self.load_button = QtWidgets.QPushButton("Load checkpoint…")
        model_row.addWidget(self.save_button)
        model_row.addWidget(self.load_button)
        training_layout.addLayout(model_row)

        self.loss_plot = LossPlotWidget()
        training_layout.addWidget(self.loss_plot, 1)

        self.training_log = QtWidgets.QPlainTextEdit()
        self.training_log.setReadOnly(True)
        self.training_log.setMaximumBlockCount(250)
        self.training_log.setFixedHeight(100)
        training_layout.addWidget(self.training_log)

        inference_layout = QtWidgets.QVBoxLayout(inference_tab)
        inference_layout.addWidget(QtWidgets.QLabel("Prompt (UTF-8 text)"))
        self.prompt_input = QtWidgets.QPlainTextEdit()
        self.prompt_input.setPlaceholderText("Enter a prompt, then generate…")
        self.prompt_input.setPlainText("ROMEO:\n")
        self.prompt_input.setMaximumHeight(150)
        inference_layout.addWidget(self.prompt_input)

        generation_box = QtWidgets.QGroupBox("Generation controls")
        generation_layout = QtWidgets.QGridLayout(generation_box)
        self.temperature_input = QtWidgets.QDoubleSpinBox()
        self.temperature_input.setRange(0.05, 3.0)
        self.temperature_input.setSingleStep(0.05)
        self.temperature_input.setValue(0.8)
        self.top_p_input = QtWidgets.QDoubleSpinBox()
        self.top_p_input.setRange(0.01, 1.0)
        self.top_p_input.setSingleStep(0.01)
        self.top_p_input.setValue(0.95)
        self.thought_steps_input = QtWidgets.QSpinBox()
        self.thought_steps_input.setRange(0, 5)
        self.thought_steps_input.setValue(1)
        self.mtp_toggle = QtWidgets.QCheckBox("Use both MTP draft heads")
        self.mtp_toggle.setChecked(True)
        self.max_bytes_input = QtWidgets.QSpinBox()
        self.max_bytes_input.setRange(1, 4096)
        self.max_bytes_input.setValue(256)
        generation_layout.addWidget(QtWidgets.QLabel("Temperature"), 0, 0)
        generation_layout.addWidget(self.temperature_input, 0, 1)
        generation_layout.addWidget(QtWidgets.QLabel("Top-p"), 0, 2)
        generation_layout.addWidget(self.top_p_input, 0, 3)
        generation_layout.addWidget(QtWidgets.QLabel("Recurrent thought steps (0–5)"), 1, 0)
        generation_layout.addWidget(self.thought_steps_input, 1, 1)
        generation_layout.addWidget(self.mtp_toggle, 1, 2)
        generation_layout.addWidget(QtWidgets.QLabel("Maximum new bytes"), 2, 0)
        generation_layout.addWidget(self.max_bytes_input, 2, 1)
        self.generate_button = QtWidgets.QPushButton("Generate")
        self.generate_button.setObjectName("primary")
        self.stop_generation_button = QtWidgets.QPushButton("Stop generation")
        self.stop_generation_button.setObjectName("danger")
        self.clear_output_button = QtWidgets.QPushButton("Clear output")
        generation_layout.addWidget(self.generate_button, 2, 2)
        generation_layout.addWidget(self.stop_generation_button, 2, 3)
        generation_layout.addWidget(self.clear_output_button, 2, 4)
        inference_layout.addWidget(generation_box)

        output_header = QtWidgets.QHBoxLayout()
        output_header.addWidget(QtWidgets.QLabel("Streaming generated bytes → UTF-8"))
        self.generation_metric = QtWidgets.QLabel("Idle")
        output_header.addStretch(1)
        output_header.addWidget(self.generation_metric)
        inference_layout.addLayout(output_header)
        self.generated_output = QtWidgets.QPlainTextEdit()
        self.generated_output.setReadOnly(True)
        fixed_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.generated_output.setFont(fixed_font)
        inference_layout.addWidget(self.generated_output, 1)

        self.start_training_button.clicked.connect(self.start_training)
        self.pause_training_button.clicked.connect(self.pause_training)
        self.resume_training_button.clicked.connect(self.resume_training)
        self.stop_training_button.clicked.connect(self.stop_training)
        self.save_button.clicked.connect(self.save_checkpoint)
        self.load_button.clicked.connect(self.load_checkpoint)
        self.generate_button.clicked.connect(self.start_generation)
        self.stop_generation_button.clicked.connect(self.stop_generation)
        self.clear_output_button.clicked.connect(self.generated_output.clear)

        file_menu = self.menuBar().addMenu("Model")
        save_action = file_menu.addAction("Save checkpoint…")
        load_action = file_menu.addAction("Load checkpoint…")
        save_action.triggered.connect(self.save_checkpoint)
        load_action.triggered.connect(self.load_checkpoint)

    def _refresh_model_summary(self, cpu_threads: Optional[int] = None) -> None:
        if cpu_threads is None:
            cpu_threads = torch.get_num_threads()
        count = self.model.parameter_count()
        self.model_summary_label.setText(
            f"{count / 1_000_000:.2f}M parameters  •  K={self.model_config.patch_size}  •  "
            f"{self.model_config.n_experts} routed + 1 shared expert  •  CPU threads={cpu_threads}"
        )

    def _post(self, event: str, payload: Any = None) -> None:
        self.ui_queue.put((event, payload))

    def _append_log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.training_log.appendPlainText(f"[{stamp}] {text}")

    def _set_training_buttons(self, active: bool, paused: bool) -> None:
        self.start_training_button.setEnabled(not active)
        self.pause_training_button.setEnabled(active and not paused)
        self.resume_training_button.setEnabled(active and paused)
        self.stop_training_button.setEnabled(active)
        self.load_button.setEnabled(not active and not self._generation_active())

    def _set_generation_buttons(self, active: bool) -> None:
        self.generate_button.setEnabled(not active)
        self.stop_generation_button.setEnabled(active)
        self.load_button.setEnabled(not active and not self._training_active())

    def _training_active(self) -> bool:
        return self.training_thread is not None and self.training_thread.is_alive()

    def _generation_active(self) -> bool:
        return self.generation_thread is not None and self.generation_thread.is_alive()

    def _training_settings(self) -> TrainingSettings:
        sequence = int(self.sequence_input.value())
        if sequence % self.model_config.patch_size:
            raise ValueError("Sequence length must be divisible by patch size K=4.")
        return TrainingSettings(
            batch_size=int(self.batch_input.value()),
            sequence_length=sequence,
            learning_rate=float(self.learning_rate_input.value()),
            recurrent_steps=int(self.training_recurrence_input.value()),
            top_k_experts=int(self.top_k_input.value()),
            max_steps=int(self.max_steps_input.value()),
            precision=self.precision_input.currentText(),
        )

    def start_training(self) -> None:
        if self._training_active():
            return
        try:
            settings = self._training_settings()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid training settings", str(exc))
            return
        self.training_stop.clear()
        self.training_run.set()
        self.loss_plot.clear_values()
        self._set_training_buttons(True, False)
        self._append_log(
            f"Training started: batch={settings.batch_size}, seq={settings.sequence_length}, "
            f"recurrence={settings.recurrent_steps}, Top-{settings.top_k_experts}."
        )
        self.training_thread = threading.Thread(
            target=self._training_worker,
            args=(settings,),
            name="byte-model-training",
            daemon=True,
        )
        self.training_thread.start()

    def pause_training(self) -> None:
        if self._training_active():
            self.training_run.clear()
            self._set_training_buttons(True, True)
            self.statusBar().showMessage("Training paused", 0)
            self._append_log("Pause requested; it takes effect between optimizer steps.")

    def resume_training(self) -> None:
        if self._training_active():
            self.training_run.set()
            self._set_training_buttons(True, False)
            self.statusBar().showMessage("Training resumed", 0)
            self._append_log("Training resumed.")

    def stop_training(self) -> None:
        if self._training_active():
            self.training_stop.set()
            self.training_run.set()  # release a paused worker so it can exit
            self.statusBar().showMessage("Stopping after the current optimizer step…", 0)
            self._append_log("Stop requested.")

    def _training_worker(self, settings: TrainingSettings) -> None:
        loader: Optional[StreamingByteLoader] = None
        try:
            while not self.dataset_ready.wait(timeout=0.1):
                if self.training_stop.is_set():
                    self._post("training_finished", "stopped")
                    return
            if self.dataset_error is not None:
                raise RuntimeError("dataset preparation failed") from self.dataset_error
            if self.dataset_path is None:
                raise RuntimeError("dataset preparation completed without a file")
            dataset_path = self.dataset_path
            loader = StreamingByteLoader(
                dataset_path, settings.batch_size, settings.sequence_length,
                seed=1337 + self.global_step,
            )
            with self.model_lock:
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=settings.learning_rate,
                    betas=(0.9, 0.98),
                    weight_decay=0.01,
                )
                if self.pending_optimizer_state is not None:
                    try:
                        optimizer.load_state_dict(self.pending_optimizer_state)
                        self._post("log", "Restored optimizer state from checkpoint.")
                    except (ValueError, RuntimeError):
                        self._post("log", "Checkpoint optimizer state was incompatible; reset it.")
                    self.pending_optimizer_state = None
                self.optimizer = optimizer
            schedule = WarmupCosineSchedule(
                optimizer, settings.learning_rate, settings.max_steps
            )
            speed_ema: Optional[float] = None

            for run_step in range(settings.max_steps):
                if self.training_stop.is_set():
                    break
                while not self.training_run.wait(timeout=0.1):
                    if self.training_stop.is_set():
                        break
                if self.training_stop.is_set():
                    break

                inputs, targets = loader.next_batch()
                started = time.perf_counter()
                with self.model_lock:
                    self.model.train(True)
                    lr = schedule.step(run_step)
                    optimizer.zero_grad(set_to_none=True)
                    with precision_context(settings.precision):
                        losses = compute_joint_loss(
                            self.model,
                            inputs,
                            targets,
                            settings.recurrent_steps,
                            settings.top_k_experts,
                        )
                    if not bool(torch.isfinite(losses.total).item()):
                        raise FloatingPointError("non-finite total loss encountered")
                    losses.total.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    self.global_step += 1
                    completed_step = self.global_step

                elapsed = max(1e-9, time.perf_counter() - started)
                current_speed = inputs.numel() / elapsed
                speed_ema = (
                    current_speed if speed_ema is None
                    else 0.9 * speed_ema + 0.1 * current_speed
                )
                metrics = {
                    "step": completed_step,
                    "loss": float(losses.total.detach().float().item()),
                    "byte_ce": float(losses.byte_ce.detach().float().item()),
                    "mtp": float(losses.mtp.detach().float().item()),
                    "balance": float(losses.balance_core.detach().float().item()),
                    "router_z": float(losses.router_z_core.detach().float().item()),
                    "speed": float(speed_ema),
                    "lr": float(lr),
                    "grad_norm": float(torch.as_tensor(gradient_norm).float().item()),
                }
                self._post("training_metrics", metrics)

            reason = "stopped" if self.training_stop.is_set() else "completed"
            self._post("training_finished", reason)
        except Exception as exc:
            self._post(
                "error",
                ("Training failed", str(exc), traceback.format_exc()),
            )
            self._post("training_finished", "failed")
        finally:
            if loader is not None:
                loader.close()

    def _dataset_worker(self) -> None:
        try:
            self.dataset_path = ensure_dataset(
                lambda message: self._post("status", message)
            )
            self._post("status", f"Byte dataset ready: {self.dataset_path}")
        except BaseException as exc:  # preserve the original cause for training
            self.dataset_error = exc
            self._post(
                "error",
                ("Dataset preparation failed", str(exc), traceback.format_exc()),
            )
        finally:
            self.dataset_ready.set()

    def start_generation(self) -> None:
        if self._generation_active():
            return
        prompt = self.prompt_input.toPlainText()
        try:
            training_settings = self._training_settings()
            settings = GenerationSettings(
                temperature=float(self.temperature_input.value()),
                top_p=float(self.top_p_input.value()),
                recurrent_steps=int(self.thought_steps_input.value()),
                use_mtp=bool(self.mtp_toggle.isChecked()),
                max_new_bytes=int(self.max_bytes_input.value()),
                top_k_experts=training_settings.top_k_experts,
                maximum_context=max(64, min(training_settings.sequence_length, 2048)),
                precision=training_settings.precision,
            )
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid generation settings", str(exc))
            return
        self.generated_output.clear()
        self.generation_stop.clear()
        self._set_generation_buttons(True)
        self.generation_metric.setText("Starting…")
        self.generation_thread = threading.Thread(
            target=self._generation_worker,
            args=(prompt, settings),
            name="byte-model-generation",
            daemon=True,
        )
        self.generation_thread.start()

    def stop_generation(self) -> None:
        if self._generation_active():
            self.generation_stop.set()
            self.generation_metric.setText("Stopping…")

    def _generation_worker(self, prompt: str, settings: GenerationSettings) -> None:
        try:
            context_bytes = list(prompt.encode("utf-8"))
            if not context_bytes:
                context_bytes = [10]
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            produced_count = 0
            mtp_accepted = 0
            started = time.perf_counter()
            while produced_count < settings.max_new_bytes and not self.generation_stop.is_set():
                with self.model_lock:
                    prior_mode = self.model.training
                    self.model.eval()
                    try:
                        with torch.inference_mode(), precision_context(settings.precision):
                            new_bytes, accepted = speculative_bytes(
                                self.model,
                                context_bytes,
                                settings.recurrent_steps,
                                settings.top_k_experts,
                                settings.temperature,
                                settings.top_p,
                                settings.use_mtp,
                                settings.maximum_context,
                            )
                    finally:
                        self.model.train(prior_mode)

                remaining = settings.max_new_bytes - produced_count
                new_bytes = new_bytes[:remaining]
                accepted = min(accepted, len(new_bytes))
                for byte_value in new_bytes:
                    if self.generation_stop.is_set():
                        break
                    context_bytes.append(byte_value)
                    produced_count += 1
                    decoded = decoder.decode(bytes((byte_value,)), final=False)
                    if decoded:
                        self._post("generation_chunk", decoded)
                mtp_accepted += accepted
                elapsed = max(1e-9, time.perf_counter() - started)
                self._post(
                    "generation_metrics",
                    {
                        "count": produced_count,
                        "speed": produced_count / elapsed,
                        "accepted": mtp_accepted,
                    },
                )

            tail = decoder.decode(b"", final=True)
            if tail:
                self._post("generation_chunk", tail)
            self._post(
                "generation_finished",
                "stopped" if self.generation_stop.is_set() else "completed",
            )
        except Exception as exc:
            self._post(
                "error",
                ("Generation failed", str(exc), traceback.format_exc()),
            )
            self._post("generation_finished", "failed")

    def save_checkpoint(self) -> None:
        if self.checkpoint_thread is not None and self.checkpoint_thread.is_alive():
            return
        default = str(application_data_dir() / "stratabyte_7m.pt")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save model checkpoint", default, "PyTorch checkpoint (*.pt)"
        )
        if not path:
            return
        if not path.lower().endswith(".pt"):
            path += ".pt"
        self.checkpoint_thread = threading.Thread(
            target=self._save_checkpoint_worker,
            args=(Path(path),),
            name="checkpoint-save",
            daemon=True,
        )
        self.checkpoint_thread.start()

    def _save_checkpoint_worker(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self.model_lock:
                checkpoint = {
                    "format_version": FORMAT_VERSION,
                    "config": asdict(self.model_config),
                    "model_state": copy.deepcopy(self.model.state_dict()),
                    "optimizer_state": (
                        copy.deepcopy(self.optimizer.state_dict())
                        if self.optimizer is not None else None
                    ),
                    "global_step": self.global_step,
                }
            temporary = path.with_suffix(path.suffix + ".tmp")
            torch.save(checkpoint, temporary)
            os.replace(temporary, path)
            self._post("checkpoint_saved", str(path))
        except Exception as exc:
            self._post("error", ("Checkpoint save failed", str(exc), traceback.format_exc()))

    def load_checkpoint(self) -> None:
        if self._training_active() or self._generation_active():
            QtWidgets.QMessageBox.information(
                self, "Model busy", "Stop training and generation before loading a checkpoint."
            )
            return
        if self.checkpoint_thread is not None and self.checkpoint_thread.is_alive():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load model checkpoint",
            str(application_data_dir()),
            "PyTorch checkpoint (*.pt);;All files (*)",
        )
        if not path:
            return
        self.checkpoint_thread = threading.Thread(
            target=self._load_checkpoint_worker,
            args=(Path(path),),
            name="checkpoint-load",
            daemon=True,
        )
        self.checkpoint_thread.start()

    def _load_checkpoint_worker(self, path: Path) -> None:
        try:
            checkpoint = safe_torch_load(path)
            if not isinstance(checkpoint, dict):
                raise ValueError("checkpoint root must be a dictionary")
            if "model_state" in checkpoint:
                state = checkpoint["model_state"]
                supplied = checkpoint.get("config", {})
                allowed = {field.name for field in fields(ModelConfig)}
                config_values = {key: value for key, value in supplied.items() if key in allowed}
                config = ModelConfig(**config_values)
                step = int(checkpoint.get("global_step", 0))
                optimizer_state = checkpoint.get("optimizer_state")
            else:
                state = checkpoint
                config = self.model_config
                step = 0
                optimizer_state = None
            config.validate()
            replacement = HierarchicalByteModel(config).cpu()
            replacement.load_state_dict(state, strict=True)
            replacement.eval()
            with self.model_lock:
                self.model = replacement
                self.model_config = config
                self.global_step = step
                self.optimizer = None
                self.pending_optimizer_state = optimizer_state
            self._post(
                "checkpoint_loaded",
                {"path": str(path), "step": step, "parameters": replacement.parameter_count()},
            )
        except Exception as exc:
            self._post("error", ("Checkpoint load failed", str(exc), traceback.format_exc()))

    def _poll_ui_queue(self) -> None:
        # Limit each pass so a very chatty generator cannot starve Qt events.
        for _ in range(300):
            try:
                event, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if event == "status":
                self.statusBar().showMessage(str(payload), 0)
                self._append_log(str(payload))
            elif event == "log":
                self._append_log(str(payload))
            elif event == "training_metrics":
                data = payload
                self.step_metric.setText(f"{data['step']:,}")
                self.loss_metric.setText(f"{data['loss']:.4f}")
                self.byte_loss_metric.setText(f"{data['byte_ce']:.4f}")
                self.mtp_loss_metric.setText(f"{data['mtp']:.4f}")
                self.speed_metric.setText(f"{data['speed']:,.1f}")
                self.lr_metric.setText(f"{data['lr']:.3e}")
                self.grad_metric.setText(f"{data['grad_norm']:.3f}")
                self.route_metric.setText(
                    f"{data['balance']:.3f} / {data['router_z']:.3f}"
                )
                self.loss_plot.add_value(data["loss"])
                self.statusBar().showMessage(
                    f"Training step {data['step']:,} — {data['speed']:,.1f} byte tokens/s", 0
                )
            elif event == "training_finished":
                self._set_training_buttons(False, False)
                self.statusBar().showMessage(f"Training {payload}", 0)
                self._append_log(f"Training {payload} at global step {self.global_step:,}.")
            elif event == "generation_chunk":
                cursor = self.generated_output.textCursor()
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                cursor.insertText(str(payload))
                self.generated_output.setTextCursor(cursor)
                self.generated_output.ensureCursorVisible()
            elif event == "generation_metrics":
                data = payload
                self.generation_metric.setText(
                    f"{data['count']} bytes  •  {data['speed']:.1f}/s  •  "
                    f"MTP accepted {data['accepted']}"
                )
            elif event == "generation_finished":
                self._set_generation_buttons(False)
                self.generation_metric.setText(
                    self.generation_metric.text() + f"  •  {payload}"
                )
            elif event == "checkpoint_saved":
                self.statusBar().showMessage(f"Checkpoint saved: {payload}", 8_000)
                self._append_log(f"Checkpoint saved: {payload}")
            elif event == "checkpoint_loaded":
                self._refresh_model_summary()
                self.training_recurrence_input.setMaximum(self.model_config.max_recurrence)
                self.top_k_input.setMaximum(self.model_config.n_experts)
                self.thought_steps_input.setMaximum(min(5, self.model_config.max_recurrence))
                self.step_metric.setText(f"{payload['step']:,}")
                self.statusBar().showMessage(f"Loaded checkpoint: {payload['path']}", 8_000)
                self._append_log(
                    f"Loaded {payload['parameters'] / 1_000_000:.2f}M-parameter checkpoint "
                    f"at step {payload['step']:,}."
                )
            elif event == "error":
                title, message, details = payload
                dialog = QtWidgets.QMessageBox(self)
                dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                dialog.setWindowTitle(title)
                dialog.setText(message)
                dialog.setDetailedText(details)
                dialog.exec()
                self._append_log(f"{title}: {message}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 - Qt API
        self.training_stop.set()
        self.training_run.set()
        self.generation_stop.set()
        event.accept()


def run_self_test() -> None:
    """Fast CPU test of shapes, losses, causality, recurrence, save/load, and MTP."""
    print("Running StrataByte-7M self-test…")
    torch.manual_seed(7)
    config = ModelConfig(
        d_model=32,
        n_heads=8,
        n_layers=1,
        q_rank=16,
        kv_rank=16,
        expert_hidden=48,
        n_experts=8,
        default_top_k=2,
        dropout=0.0,
        max_recurrence=2,
    )
    model = HierarchicalByteModel(config).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.98), weight_decay=0.01
    )
    schedule = WarmupCosineSchedule(optimizer, 3e-4, 400)
    assert math.isclose(schedule.step(0), 3e-4 / 200.0, rel_tol=1e-7)
    assert math.isclose(schedule.learning_rate(199), 3e-4, rel_tol=1e-7)

    sequence = torch.randint(0, 256, (2, 16), dtype=torch.long)
    targets = torch.roll(sequence, shifts=-1, dims=1)
    model.train()
    losses = compute_joint_loss(model, sequence, targets, 1, 2)
    assert losses.total.ndim == 0 and bool(torch.isfinite(losses.total).item())
    losses.total.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert bool(torch.isfinite(torch.as_tensor(norm)).item())
    optimizer.step()

    model.eval()
    causal_a = torch.randint(0, 256, (1, 16), dtype=torch.long)
    causal_b = causal_a.clone()
    pivot = 6
    causal_b[:, pivot + 1:] = torch.randint(
        0, 256, causal_b[:, pivot + 1:].shape, dtype=torch.long
    )
    with torch.inference_mode():
        output_a = model(causal_a, recurrent_steps=1, top_k_experts=2)["logits"]
        output_b = model(causal_b, recurrent_steps=1, top_k_experts=2)["logits"]
    causal_error = float((output_a[:, :pivot + 1] - output_b[:, :pivot + 1]).abs().max())
    assert causal_error < 1e-5, f"future-byte leakage detected: {causal_error}"

    strict_attention = StrictLocalCausalAttention(32, 8, 0.0).eval()
    with torch.inference_mode():
        first_row = strict_attention(torch.randn(2, 4, 32))[:, 0]
    assert float(first_row.abs().max()) < 1e-7, "strict local mask includes its diagonal"

    buffer = io.BytesIO()
    torch.save(
        {"format_version": FORMAT_VERSION, "config": asdict(config),
         "model_state": model.state_dict(), "global_step": 1},
        buffer,
    )
    buffer.seek(0)
    restored_data = safe_torch_load(buffer)
    restored = HierarchicalByteModel(ModelConfig(**restored_data["config"]))
    restored.load_state_dict(restored_data["model_state"], strict=True)
    restored.eval()
    with torch.inference_mode():
        generated, accepted = speculative_bytes(
            restored, list(b"Test prompt here"), 1, 2, 0.8, 0.95, True, 64
        )
    assert generated and all(0 <= value <= 255 for value in generated)
    assert 0 <= accepted <= 8
    print(
        f"PASS — {model.parameter_count():,} test parameters; finite forward/backward; "
        f"causal error={causal_error:.2e}; checkpoint and generation verified."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument(
        "--self-test", action="store_true",
        help="run a small forward/backward/causality/checkpoint test and exit",
    )
    parser.add_argument(
        "--download-data", action="store_true",
        help="download/prepare the dataset and exit",
    )
    arguments = parser.parse_args()
    if arguments.self_test:
        run_self_test()
        return 0
    if arguments.download_data:
        path = ensure_dataset(print)
        print(path)
        return 0

    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName(APP_NAME)
    application.setOrganizationName("StrataByte Research")
    window = ByteLabWindow()
    window.show()
    return int(application.exec())


if __name__ == "__main__":
    raise SystemExit(main())
