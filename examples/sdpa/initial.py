# EVOLVE-BLOCK-START
"""Scaled Dot-Product Attention (SDPA) using Triton (with safe fallback).

This block contains the core implementation that Shinka will evolve. It
implements a Triton kernel for forward SDPA and exposes a PyTorch-facing
function `pure_scaled_dot_product_attention` that returns the attention output.

If Triton or a compatible GPU is unavailable, a numerically stable PyTorch
fallback is used to ensure evaluation can still run.
"""

import math
import time
from typing import Optional

import torch

try:  # Optional Triton dependency for GPU kernels
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    TRITON_AVAILABLE = False
    triton = None  # type: ignore
    tl = None  # type: ignore


if TRITON_AVAILABLE:
    @triton.jit
    def _attn_fwd(
        Q,
        K,
        V,
        Out,
        stride_q_bh,
        stride_q_t,
        stride_q_d,
        stride_k_bh,
        stride_k_t,
        stride_k_d,
        stride_v_bh,
        stride_v_t,
        stride_v_d,
        stride_o_bh,
        stride_o_t,
        stride_o_d,
        B,
        H,
        TQ,
        TK,
        D,
        CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        SCALE: tl.constexpr,
    ):
        # Each program handles one block of queries [BLOCK_M, D] for one (b,h)
        bh = tl.program_id(0)
        row_block = tl.program_id(1)

        b = bh // H
        h = bh % H

        q_offs_m = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        q_offs_d = tl.arange(0, BLOCK_D)

        # Pointers (assumes input flattened to BH x T x D stride space)
        Q_ptrs = (
            Q
            + b * stride_q_bh
            + h * stride_q_bh * 0
            + q_offs_m[:, None] * stride_q_t
            + q_offs_d[None, :] * stride_q_d
        )
        O_ptrs = (
            Out
            + b * stride_o_bh
            + h * stride_o_bh * 0
            + q_offs_m[:, None] * stride_o_t
            + q_offs_d[None, :] * stride_o_d
        )

        # Init accumulators
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        # Load Q
        q_mask = q_offs_m < TQ
        q = tl.load(Q_ptrs, mask=q_mask[:, None] & (q_offs_d[None, :] < D), other=0.0)
        q = q.to(tl.float32)

        # Iterate over K/V blocks
        for start_n in range(0, TK, BLOCK_N):
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
            k_offs_d = tl.arange(0, BLOCK_D)

            K_ptrs = (
                K
                + b * stride_k_bh
                + h * stride_k_bh * 0
                + k_offs_n[:, None] * stride_k_t
                + k_offs_d[None, :] * stride_k_d
            )
            V_ptrs = (
                V
                + b * stride_v_bh
                + h * stride_v_bh * 0
                + k_offs_n[:, None] * stride_v_t
                + k_offs_d[None, :] * stride_v_d
            )

            k = tl.load(
                K_ptrs,
                mask=(k_offs_n[:, None] < TK) & (k_offs_d[None, :] < D),
                other=0.0,
            )
            v = tl.load(
                V_ptrs,
                mask=(k_offs_n[:, None] < TK) & (k_offs_d[None, :] < D),
                other=0.0,
            )
            k = k.to(tl.float32)
            v = v.to(tl.float32)

            # Attention scores: [M, N]
            qk = tl.dot(q, tl.trans(k)) * SCALE

            # Causal mask
            if CAUSAL:
                q_idx = q_offs_m[:, None]
                k_idx = k_offs_n[None, :]
                causal_mask = k_idx > q_idx
                qk = tl.where(causal_mask, -float("inf"), qk)

            # Numerically stable softmax
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)

            # Update accumulators
            l_i_new = l_i * tl.exp(m_i - m_ij) + l_ij
            acc = acc * ((l_i * tl.exp(m_i - m_ij)) / l_i_new)[:, None] + tl.dot(
                p.to(v.dtype), v
            ) / l_i_new[:, None]

            l_i = l_i_new
            m_i = m_ij

        # Write output
        o = acc.to(tl.float32)
        tl.store(O_ptrs, o, mask=q_mask[:, None] & (q_offs_d[None, :] < D))


def _torch_sdpa_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Numerically stable PyTorch SDPA (CPU/GPU), used as a safe fallback."""
    B, H, TQ, D = q.shape
    _, _, TK, _ = k.shape
    scale_val = (1.0 / math.sqrt(D)) if scale is None else float(scale)

    logits = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))
    logits = logits * scale_val

    if is_causal:
        q_idx = torch.arange(TQ, device=logits.device).view(TQ, 1)
        k_idx = torch.arange(TK, device=logits.device).view(1, TK)
        causal_mask = (k_idx > q_idx).to(torch.bool)
        logits = logits.masked_fill(causal_mask.view(1, 1, TQ, TK), float("-inf"))

    logits = logits - logits.amax(dim=-1, keepdim=True)
    attn = torch.exp(logits)
    denom = attn.sum(dim=-1, keepdim=True)
    attn = attn / denom.clamp_min(1e-8)
    out = torch.matmul(attn.to(v.dtype), v)
    return out.to(torch.float32)


def pure_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute SDPA via Triton kernel if available, otherwise fallback to PyTorch.

    Args:
        q, k, v: Tensors of shape [B, H, T, D]
        is_causal: Whether to apply a causal mask
        scale: Optional custom scaling factor (default 1/sqrt(D))
    Returns:
        out: Tensor of shape [B, H, T, D]
    """
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4, "Expected 4D Q,K,V"
    B, H, TQ, D = q.shape
    _, _, TK, _ = k.shape

    # Heuristic: Use fallback unless Triton and CUDA are available
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
        return _torch_sdpa_fallback(q, k, v, is_causal=is_causal, scale=scale)

    # Ensure contiguous and on CUDA
    device = torch.device("cuda")
    q = q.contiguous().to(device)
    k = k.contiguous().to(device)
    v = v.contiguous().to(device)
    o = torch.empty_like(q, dtype=torch.float32, device=device)

    # Compute logical strides for the 3D view [BH, T, D]
    q_bh = q.view(B * H, TQ, D)
    k_bh = k.view(B * H, TK, D)
    v_bh = v.view(B * H, TK, D)
    o_bh = o.view(B * H, TQ, D)

    # Strides in elements
    stride_q_bh, stride_q_t, stride_q_d = q_bh.stride(0), q_bh.stride(1), q_bh.stride(2)
    stride_k_bh, stride_k_t, stride_k_d = k_bh.stride(0), k_bh.stride(1), k_bh.stride(2)
    stride_v_bh, stride_v_t, stride_v_d = v_bh.stride(0), v_bh.stride(1), v_bh.stride(2)
    stride_o_bh, stride_o_t, stride_o_d = o_bh.stride(0), o_bh.stride(1), o_bh.stride(2)

    # Tiling parameters
    BLOCK_M, BLOCK_N = 64, 64
    BLOCK_D = min(128, triton.next_power_of_2(D)) if TRITON_AVAILABLE else 128
    SCALE = (1.0 / math.sqrt(D)) if scale is None else float(scale)

    grid = (B * H, triton.cdiv(TQ, BLOCK_M)) if TRITON_AVAILABLE else (1, 1)

    # Launch Triton kernel
    if TRITON_AVAILABLE:
        _attn_fwd[grid](
            q_bh,
            k_bh,
            v_bh,
            o_bh,
            stride_q_bh,
            stride_q_t,
            stride_q_d,
            stride_k_bh,
            stride_k_t,
            stride_k_d,
            stride_v_bh,
            stride_v_t,
            stride_v_d,
            stride_o_bh,
            stride_o_t,
            stride_o_d,
            B,
            H,
            TQ,
            TK,
            D,
            is_causal,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            SCALE=SCALE,
            num_warps=4,
            num_stages=2,
        )
        return o

    # Safety fallback (shouldn't reach here)
    return _torch_sdpa_fallback(q, k, v, is_causal=is_causal, scale=scale)


# EVOLVE-BLOCK-END


# Immutable runner wrapper (kept fixed for evaluation)
def run_sdpa(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    """Entry point invoked by the evaluator.

    Returns a tuple of (outputs, reported_score). Shinka will evolve only the
    implementation inside the EVOLVE-BLOCK above.
    """
    # Measure latency and memory; prefer accurate CUDA stats when available.
    B, H, TQ, D = q.shape
    TK = k.shape[2]

    used_triton = bool(TRITON_AVAILABLE and torch.cuda.is_available())
    latency_ms: float
    peak_mem_bytes: int

    if used_triton:
        device = torch.device("cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        out = pure_scaled_dot_product_attention(
            q=q, k=k, v=v, is_causal=is_causal, scale=scale
        )
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        peak_mem_bytes = int(torch.cuda.max_memory_allocated(device))
    else:
        # CPU fallback timing and a conservative memory estimate
        t0 = time.perf_counter()
        out = pure_scaled_dot_product_attention(
            q=q, k=k, v=v, is_causal=is_causal, scale=scale
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Approximate working set of naive SDPA (float32): Q + K + V + logits + attn + out
        elem_bytes = 4
        est_bytes = (
            B * H * (TQ * D + TK * D + TK * D + TQ * TK + TQ * TK + TQ * D) * elem_bytes
        )
        peak_mem_bytes = int(est_bytes)

    # Provide a simple, consistent scalar score (used by evaluator as a sanity signal)
    reported_score = float(out.abs().mean().item())
    perf = {
        "latency_ms": float(latency_ms),
        "peak_mem_bytes": int(peak_mem_bytes),
        "used_triton": used_triton,
        "device": "cuda" if used_triton else "cpu",
        "B": int(B),
        "H": int(H),
        "TQ": int(TQ),
        "TK": int(TK),
        "D": int(D),
    }
    return out, reported_score, perf
