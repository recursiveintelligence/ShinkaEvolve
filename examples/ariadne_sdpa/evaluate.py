"""Evaluator for the Ariadne streaming SDPA example.

This module mirrors the structure of other examples in the repository while providing
a richer score that reflects accuracy, latency, and memory efficiency relative to a
materialize-then-softmax baseline. The default Ariadne program is treated as a modest
baseline so that evolutionary improvements have room to register on the final score.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from shinka.core import run_shinka_eval

BASE_DIR = Path(__file__).resolve().parent


def _torch_reference_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """High-precision SDPA reference used for both accuracy checks and timing."""
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H, M, D = q.shape
    _, _, N, _ = k.shape
    scale_val = (1.0 / float(np.sqrt(D))) if scale is None else float(scale)

    q64 = q.to(torch.float64)
    k64 = k.to(torch.float64)
    v64 = v.to(torch.float64)

    logits = torch.matmul(q64, k64.transpose(-2, -1))
    logits = logits * scale_val

    if is_causal:
        q_idx = torch.arange(M, device=logits.device).view(M, 1)
        k_idx = torch.arange(N, device=logits.device).view(1, N)
        causal_mask = k_idx > q_idx
        logits = logits.masked_fill(causal_mask.view(1, 1, M, N), float("-inf"))

    logits = logits - logits.amax(dim=-1, keepdim=True)
    attn = torch.exp(logits)
    denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-16)
    attn = attn / denom
    out = torch.matmul(attn, v64)
    return out.to(torch.float32)


def _time_reference_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    scale: Optional[float],
    num_warmup: int = 1,
    num_iters: int = 3,
) -> float:
    """Time the reference SDPA on CPU to obtain a baseline latency in milliseconds."""
    for _ in range(max(0, num_warmup)):
        _ = _torch_reference_sdpa(q, k, v, is_causal=is_causal, scale=scale)
    t0 = time.perf_counter()
    for _ in range(max(1, num_iters)):
        _ = _torch_reference_sdpa(q, k, v, is_causal=is_causal, scale=scale)
    t1 = time.perf_counter()
    return float((t1 - t0) / max(1, num_iters) * 1000.0)


def _estimate_naive_bytes(B: int, H: int, M: int, N: int, D: int, elem_bytes: int) -> int:
    """Approximate bytes touched by a naive SDPA: Q + K + V + logits + attn + out."""
    return int(elem_bytes * (B * H * (M * D + N * D + N * D + M * N + M * N + M * D)))


def _estimate_stream_bytes(
    B: int,
    H: int,
    M: int,
    N: int,
    D: int,
    size_q: int,
    size_k: int,
    size_v: int,
    accum_bytes: int = 4,
) -> int:
    """Estimate bytes for an exact streaming SDPA implementation."""
    return int(B * H * (M * D * size_q + N * D * size_k + N * D * size_v + M * D * accum_bytes))


def _supports_fp8() -> bool:
    return all(hasattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2"))


def _build_cases(seed: int = 614) -> List[Dict[str, Any]]:
    """Construct evaluation cases covering different shapes, dtypes, and masking."""
    g = torch.Generator().manual_seed(seed)

    float32 = torch.float32
    float16 = torch.float16 if torch.cuda.is_available() else torch.float32

    cases: List[Dict[str, Any]] = []
    base_specs: Iterable[Dict[str, Any]] = [
        dict(B=1, H=1, M=32, N=32, D=64, causal=False, dtype=float32, desc="cpu_fp32_small"),
        dict(B=2, H=4, M=48, N=40, D=64, causal=True, dtype=float32, desc="cpu_fp32_medium"),
        dict(B=1, H=8, M=64, N=64, D=80, causal=False, dtype=float16, desc="mixed_precision"),
        dict(B=2, H=4, M=96, N=80, D=64, causal=True, dtype=float16, desc="causal_long"),
    ]

    for spec in base_specs:
        B, H, M, N, D = spec["B"], spec["H"], spec["M"], spec["N"], spec["D"]
        dtype = spec["dtype"]
        causal = bool(spec["causal"])
        q = torch.randn(B, H, M, D, generator=g, dtype=torch.float32)
        k = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
        v = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
        if dtype != torch.float32:
            q = q.to(dtype=dtype)
            k = k.to(dtype=dtype)
            v = v.to(dtype=dtype)
        cases.append(
            {
                "q": q,
                "k": k,
                "v": v,
                "is_causal": causal,
                "scale": None,
                "description": spec["desc"],
            }
        )

    # Optional FP8 case if the environment supports it and Triton is likely available.
    if _supports_fp8() and torch.cuda.is_available():
        fp8_dtype = getattr(torch, "float8_e4m3fn")
        q = torch.randn(2, 4, 64, 128, generator=g, dtype=torch.float16).cuda()
        k = torch.randn(2, 4, 64, 128, generator=g, dtype=torch.float16).cuda()
        v = torch.randn(2, 4, 64, 128, generator=g, dtype=torch.float16).cuda()
        k_fp8 = k.to(dtype=fp8_dtype)
        v_fp8 = v.to(dtype=fp8_dtype)
        scales_k = torch.full(k_fp8.shape[:3], 0.25, device=k_fp8.device, dtype=torch.float32)
        scales_v = torch.full(v_fp8.shape[:3], 0.5, device=v_fp8.device, dtype=torch.float32)
        k_fp8.scales = scales_k  # type: ignore[attr-defined]
        v_fp8.scales = scales_v  # type: ignore[attr-defined]
        cases.append(
            {
                "q": q,
                "k": k_fp8,
                "v": v_fp8,
                "is_causal": False,
                "scale": None,
                "description": "gpu_fp8_stream",
            }
        )

    return cases


CASES = _build_cases()
REFERENCE_OUTPUTS = [
    _torch_reference_sdpa(case["q"], case["k"], case["v"], case["is_causal"], case["scale"]).detach().cpu()
    for case in CASES
]
REFERENCE_LATENCIES_MS = [
    _time_reference_sdpa(case["q"], case["k"], case["v"], case["is_causal"], case["scale"])
    for case in CASES
]
NAIVE_BYTE_BASELINES = [
    _estimate_naive_bytes(
        case["q"].shape[0],
        case["q"].shape[1],
        case["q"].shape[2],
        case["k"].shape[2],
        case["q"].shape[3],
        max(case["q"].element_size(), case["k"].element_size(), case["v"].element_size()),
    )
    for case in CASES
]
EXPECTED_OUTPUT_SHAPES = [tuple(ref.shape) for ref in REFERENCE_OUTPUTS]
VALIDATION_STATE = {"index": 0}


def get_ariadne_kwargs(run_index: int) -> Dict[str, Any]:
    """Provide fresh tensors for the program under evaluation."""
    case = CASES[run_index % len(CASES)]
    return {
        "q": case["q"].clone(),
        "k": case["k"].clone(),
        "v": case["v"].clone(),
        "scale": case["scale"],
        "causal": case["is_causal"],
        "use_kahan": True,
    }


def validate_ariadne_result(result: Any) -> Tuple[bool, Optional[str]]:
    """Ensure the candidate program returned the correct structure and metadata."""
    idx = VALIDATION_STATE["index"]
    if idx >= len(EXPECTED_OUTPUT_SHAPES):
        return False, "Received more results than expected"
    VALIDATION_STATE["index"] += 1

    if not isinstance(result, tuple) or len(result) not in (2, 3):
        return False, "Result must be a tuple: (outputs, reported_score[, perf_dict])"

    if len(result) == 2:
        outputs, reported_score = result
        perf = None
    else:
        outputs, reported_score, perf = result

    if isinstance(outputs, torch.Tensor):
        out_shape = tuple(outputs.shape)
        finite_ok = bool(torch.isfinite(outputs).all().item())
    else:
        arr = np.asarray(outputs)
        out_shape = arr.shape
        finite_ok = bool(np.isfinite(arr).all())

    if out_shape != EXPECTED_OUTPUT_SHAPES[idx]:
        return False, f"Output shape mismatch. Expected {EXPECTED_OUTPUT_SHAPES[idx]}, got {out_shape}"

    if not finite_ok:
        return False, "Outputs contain non-finite values"

    # reported_score can be a dict or float; just make sure it is representable.
    try:
        if isinstance(reported_score, dict):
            _ = float(reported_score.get("ariadne_score", 0.0))
        else:
            _ = float(reported_score)
    except Exception:
        return False, "Reported score must be castable to float"

    if perf is not None:
        if not isinstance(perf, dict):
            return False, "perf must be a dict if provided"
        for key in ("latency_ms", "peak_mem_bytes", "stream_bytes_est"):
            if key not in perf:
                return False, f"perf missing required key: {key}"
            try:
                _ = float(perf[key])
            except Exception:
                return False, f"perf[{key}] must be numeric"

    return True, None


def _ratio_to_score(ratio: float) -> float:
    if not np.isfinite(ratio) or ratio <= 0:
        return 0.0
    return float(ratio / (1.0 + ratio))


def aggregate_ariadne_metrics(
    results: List[Tuple[Any, Any]],
    results_dir: str,
) -> Dict[str, Any]:
    """Combine per-case metrics into a single scalar with supporting diagnostics."""
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    SPEED_W, MEMORY_W, STREAM_W, ACCURACY_W, SELF_W = 0.45, 0.2, 0.15, 0.15, 0.05

    extras: List[Dict[str, Any]] = []
    speedups: List[float] = []
    memory_ratios: List[float] = []
    stream_ratios: List[float] = []
    rmses: List[float] = []
    max_abs_errors: List[float] = []
    candidate_latencies: List[float] = []
    reference_latencies: List[float] = []
    candidate_peak_mem: List[float] = []
    naive_bytes_list: List[float] = []

    for idx, item in enumerate(results):
        if len(item) == 2:
            outputs, reported_score = item
            perf = {}
        else:
            outputs, reported_score, perf = item

        ref = REFERENCE_OUTPUTS[idx]
        ref_latency = REFERENCE_LATENCIES_MS[idx]
        naive_bytes = NAIVE_BYTE_BASELINES[idx]

        if isinstance(outputs, torch.Tensor):
            out = outputs.detach().cpu().to(torch.float64)
            ref64 = ref.to(torch.float64)
            diff = (out - ref64).abs()
            rmse = float(torch.sqrt((diff ** 2).mean()).item())
            max_err = float(diff.max().item())
        else:
            out = np.asarray(outputs, dtype=np.float64)
            ref64 = ref.detach().cpu().numpy().astype(np.float64)
            diff = np.abs(out - ref64)
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            max_err = float(np.max(diff))

        try:
            candidate_latency = float(perf.get("latency_ms", float("nan")))
        except Exception:
            candidate_latency = float("nan")

        try:
            peak_mem = float(perf.get("peak_mem_bytes", float("nan")))
        except Exception:
            peak_mem = float("nan")

        try:
            stream_bytes_est = float(perf.get("stream_bytes_est", float("nan")))
        except Exception:
            stream_bytes_est = float("nan")

        if isinstance(reported_score, dict):
            self_report = float(reported_score.get("ariadne_score", 0.0))
        else:
            self_report = float(reported_score)

        speedup = ref_latency / candidate_latency if np.isfinite(candidate_latency) and candidate_latency > 0 else 1.0
        mem_ratio = naive_bytes / peak_mem if np.isfinite(peak_mem) and peak_mem > 0 else 1.0
        stream_ratio = naive_bytes / stream_bytes_est if np.isfinite(stream_bytes_est) and stream_bytes_est > 0 else 1.0

        speed_score = _ratio_to_score(speedup)
        memory_score = _ratio_to_score(mem_ratio)
        stream_score = _ratio_to_score(stream_ratio)
        accuracy_score = float(1.0 / (1.0 + rmse * 1e3))
        self_score = float(max(0.0, min(1.0, self_report)))

        combined = (
            SPEED_W * speed_score
            + MEMORY_W * memory_score
            + STREAM_W * stream_score
            + ACCURACY_W * accuracy_score
            + SELF_W * self_score
        )

        extras.append(
            {
                "case_index": idx,
                "description": CASES[idx]["description"],
                "rmse": rmse,
                "max_abs_error": max_err,
                "speedup_vs_reference": speedup,
                "memory_ratio_vs_naive": mem_ratio,
                "stream_ratio_vs_naive": stream_ratio,
                "reported_score": self_report,
                "candidate_latency_ms": candidate_latency,
                "reference_latency_ms": ref_latency,
                "candidate_peak_mem_bytes": peak_mem,
                "naive_bytes": naive_bytes,
                "combined_case_score": combined,
            }
        )

        speedups.append(speedup)
        memory_ratios.append(mem_ratio)
        stream_ratios.append(stream_ratio)
        rmses.append(rmse)
        max_abs_errors.append(max_err)
        candidate_latencies.append(candidate_latency)
        reference_latencies.append(ref_latency)
        candidate_peak_mem.append(peak_mem)
        naive_bytes_list.append(naive_bytes)

    combined_score = float(np.mean([e["combined_case_score"] for e in extras]))

    metrics: Dict[str, Any] = {
        "combined_score": combined_score,
        "public": {
            "mean_speedup": float(np.mean(speedups)),
            "mean_memory_ratio": float(np.mean(memory_ratios)),
            "mean_stream_ratio": float(np.mean(stream_ratios)),
            "mean_rmse": float(np.mean(rmses)),
        },
        "private": {
            "weights": {
                "speed": SPEED_W,
                "memory": MEMORY_W,
                "stream": STREAM_W,
                "accuracy": ACCURACY_W,
                "reported": SELF_W,
            },
            "max_abs_error": float(np.max(max_abs_errors)),
            "mean_self_report": float(np.mean([e["reported_score"] for e in extras])),
            "mean_candidate_latency_ms": float(np.nanmean(candidate_latencies)),
            "mean_reference_latency_ms": float(np.mean(reference_latencies)),
            "mean_candidate_peak_mem_bytes": float(np.nanmean(candidate_peak_mem)),
            "mean_naive_bytes": float(np.mean(naive_bytes_list)),
        },
        "extra_data": extras,
    }

    try:
        os.makedirs(results_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(results_dir, "ariadne_metrics.npz"),
            speedups=np.array(speedups, dtype=np.float64),
            memory_ratios=np.array(memory_ratios, dtype=np.float64),
            stream_ratios=np.array(stream_ratios, dtype=np.float64),
            candidate_latencies=np.array(candidate_latencies, dtype=np.float64),
            reference_latencies=np.array(reference_latencies, dtype=np.float64),
            rmses=np.array(rmses, dtype=np.float64),
        )
    except Exception as exc:  # pragma: no cover
        metrics.setdefault("private", {})["npz_save_error"] = str(exc)

    return metrics


def main(program_path: str, results_dir: str) -> None:
    print(f"Evaluating program: {program_path}")
    print(f"Results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_runs = len(CASES)
    VALIDATION_STATE["index"] = 0

    def _agg_fn(res: List[Tuple[Any, Any]]) -> Dict[str, Any]:
        return aggregate_ariadne_metrics(res, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_ariadne",
        num_runs=num_runs,
        get_experiment_kwargs=get_ariadne_kwargs,
        validate_fn=validate_ariadne_result,
        aggregate_metrics_fn=_agg_fn,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation finished with validation errors: {error_msg}")

    print("Metrics summary:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_val in value.items():
                print(f"    {sub_key}: {sub_val}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ariadne SDPA evaluator")
    parser.add_argument(
        "--program_path",
        type=str,
        default=str(BASE_DIR / "initial.py"),
        help="Path to the program containing 'run_ariadne'",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(BASE_DIR / "results"),
        help="Directory to store metrics and artifacts",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
