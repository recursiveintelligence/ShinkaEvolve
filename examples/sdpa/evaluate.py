"""Evaluator for Triton SDPA (scaled dot-product attention) example.

This evaluator runs multiple SDPA test cases, validates shapes and numerics,
and aggregates a score based on RMSE versus a PyTorch reference
implementation. It is modeled after the circle_packing example structure.
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from shinka.core import run_shinka_eval


def _torch_reference_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Numerically stable PyTorch SDPA used as the ground truth for scoring."""
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, H, TQ, D = q.shape
    _, _, TK, _ = k.shape
    scale_val = (1.0 / float(np.sqrt(D))) if scale is None else float(scale)

    logits = torch.matmul(q.to(torch.float64), k.transpose(-2, -1).to(torch.float64))
    logits = logits * scale_val

    if is_causal:
        q_idx = torch.arange(TQ, device=logits.device).view(TQ, 1)
        k_idx = torch.arange(TK, device=logits.device).view(1, TK)
        causal_mask = (k_idx > q_idx)  # [TQ, TK]
        logits = logits.masked_fill(causal_mask.view(1, 1, TQ, TK), float("-inf"))

    logits = logits - logits.amax(dim=-1, keepdim=True)
    attn = torch.exp(logits)
    denom = attn.sum(dim=-1, keepdim=True)
    attn = attn / denom.clamp_min(1e-12)
    out = torch.matmul(attn, v.to(torch.float64))
    return out.to(torch.float32)


def _build_cases(seed: int = 123) -> List[Dict[str, Any]]:
    g = torch.Generator().manual_seed(seed)
    shapes = [
        # (B, H, TQ, TK, D, causal)
        (1, 1, 16, 16, 32, False),
        (2, 2, 32, 24, 64, False),
        (1, 4, 33, 33, 40, True),
        (2, 8, 48, 48, 64, True),
        (1, 2, 17, 29, 32, False),
    ]
    cases: List[Dict[str, Any]] = []
    for (B, H, TQ, TK, D, causal) in shapes:
        q = torch.randn(B, H, TQ, D, dtype=torch.float32, generator=g)
        k = torch.randn(B, H, TK, D, dtype=torch.float32, generator=g)
        v = torch.randn(B, H, TK, D, dtype=torch.float32, generator=g)
        cases.append({"q": q, "k": k, "v": v, "is_causal": causal})
    return cases


CASES = _build_cases()
REFERENCE_OUTPUTS = [
    _torch_reference_sdpa(case["q"], case["k"], case["v"], case["is_causal"]).detach().cpu()
    for case in CASES
]
EXPECTED_OUTPUT_SHAPES = [tuple(ref.shape) for ref in REFERENCE_OUTPUTS]
VALIDATION_STATE = {"index": 0}


def get_sdpa_kwargs(run_index: int) -> Dict[str, Any]:
    case = CASES[run_index % len(CASES)]
    # Provide fresh tensors to the program (avoid in-place modifications)
    return {
        "q": case["q"].clone(),
        "k": case["k"].clone(),
        "v": case["v"].clone(),
        "is_causal": bool(case["is_causal"]),
    }


def validate_sdpa_result(result: Any) -> Tuple[bool, Optional[str]]:
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

    # Accept both numpy arrays and torch tensors
    if isinstance(outputs, np.ndarray):
        out_shape = outputs.shape
        finite_ok = np.all(np.isfinite(outputs))
    elif isinstance(outputs, torch.Tensor):
        out_shape = tuple(outputs.shape)
        finite_ok = torch.isfinite(outputs).all().item()
    else:
        return False, "Outputs must be a numpy array or a torch.Tensor"

    if out_shape != EXPECTED_OUTPUT_SHAPES[idx]:
        return False, f"Output shape mismatch. Expected {EXPECTED_OUTPUT_SHAPES[idx]}, got {out_shape}"

    if not finite_ok:
        return False, "Outputs contain non-finite values"

    try:
        float(reported_score)
    except Exception:
        return False, "Reported score must be a finite float"

    # If perf is present, sanity check expected keys
    if perf is not None:
        if not isinstance(perf, dict):
            return False, "perf must be a dict if provided"
        for k in ("latency_ms", "peak_mem_bytes"):
            if k not in perf:
                return False, f"perf missing required key: {k}"
            try:
                _ = float(perf[k])
            except Exception:
                return False, f"perf[{k}] must be numeric"

    return True, None


def aggregate_sdpa_metrics(
    results: List[Tuple[Any, float]],
    results_dir: str,
) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    # Weights: prioritize speed and memory
    SPEED_W, MEMORY_W, ACCURACY_W = 0.6, 0.3, 0.1

    def _estimate_naive_bytes(B: int, H: int, TQ: int, TK: int, D: int, elem_bytes: int = 4) -> int:
        # Rough working set for naive SDPA in float32: Q + K + V + logits + attn + out
        return int(elem_bytes * (B * H * (TQ * D + TK * D + TK * D + TQ * TK + TQ * TK + TQ * D)))

    def _time_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> float:
        # Warmup + timed mean (ms) on CPU
        for _ in range(1):
            _ = _torch_reference_sdpa(q, k, v, is_causal)
        import time as _time
        t0 = _time.perf_counter()
        for _ in range(3):
            _ = _torch_reference_sdpa(q, k, v, is_causal)
        t1 = _time.perf_counter()
        return float((t1 - t0) / 3.0 * 1000.0)

    rmses: List[float] = []
    reported_scores: List[float] = []
    accuracy_scores: List[float] = []
    max_abs_errors: List[float] = []
    cand_latencies_ms: List[float] = []
    ref_latencies_ms: List[float] = []
    speedups: List[float] = []
    cand_peak_mem: List[float] = []
    ref_mem_est: List[float] = []
    mem_savings: List[float] = []

    extras: List[Dict[str, Any]] = []

    for idx, item in enumerate(results):
        # Unpack allowing optional perf dict in third element
        if len(item) == 2:
            outputs, reported_score = item
            perf = None
        else:
            outputs, reported_score, perf = item

        ref = REFERENCE_OUTPUTS[idx]
        # Accuracy vs reference
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

        accuracy_score = float(1.0 / (1.0 + rmse))
        rmses.append(rmse)
        max_abs_errors.append(max_err)
        reported_scores.append(float(reported_score))
        accuracy_scores.append(accuracy_score)

        # Gather shapes for memory estimation
        case = CASES[idx]
        B, H, TQ, D = case["q"].shape
        TK = case["k"].shape[2]

        # Candidate latency/memory from perf if provided; else estimate
        if isinstance(perf, dict):
            cand_latency = float(perf.get("latency_ms", float("nan")))
            cand_mem = float(perf.get("peak_mem_bytes", _estimate_naive_bytes(B, H, TQ, TK, D)))
        else:
            cand_latency = float("nan")
            cand_mem = float(_estimate_naive_bytes(B, H, TQ, TK, D))

        # Reference latency/memory estimates
        ref_latency = _time_reference(case["q"], case["k"], case["v"], bool(case["is_causal"]))
        ref_bytes = float(_estimate_naive_bytes(B, H, TQ, TK, D))

        # Compute ratios and scores
        if np.isnan(cand_latency) or cand_latency <= 0:
            speedup = 1.0  # neutral
        else:
            speedup = float(ref_latency / cand_latency)

        if cand_mem <= 0:
            mem_ratio = 1.0
        else:
            mem_ratio = float(ref_bytes / cand_mem)

        # Map ratios -> [0,1): s/(1+s) has 0.5 at 1x, asymptote to 1
        speed_score = float(speedup / (1.0 + speedup))
        memory_score = float(mem_ratio / (1.0 + mem_ratio))

        combined = SPEED_W * speed_score + MEMORY_W * memory_score + ACCURACY_W * accuracy_score

        cand_latencies_ms.append(cand_latency)
        ref_latencies_ms.append(ref_latency)
        speedups.append(speedup)
        cand_peak_mem.append(cand_mem)
        ref_mem_est.append(ref_bytes)
        mem_savings.append(mem_ratio)

        extras.append(
            {
                "case_index": idx,
                "rmse": rmse,
                "accuracy_score": accuracy_score,
                "max_abs_error": max_err,
                "reported_score": float(reported_score),
                "cand_latency_ms": cand_latency,
                "ref_latency_ms": ref_latency,
                "speedup": speedup,
                "cand_peak_mem_bytes": cand_mem,
                "ref_peak_mem_est_bytes": ref_bytes,
                "mem_saving_ratio": mem_ratio,
                "combined_case_score": combined,
            }
        )

    combined_score = float(np.mean([e["combined_case_score"] for e in extras]))

    metrics: Dict[str, Any] = {
        "combined_score": combined_score,
        "public": {
            "mean_speedup": float(np.mean(speedups)),
            "mean_mem_saving": float(np.mean(mem_savings)),
            "mean_cand_latency_ms": float(np.nanmean(cand_latencies_ms)),
            "mean_rmse": float(np.mean(rmses)),
        },
        "private": {
            "weights": {"speed": SPEED_W, "memory": MEMORY_W, "accuracy": ACCURACY_W},
            "mean_reported_score": float(np.mean(reported_scores)),
            "mean_accuracy_score": float(np.mean(accuracy_scores)),
            "max_abs_error": float(np.max(max_abs_errors)),
            "mean_ref_latency_ms": float(np.mean(ref_latencies_ms)),
            "mean_cand_peak_mem_bytes": float(np.mean(cand_peak_mem)),
            "mean_ref_mem_est_bytes": float(np.mean(ref_mem_est)),
        },
        "extra_data": extras,
    }

    # Save simplified arrays for quick inspection
    try:
        os.makedirs(results_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(results_dir, "sdpa_metrics.npz"),
            speedups=np.array(speedups, dtype=np.float64),
            mem_savings=np.array(mem_savings, dtype=np.float64),
            cand_latencies_ms=np.array(cand_latencies_ms, dtype=np.float64),
            ref_latencies_ms=np.array(ref_latencies_ms, dtype=np.float64),
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

    def _agg_fn(res: List[Tuple[Any, float]]) -> Dict[str, Any]:
        return aggregate_sdpa_metrics(res, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_sdpa",
        num_runs=num_runs,
        get_experiment_kwargs=get_sdpa_kwargs,
        validate_fn=validate_sdpa_result,
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
    parser = argparse.ArgumentParser(description="SDPA evaluator using shinka.eval")
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program containing 'run_sdpa'",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store metrics and artifacts",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
