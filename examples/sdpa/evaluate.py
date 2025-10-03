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

    if not isinstance(result, tuple) or len(result) != 2:
        return False, "Result must be a tuple: (outputs, reported_score)"

    outputs, reported_score = result

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

    return True, None


def aggregate_sdpa_metrics(
    results: List[Tuple[Any, float]],
    results_dir: str,
) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    rmses: List[float] = []
    reported_scores: List[float] = []
    recomputed_scores: List[float] = []
    max_abs_errors: List[float] = []

    extras: List[Dict[str, Any]] = []

    for idx, (outputs, reported_score) in enumerate(results):
        ref = REFERENCE_OUTPUTS[idx]
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

        score = float(1.0 / (1.0 + rmse))
        rmses.append(rmse)
        max_abs_errors.append(max_err)
        reported_scores.append(float(reported_score))
        recomputed_scores.append(score)

        extras.append(
            {
                "case_index": idx,
                "rmse": rmse,
                "max_abs_error": max_err,
                "reported_score": float(reported_score),
                "recomputed_score": score,
            }
        )

    combined = float(np.mean(recomputed_scores))
    metrics: Dict[str, Any] = {
        "combined_score": combined,
        "public": {
            "mean_rmse": float(np.mean(rmses)),
            "max_rmse": float(np.max(rmses)),
            "max_abs_error": float(np.max(max_abs_errors)),
        },
        "private": {
            "mean_reported_score": float(np.mean(reported_scores)),
            "mean_recomputed_score": float(np.mean(recomputed_scores)),
            "score_reporting_error": float(
                np.max(np.abs(np.array(reported_scores) - np.array(recomputed_scores)))
            ),
        },
        "extra_data": extras,
    }

    # Save simplified arrays for quick inspection
    try:
        os.makedirs(results_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(results_dir, "sdpa_metrics.npz"),
            rmses=np.array(rmses, dtype=np.float64),
            reported_scores=np.array(reported_scores, dtype=np.float64),
            recomputed_scores=np.array(recomputed_scores, dtype=np.float64),
            max_abs_errors=np.array(max_abs_errors, dtype=np.float64),
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

