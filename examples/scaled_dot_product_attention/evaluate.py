"""Evaluator for scaled dot-product attention optimization example."""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from shinka.core import run_shinka_eval


def _build_attention_cases() -> List[Dict[str, np.ndarray]]:
    rng = np.random.default_rng(314159)
    configs = [
        (2, 3, 4, 3),
        (3, 5, 6, 4),
        (4, 4, 5, 5),
    ]
    cases: List[Dict[str, np.ndarray]] = []
    for num_queries, num_keys, depth, value_dim in configs:
        queries = rng.normal(loc=0.0, scale=0.7, size=(num_queries, depth)).astype(
            np.float32
        )
        keys = rng.normal(loc=0.0, scale=0.8, size=(num_keys, depth)).astype(np.float32)
        values = rng.normal(loc=0.0, scale=1.0, size=(num_keys, value_dim)).astype(
            np.float32
        )
        cases.append({"queries": queries, "keys": keys, "values": values})
    return cases


def _reference_scaled_dot_product_attention(
    queries: np.ndarray, keys: np.ndarray, values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    depth = queries.shape[-1]
    scale = np.sqrt(float(depth))
    logits = (queries.astype(np.float64) @ keys.astype(np.float64).T) / scale
    logits -= np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    attention = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    outputs = attention @ values.astype(np.float64)
    return outputs, attention


ATTENTION_CASES = _build_attention_cases()
REFERENCE_RESULTS = [
    _reference_scaled_dot_product_attention(
        case["queries"], case["keys"], case["values"]
    )
    for case in ATTENTION_CASES
]
EXPECTED_SHAPES = [
    (
        case["queries"].shape[0],
        case["keys"].shape[0],
        case["queries"].shape[1],
        case["values"].shape[1],
    )
    for case in ATTENTION_CASES
]
VALIDATION_STATE = {"index": 0}


def get_scaled_attention_kwargs(run_index: int) -> Dict[str, Any]:
    case = ATTENTION_CASES[run_index % len(ATTENTION_CASES)]
    return {
        "queries": case["queries"].copy(),
        "keys": case["keys"].copy(),
        "values": case["values"].copy(),
    }


def validate_attention_result(
    run_output: Tuple[np.ndarray, np.ndarray, float]
) -> Tuple[bool, Optional[str]]:
    idx = VALIDATION_STATE["index"]
    if idx >= len(EXPECTED_SHAPES):
        return False, "Received more results than expected"

    VALIDATION_STATE["index"] += 1
    expected_num_queries, expected_num_keys, _, expected_value_dim = EXPECTED_SHAPES[idx]

    if not isinstance(run_output, tuple) or len(run_output) != 3:
        return False, "Result must be a tuple: (outputs, attention, reported_score)"

    outputs, attention, reported_score = run_output

    if not isinstance(outputs, np.ndarray):
        outputs = np.array(outputs)
    if not isinstance(attention, np.ndarray):
        attention = np.array(attention)

    if outputs.shape != (expected_num_queries, expected_value_dim):
        return (
            False,
            f"Outputs shape mismatch. Expected {(expected_num_queries, expected_value_dim)},"
            f" got {outputs.shape}",
        )
    if attention.shape != (expected_num_queries, expected_num_keys):
        return (
            False,
            f"Attention weights shape mismatch. Expected {(expected_num_queries, expected_num_keys)},"
            f" got {attention.shape}",
        )

    if not np.all(np.isfinite(outputs)):
        return False, "Outputs contain non-finite values"
    if not np.all(np.isfinite(attention)):
        return False, "Attention weights contain non-finite values"

    min_weight = np.min(attention)
    if min_weight < -1e-6:
        return False, f"Attention weights must be non-negative. Min={min_weight}"

    row_sums = np.sum(attention, axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        return (
            False,
            "Attention rows must sum to approximately 1.0",
        )

    if not np.isfinite(reported_score):
        return False, "Reported score must be a finite float"

    return True, None


def aggregate_scaled_attention_metrics(
    results: List[Tuple[np.ndarray, np.ndarray, float]],
    results_dir: str,
) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    output_rmses: List[float] = []
    weight_rmses: List[float] = []
    reported_scores: List[float] = []
    recomputed_scores: List[float] = []
    max_output_errors: List[float] = []
    extras: List[Dict[str, Any]] = []

    for idx, run_output in enumerate(results):
        outputs, attention, reported_score = run_output
        outputs = np.array(outputs, dtype=np.float64)
        attention = np.array(attention, dtype=np.float64)

        ref_outputs, ref_attention = REFERENCE_RESULTS[idx]

        diff_outputs = outputs - ref_outputs
        output_rmse = float(np.sqrt(np.mean(diff_outputs**2)))
        weight_rmse = float(np.sqrt(np.mean((attention - ref_attention) ** 2)))
        max_error = float(np.max(np.abs(diff_outputs)))
        recomputed_score = float(1.0 / (1.0 + output_rmse))

        output_rmses.append(output_rmse)
        weight_rmses.append(weight_rmse)
        max_output_errors.append(max_error)
        reported_scores.append(float(reported_score))
        recomputed_scores.append(recomputed_score)

        extras.append(
            {
                "case_index": idx,
                "outputs": outputs,
                "attention": attention,
                "reference_outputs": ref_outputs,
                "reference_attention": ref_attention,
                "reported_score": float(reported_score),
                "recomputed_score": recomputed_score,
                "output_rmse": output_rmse,
                "weight_rmse": weight_rmse,
                "max_abs_output_error": max_error,
            }
        )

    combined_score = float(np.mean(recomputed_scores))
    public_metrics = {
        "mean_output_rmse": float(np.mean(output_rmses)),
        "mean_weight_rmse": float(np.mean(weight_rmses)),
        "max_output_error": float(np.max(max_output_errors)),
    }
    private_metrics = {
        "mean_reported_score": float(np.mean(reported_scores)),
        "mean_recomputed_score": float(np.mean(recomputed_scores)),
        "score_reporting_error": float(
            np.max(np.abs(np.array(reported_scores) - np.array(recomputed_scores)))
        ),
    }

    metrics: Dict[str, Any] = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
        "extra_data": extras,
    }

    extra_path = os.path.join(results_dir, "attention_metrics.npz")
    try:
        np.savez_compressed(
            extra_path,
            output_rmses=np.array(output_rmses, dtype=np.float64),
            weight_rmses=np.array(weight_rmses, dtype=np.float64),
            reported_scores=np.array(reported_scores, dtype=np.float64),
            recomputed_scores=np.array(recomputed_scores, dtype=np.float64),
            max_output_errors=np.array(max_output_errors, dtype=np.float64),
        )
        print(f"Saved per-case metrics to {extra_path}")
    except Exception as exc:  # pragma: no cover
        metrics["extra_npz_save_error"] = str(exc)

    return metrics


def main(program_path: str, results_dir: str) -> None:
    print(f"Evaluating program: {program_path}")
    print(f"Results directory: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    num_runs = len(ATTENTION_CASES)
    VALIDATION_STATE["index"] = 0

    def _aggregate(results: List[Tuple[np.ndarray, np.ndarray, float]]) -> Dict[str, Any]:
        return aggregate_scaled_attention_metrics(results, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_attention",
        num_runs=num_runs,
        get_experiment_kwargs=get_scaled_attention_kwargs,
        validate_fn=validate_attention_result,
        aggregate_metrics_fn=_aggregate,
    )

    if correct:
        print("Evaluation completed successfully.")
    else:
        print(f"Evaluation finished with validation errors: {error_msg}")

    print("Metrics summary:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scaled dot-product attention evaluator using shinka.eval"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program containing 'run_attention'",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory to store metrics and artifacts",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
