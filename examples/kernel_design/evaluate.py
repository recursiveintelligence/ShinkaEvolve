"""
Evaluation harness for Triton kernel design (RMSNorm, RoPE, ReLU^2).

This mirrors the structure of other examples by using `run_shinka_eval`
to invoke the candidate program's `run_experiment` entrypoint, and
benchmarks correctness, latency, and FLOPs on synthetic LLM-like tensors.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

# Ensure project root is importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import run_shinka_eval directly from its module to avoid heavyweight extras.
WRAP_EVAL_PATH = PROJECT_ROOT / "shinka" / "core" / "wrap_eval.py"
spec = importlib.util.spec_from_file_location("shinka.core.wrap_eval", WRAP_EVAL_PATH)
wrap_eval = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(wrap_eval)  # type: ignore
run_shinka_eval = wrap_eval.run_shinka_eval
save_json_results = wrap_eval.save_json_results


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def select_device(device_arg: str) -> torch.device:
    """Return requested device, requiring CUDA for Triton benchmarks."""
    if device_arg in ("auto", None):
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA device is required for kernel benchmark.")
    dev = torch.device(device_arg)
    if dev.type != "cuda":
        raise RuntimeError("CUDA device is required for kernel benchmark.")
    return dev


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """Map CLI dtype to torch dtype, with sensible defaults per device."""
    if dtype_arg in ("auto", None):
        return torch.float16 if device.type == "cuda" else torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_arg not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_arg}")
    return mapping[dtype_arg]


def validate_kernel_suite(run_result: Any) -> Tuple[bool, str | None]:
    """Ensure the candidate exposes the expected callables."""
    required = ["fast_rmsnorm", "apply_rope_triton", "fast_relusqr"]
    if not isinstance(run_result, dict):
        return False, "run_experiment must return a dict of callables."
    for name in required:
        fn = run_result.get(name)
        if fn is None or not callable(fn):
            return False, f"Missing callable '{name}' in run_experiment output."
    return True, None


def _clone_requires_grad(x: torch.Tensor) -> torch.Tensor:
    """Clone tensor and enable gradients."""
    return x.clone().detach().requires_grad_(True)


def measure_latency(
    fn: Callable[..., torch.Tensor],
    arg_builder: Callable[[], Tuple[Any, ...]],
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
    backward: bool = True,
) -> Tuple[float, float]:
    """
    Measure forward/backward latency (ms) for a callable using new args each time.
    Uses CUDA events when available for accuracy.
    """
    forward_times: List[float] = []
    backward_times: List[float] = []

    def _synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(max(warmup_iters, 0)):
        args = arg_builder()
        out = fn(*args)
        if backward:
            (out.sum()).backward()
    _synchronize()

    if device.type == "cuda":
        f_start = torch.cuda.Event(enable_timing=True)
        f_mid = torch.cuda.Event(enable_timing=True)
        f_end = torch.cuda.Event(enable_timing=True)

        for _ in range(bench_iters):
            args = arg_builder()
            f_start.record()
            out = fn(*args)
            f_mid.record()
            if backward:
                (out.sum()).backward()
            f_end.record()
            _synchronize()
            forward_times.append(f_start.elapsed_time(f_mid))
            backward_times.append(f_mid.elapsed_time(f_end) if backward else 0.0)
    else:
        for _ in range(bench_iters):
            args = arg_builder()
            t0 = time.perf_counter()
            out = fn(*args)
            t1 = time.perf_counter()
            if backward:
                (out.sum()).backward()
            t2 = time.perf_counter()
            forward_times.append((t1 - t0) * 1000.0)
            backward_times.append((t2 - t1) * 1000.0 if backward else 0.0)

    # Median is robust to outliers
    forward_ms = float(np.median(forward_times)) if forward_times else 0.0
    backward_ms = float(np.median(backward_times)) if backward_times else 0.0
    return forward_ms, backward_ms


# -----------------------------------------------------------------------------
# Synthetic workloads & references
# -----------------------------------------------------------------------------

def make_llm_like_inputs(
    device: torch.device, dtype: torch.dtype
) -> Dict[str, torch.Tensor]:
    """Generate small-but-realistic tensors that mimic LLM shapes."""
    g = torch.Generator(device=device).manual_seed(2024)

    rms_x = torch.randn((4, 128, 1024), device=device, dtype=dtype, generator=g)
    relu_x = torch.randn((4, 128, 2048), device=device, dtype=dtype, generator=g)

    B, T, H, D = 2, 128, 16, 128
    rope_x = torch.randn((B, T, H, D), device=device, dtype=dtype, generator=g)

    # Standard RoPE frequency table
    half_dim = D // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=device) / float(half_dim)))
    t = torch.arange(T, device=device)
    freqs = torch.einsum("t,d->td", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)

    return {
        "rms_x": rms_x,
        "relu_x": relu_x,
        "rope_x": rope_x,
        "cos": cos,
        "sin": sin,
        "eps": 1e-5,
    }


def reference_rmsnorm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def reference_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)


def reference_relusqr(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2


# -----------------------------------------------------------------------------
# FLOP estimates and scoring
# -----------------------------------------------------------------------------

def estimate_rmsnorm_flops(shape: Tuple[int, ...]) -> float:
    numel = int(np.prod(shape))
    hidden = shape[-1]
    reduce_ops = (hidden - 1) * (numel // hidden)
    forward = numel * 2 + reduce_ops  # square + scale
    backward = forward * 2
    return float(forward + backward)


def estimate_rope_flops(shape: Tuple[int, ...]) -> float:
    num_pairs = int(np.prod(shape[:-1]) * (shape[-1] // 2))
    forward = num_pairs * 8  # 4 mul + 2 add approx, scaled
    backward = forward  # symmetric for this op
    return float(forward + backward)


def estimate_relusqr_flops(shape: Tuple[int, ...]) -> float:
    numel = int(np.prod(shape))
    forward = numel * 2  # mul + compare/select
    backward = numel * 2
    return float(forward + backward)


def kernel_score(
    forward_err: float,
    grad_err: float,
    baseline_fwd_ms: float,
    baseline_bwd_ms: float,
    cand_fwd_ms: float,
    cand_bwd_ms: float,
) -> float:
    """Combine accuracy and speed into a single scalar."""
    acc_term = 1.0 / (1.0 + forward_err + grad_err + 1e-9)
    speed_f = baseline_fwd_ms / max(cand_fwd_ms, 1e-6)
    speed_b = baseline_bwd_ms / max(cand_bwd_ms, 1e-6)
    speed_term = 0.6 * speed_f + 0.4 * speed_b
    return float(acc_term * speed_term)


# -----------------------------------------------------------------------------
# Per-kernel evaluation
# -----------------------------------------------------------------------------

def eval_rmsnorm(
    kernels: Dict[str, Callable[..., torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, float]:
    fn = kernels["fast_rmsnorm"]
    x_base = inputs["rms_x"]
    eps = inputs["eps"]

    # Correctness check
    ref_x = _clone_requires_grad(x_base)
    ref_out = reference_rmsnorm(ref_x, eps)
    ref_out.sum().backward()
    ref_grad = ref_x.grad.detach()

    test_x = _clone_requires_grad(x_base)
    test_out = fn(test_x, eps)
    test_out.sum().backward()
    test_grad = test_x.grad.detach()

    forward_err = float((test_out - ref_out).abs().max().item())
    grad_err = float((test_grad - ref_grad).abs().max().item())

    # Latency
    ref_args = lambda: (_clone_requires_grad(x_base), eps)
    cand_args = lambda: (_clone_requires_grad(x_base), eps)
    ref_fwd, ref_bwd = measure_latency(
        reference_rmsnorm, ref_args, device, warmup_iters, bench_iters, backward=True
    )
    cand_fwd, cand_bwd = measure_latency(
        fn, cand_args, device, warmup_iters, bench_iters, backward=True
    )

    flops = estimate_rmsnorm_flops(tuple(x_base.shape))
    thrpt_gflops = flops / max(cand_fwd + cand_bwd, 1e-3) / 1e6

    score = kernel_score(
        forward_err, grad_err, ref_fwd, ref_bwd, cand_fwd, cand_bwd
    )
    return {
        "forward_ms": cand_fwd,
        "backward_ms": cand_bwd,
        "baseline_forward_ms": ref_fwd,
        "baseline_backward_ms": ref_bwd,
        "forward_max_error": forward_err,
        "grad_max_error": grad_err,
        "speedup_forward": ref_fwd / max(cand_fwd, 1e-6),
        "speedup_backward": ref_bwd / max(cand_bwd, 1e-6),
        "flops_estimate": flops,
        "throughput_gflops": thrpt_gflops,
        "score": score,
    }


def eval_rope(
    kernels: Dict[str, Callable[..., torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, float]:
    fn = kernels["apply_rope_triton"]
    x_base = inputs["rope_x"]
    cos = inputs["cos"]
    sin = inputs["sin"]

    # Correctness
    ref_x = _clone_requires_grad(x_base)
    ref_out = reference_rope(ref_x, cos, sin)
    ref_out.sum().backward()
    ref_grad = ref_x.grad.detach()

    test_x = _clone_requires_grad(x_base)
    test_out = fn(test_x, cos, sin)
    test_out.sum().backward()
    test_grad = test_x.grad.detach()

    forward_err = float((test_out - ref_out).abs().max().item())
    grad_err = float((test_grad - ref_grad).abs().max().item())

    ref_args = lambda: (_clone_requires_grad(x_base), cos, sin)
    cand_args = lambda: (_clone_requires_grad(x_base), cos, sin)
    ref_fwd, ref_bwd = measure_latency(
        reference_rope, ref_args, device, warmup_iters, bench_iters, backward=True
    )
    cand_fwd, cand_bwd = measure_latency(
        fn, cand_args, device, warmup_iters, bench_iters, backward=True
    )

    flops = estimate_rope_flops(tuple(x_base.shape))
    thrpt_gflops = flops / max(cand_fwd + cand_bwd, 1e-3) / 1e6

    score = kernel_score(
        forward_err, grad_err, ref_fwd, ref_bwd, cand_fwd, cand_bwd
    )
    return {
        "forward_ms": cand_fwd,
        "backward_ms": cand_bwd,
        "baseline_forward_ms": ref_fwd,
        "baseline_backward_ms": ref_bwd,
        "forward_max_error": forward_err,
        "grad_max_error": grad_err,
        "speedup_forward": ref_fwd / max(cand_fwd, 1e-6),
        "speedup_backward": ref_bwd / max(cand_bwd, 1e-6),
        "flops_estimate": flops,
        "throughput_gflops": thrpt_gflops,
        "score": score,
    }


def eval_relusqr(
    kernels: Dict[str, Callable[..., torch.Tensor]],
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, float]:
    fn = kernels["fast_relusqr"]
    x_base = inputs["relu_x"]

    ref_x = _clone_requires_grad(x_base)
    ref_out = reference_relusqr(ref_x)
    ref_out.sum().backward()
    ref_grad = ref_x.grad.detach()

    test_x = _clone_requires_grad(x_base)
    test_out = fn(test_x)
    test_out.sum().backward()
    test_grad = test_x.grad.detach()

    forward_err = float((test_out - ref_out).abs().max().item())
    grad_err = float((test_grad - ref_grad).abs().max().item())

    ref_args = lambda: (_clone_requires_grad(x_base),)
    cand_args = lambda: (_clone_requires_grad(x_base),)
    ref_fwd, ref_bwd = measure_latency(
        reference_relusqr, ref_args, device, warmup_iters, bench_iters, backward=True
    )
    cand_fwd, cand_bwd = measure_latency(
        fn, cand_args, device, warmup_iters, bench_iters, backward=True
    )

    flops = estimate_relusqr_flops(tuple(x_base.shape))
    thrpt_gflops = flops / max(cand_fwd + cand_bwd, 1e-3) / 1e6

    score = kernel_score(
        forward_err, grad_err, ref_fwd, ref_bwd, cand_fwd, cand_bwd
    )
    return {
        "forward_ms": cand_fwd,
        "backward_ms": cand_bwd,
        "baseline_forward_ms": ref_fwd,
        "baseline_backward_ms": ref_bwd,
        "forward_max_error": forward_err,
        "grad_max_error": grad_err,
        "speedup_forward": ref_fwd / max(cand_fwd, 1e-6),
        "speedup_backward": ref_bwd / max(cand_bwd, 1e-6),
        "flops_estimate": flops,
        "throughput_gflops": thrpt_gflops,
        "score": score,
    }


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------

def evaluate_kernel_suite(
    kernel_suite: Dict[str, Callable[..., torch.Tensor]],
    device_arg: str,
    dtype_arg: str,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, Any]:
    device = select_device(device_arg)
    dtype = resolve_dtype(dtype_arg, device)

    inputs = make_llm_like_inputs(device, dtype)

    per_kernel = {
        "rmsnorm": eval_rmsnorm(kernel_suite, inputs, device, warmup_iters, bench_iters),
        "rope": eval_rope(kernel_suite, inputs, device, warmup_iters, bench_iters),
        "relusqr": eval_relusqr(kernel_suite, inputs, device, warmup_iters, bench_iters),
    }

    combined_score_run = float(np.mean([k["score"] for k in per_kernel.values()]))
    avg_forward = float(np.mean([k["forward_ms"] for k in per_kernel.values()]))
    avg_backward = float(np.mean([k["backward_ms"] for k in per_kernel.values()]))
    max_error = float(
        max(
            k["forward_max_error"] + k["grad_max_error"]
            for k in per_kernel.values()
        )
    )

    return {
        "combined_score_run": combined_score_run,
        "per_kernel": per_kernel,
        "device": str(device),
        "dtype": str(dtype),
        "avg_forward_ms": avg_forward,
        "avg_backward_ms": avg_backward,
        "max_error": max_error,
    }


def aggregate_kernel_metrics(
    results: List[Dict[str, Any]],
    device_arg: str,
    dtype_arg: str,
    warmup_iters: int,
    bench_iters: int,
) -> Dict[str, Any]:
    """Aggregate metrics across runs (usually a single run)."""
    if not results:
        return {
            "combined_score": 0.0,
            "public": {"error": "No results to aggregate."},
            "private": {},
        }

    run_metrics: List[Dict[str, Any]] = []
    errors: List[str] = []
    for suite in results:
        try:
            run_metrics.append(
                evaluate_kernel_suite(
                    suite, device_arg, dtype_arg, warmup_iters, bench_iters
                )
            )
        except Exception as e:
            errors.append(str(e))
            run_metrics.append(
                {
                    "combined_score_run": 0.0,
                    "per_kernel": {},
                    "device": device_arg,
                    "dtype": dtype_arg,
                    "avg_forward_ms": 0.0,
                    "avg_backward_ms": 0.0,
                    "max_error": 0.0,
                    "skipped_reason": f"evaluation error: {e}",
                }
            )

    combined_score = float(np.mean([rm["combined_score_run"] for rm in run_metrics]))
    avg_forward = float(np.mean([rm["avg_forward_ms"] for rm in run_metrics]))
    avg_backward = float(np.mean([rm["avg_backward_ms"] for rm in run_metrics]))
    max_error = float(max(rm["max_error"] for rm in run_metrics))

    public_metrics = {
        "avg_forward_ms": avg_forward,
        "avg_backward_ms": avg_backward,
        "max_error": max_error,
        "device": run_metrics[0]["device"],
        "dtype": run_metrics[0]["dtype"],
    }

    private_metrics = {
        "runs": run_metrics,
    }
    if errors:
        private_metrics["errors"] = errors

    metrics = {
        "combined_score": combined_score,
        "public": public_metrics,
        "private": private_metrics,
    }
    return metrics


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

def main(
    program_path: str,
    results_dir: str,
    device: str,
    dtype: str,
    num_experiment_runs: int,
    warmup_iters: int,
    bench_iters: int,
    fail_on_missing_deps: bool,
):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    print(f"Device: {device} (auto-resolves to CUDA when available)")
    print(f"Dtype: {dtype}")
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print(f"Experiment runs: {num_experiment_runs}")

    # Pre-flight: require Triton and CUDA; write clear results if missing.
    triton_spec = importlib.util.find_spec("triton")
    if triton_spec is None or not torch.cuda.is_available():
        missing = []
        if triton_spec is None:
            missing.append("triton")
        if not torch.cuda.is_available():
            missing.append("cuda")
        error = (
            f"Evaluation skipped: missing dependencies ({', '.join(missing)}). "
            "Install Triton and ensure a CUDA device for kernel benchmarking."
        )
        print(error)
        metrics = {
            "combined_score": 0.0,
            "public": {"error": error},
            "private": {},
        }
        save_json_results(results_dir, metrics, correct=not fail_on_missing_deps, error=error)
        return

    def _aggregate_with_config(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return aggregate_kernel_metrics(
            results, device, dtype, warmup_iters, bench_iters
        )

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=num_experiment_runs,
        validate_fn=validate_kernel_suite,
        aggregate_metrics_fn=_aggregate_with_config,
        default_metrics_on_error={
            "combined_score": 0.0,
            "public": {},
            "private": {},
        },
    )

    status = "succeeded" if correct else "failed"
    print(f"Evaluation {status}.")
    if error_msg:
        print(f"Error: {error_msg}")

    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Triton kernels using synthetic LLM-like workloads."
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.py",
        help="Path to the program exposing run_experiment.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_kernel_design",
        help="Directory to save metrics.json and correct.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda only; evaluation requires CUDA/Triton).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Tensor dtype to use (auto|float16|bfloat16|float32).",
    )
    parser.add_argument(
        "--num_experiment_runs",
        type=int,
        default=1,
        help="How many times to call run_experiment before aggregation.",
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=2,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--bench_iters",
        type=int,
        default=4,
        help="Timed iterations for latency measurement.",
    )
    parser.add_argument(
        "--fail_on_missing_deps",
        action="store_true",
        help="If set, mark evaluation incorrect when Triton/CUDA are missing. "
        "By default, missing deps produce score 0 but are marked correct to keep evolution running.",
    )
    args = parser.parse_args()
    main(
        args.program_path,
        args.results_dir,
        args.device,
        args.dtype,
        args.num_experiment_runs,
        args.warmup_iters,
        args.bench_iters,
        args.fail_on_missing_deps,
    )
