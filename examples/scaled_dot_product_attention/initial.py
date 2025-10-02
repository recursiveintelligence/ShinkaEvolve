# EVOLVE-BLOCK-START
"""Baseline scaled dot-product attention experiment"""

from typing import Tuple

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_shifted = np.exp(shifted)
    denom = np.sum(exp_shifted, axis=axis, keepdims=True)
    return exp_shifted / denom


def naive_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    *,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes attention outputs without the canonical sqrt(d) scaling."""
    logits = queries @ keys.T
    scaled_logits = logits / max(temperature, 1e-8)
    weights = _softmax(scaled_logits, axis=-1)
    outputs = weights @ values
    reported_score = float(np.mean(outputs))
    return outputs, weights, reported_score


def run_attention(
    *,
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    temperature: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Entry point for the optimization problem."""
    return naive_attention(
        queries=queries,
        keys=keys,
        values=values,
        temperature=temperature,
    )


# EVOLVE-BLOCK-END
