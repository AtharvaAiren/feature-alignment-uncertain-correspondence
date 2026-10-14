"""A small labeled replay comparator that explicitly breaks the frozen-consumer constraint."""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def online_er_head(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weight: np.ndarray,
    initial_bias: np.ndarray,
    *,
    rng: np.random.Generator,
    learning_rate: float = 1e-2,
    buffer_size: int = 256,
    replay_batch_size: int = 32,
    weight_decay: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """One-pass online softmax head update with uniform reservoir replay."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    weight = np.asarray(initial_weight, dtype=np.float64).copy()
    bias = np.asarray(initial_bias, dtype=np.float64).copy()
    if x.ndim != 2 or y.shape != (len(x),) or weight.shape[1] != x.shape[1]:
        raise ValueError("invalid ER shapes")
    reservoir_x = np.empty((buffer_size, x.shape[1]), dtype=np.float64)
    reservoir_y = np.empty(buffer_size, dtype=np.int64)
    stored = 0
    seen = 0
    for index in range(len(x)):
        current_x = x[index:index + 1]
        current_y = y[index:index + 1]
        if stored:
            draw = min(replay_batch_size, stored)
            selected = rng.choice(stored, size=draw, replace=False)
            batch_x = np.concatenate([current_x, reservoir_x[selected]], axis=0)
            batch_y = np.concatenate([current_y, reservoir_y[selected]], axis=0)
        else:
            batch_x, batch_y = current_x, current_y
        probabilities = _softmax(batch_x @ weight.T + bias)
        probabilities[np.arange(len(batch_y)), batch_y] -= 1.0
        grad_w = probabilities.T @ batch_x / len(batch_y) + weight_decay * weight
        grad_b = probabilities.mean(axis=0)
        weight -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        seen += 1
        if stored < buffer_size:
            slot = stored
            stored += 1
        else:
            candidate = int(rng.integers(0, seen))
            slot = candidate if candidate < buffer_size else -1
        if slot >= 0:
            reservoir_x[slot] = current_x[0]
            reservoir_y[slot] = current_y[0]
    return weight, bias, {
        "buffer_size": int(buffer_size),
        "stored_examples": int(stored),
        "stream_examples": int(seen),
        "replay_batch_size": int(replay_batch_size),
        "learning_rate": float(learning_rate),
    }
