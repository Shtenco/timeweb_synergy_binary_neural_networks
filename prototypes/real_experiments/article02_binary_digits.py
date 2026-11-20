"""Train FP32 and binarized MLPs on UCI handwritten digits; time a Python popcount backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _ste_sign(x: torch.Tensor) -> torch.Tensor:
    quantized = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    return quantized.detach() + x - x.detach()


class FloatMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class BinaryHiddenMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ste_sign(x)
        w1 = _ste_sign(self.fc1.weight)
        x = _ste_sign(F.linear(x, w1, self.fc1.bias))
        w2 = _ste_sign(self.fc2.weight)
        x = _ste_sign(F.linear(x, w2, self.fc2.bias))
        return self.out(x)


def _pack_positive(values: np.ndarray) -> int:
    packed = 0
    for index, value in enumerate(values):
        if value >= 0:
            packed |= 1 << index
    return packed


def _packed_binary_layer(inputs: np.ndarray, packed_weights: list[int], bias: np.ndarray) -> np.ndarray:
    batch, width = inputs.shape
    mask = (1 << width) - 1
    packed_inputs = [_pack_positive(row) for row in inputs]
    output = np.empty((batch, len(packed_weights)), dtype=np.float32)
    for row_index, input_bits in enumerate(packed_inputs):
        for output_index, weight_bits in enumerate(packed_weights):
            matches = (~(input_bits ^ weight_bits) & mask).bit_count()
            dot = 2 * matches - width
            output[row_index, output_index] = 1.0 if dot + float(bias[output_index]) >= 0 else -1.0
    return output


def pack_binary_weights(model: BinaryHiddenMLP) -> tuple[list[int], np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray]:
    state = {key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    w1 = np.where(state["fc1.weight"] >= 0, 1.0, -1.0)
    w2 = np.where(state["fc2.weight"] >= 0, 1.0, -1.0)
    packed_w1 = [_pack_positive(row) for row in w1]
    packed_w2 = [_pack_positive(row) for row in w2]
    return packed_w1, state["fc1.bias"], packed_w2, state["fc2.bias"], state["out.weight"], state["out.bias"]


def packed_logits(
    packed_weights: tuple[list[int], np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray],
    inputs: np.ndarray,
) -> np.ndarray:
    packed_w1, bias1, packed_w2, bias2, output_weight, output_bias = packed_weights
    x = np.where(inputs >= 0, 1.0, -1.0).astype(np.float32, copy=False)
    h1 = _packed_binary_layer(x, packed_w1, bias1)
    h2 = _packed_binary_layer(h1, packed_w2, bias2)
    return h2 @ output_weight.T + output_bias


def _train(model: nn.Module, X: torch.Tensor, y: torch.Tensor, epochs: int, seed: int) -> float:
    torch.manual_seed(seed + 991)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        logits = model(X)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return time.perf_counter() - started


def _timed_forward(function, repeats: int) -> float:
    function()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - started) / repeats


def _inference_logits(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        return model(inputs)


def run_real_experiment(
    seeds: tuple[int, ...] = (7, 17, 27),
    epochs: int = 50,
    timing_repeats: int = 10,
) -> dict:
    if not seeds or epochs < 1 or timing_repeats < 1:
        raise ValueError("Provide seeds and positive epochs/timing_repeats")
    torch.set_num_threads(1)
    dataset = load_digits()
    raw_X = np.asarray(dataset.data, dtype=np.float32)
    y = np.asarray(dataset.target, dtype=np.int64)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.25, stratify=y, random_state=4107
    )
    X_float = torch.tensor(raw_X / 16.0, dtype=torch.float32)
    X_binary = torch.tensor(raw_X / 8.0 - 1.0, dtype=torch.float32)
    y_torch = torch.tensor(y, dtype=torch.long)
    X_test_float = X_float[test_idx]
    X_test_binary = X_binary[test_idx]
    runs = []
    for seed in seeds:
        torch.manual_seed(seed)
        float_model = FloatMLP()
        torch.manual_seed(seed)
        binary_model = BinaryHiddenMLP()
        float_train_seconds = _train(float_model, X_float[train_idx], y_torch[train_idx], epochs, seed)
        binary_train_seconds = _train(binary_model, X_binary[train_idx], y_torch[train_idx], epochs, seed)

        float_model.eval()
        binary_model.eval()
        with torch.no_grad():
            float_logits = float_model(X_test_float).numpy()
            binary_torch_logits = binary_model(X_test_binary).numpy()
        packing_started = time.perf_counter()
        packed_weights = pack_binary_weights(binary_model)
        packing_seconds = time.perf_counter() - packing_started
        packed = packed_logits(packed_weights, X_binary[test_idx].numpy())
        packed_error = float(np.max(np.abs(binary_torch_logits - packed)))
        binary_torch_accuracy = accuracy_score(y[test_idx], binary_torch_logits.argmax(axis=1))
        packed_accuracy = accuracy_score(y[test_idx], packed.argmax(axis=1))
        float_accuracy = accuracy_score(y[test_idx], float_logits.argmax(axis=1))

        float_seconds_per_test_batch = _timed_forward(
            lambda: _inference_logits(float_model, X_test_float), timing_repeats
        )
        binary_torch_seconds_per_test_batch = _timed_forward(
            lambda: _inference_logits(binary_model, X_test_binary), timing_repeats
        )
        packed_seconds_per_test_batch = _timed_forward(
            lambda: packed_logits(packed_weights, X_binary[test_idx].numpy()), timing_repeats
        )

        float_bytes = sum(parameter.numel() * 4 for parameter in float_model.parameters())
        binary_hidden_bytes = 128 * ((64 + 7) // 8) + 64 * ((128 + 7) // 8)
        binary_total_bytes = (
            binary_hidden_bytes
            + (128 + 64 + 10) * 4
            + (64 * 10) * 4
        )
        runs.append(
            {
                "seed": int(seed),
                "epochs": int(epochs),
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "float32_test_accuracy": float(float_accuracy),
                "binary_torch_test_accuracy": float(binary_torch_accuracy),
                "binary_packed_test_accuracy": float(packed_accuracy),
                "packed_forward_matches_binary_torch": packed_error < 1e-5,
                "packed_max_abs_logit_error": packed_error,
                "float_train_seconds": float_train_seconds,
                "binary_train_seconds": float(binary_train_seconds),
                "binary_weight_pack_seconds": float(packing_seconds),
                "float32_test_batch_seconds": float(float_seconds_per_test_batch),
                "binary_torch_test_batch_seconds": float(binary_torch_seconds_per_test_batch),
                "binary_packed_test_batch_seconds": float(packed_seconds_per_test_batch),
                "fp32_weight_bytes": int(float_bytes),
                "binary_hidden_packed_bytes": int(binary_hidden_bytes),
                "binary_total_weight_bytes": int(binary_total_bytes),
            }
        )

    return {
        "article_id": "02",
        "experiment": "FP32 versus STE-binarized hidden layers on UCI handwritten digits",
        "evidence_type": "measured_public_dataset",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "Optical Recognition of Handwritten Digits (scikit-learn copy)",
            "source_id": "UCI-80-digits",
            "source_url": "https://archive.ics.uci.edu/dataset/80/optical-recognition-of-handwritten-digits",
            "rows": int(len(y)),
            "features": int(raw_X.shape[1]),
            "classes": 10,
            "feature_sha256": _sha256(raw_X),
            "target_sha256": _sha256(y),
            "license": "UCI dataset record; verify attribution/terms for redistribution",
        },
        "method": {
            "split": "stratified 75/25 split, random_state=4107, shared across models",
            "float_model": "64-128-64-10 ReLU MLP, FP32 weights and activations",
            "binary_model": "64-128-64 binary hidden layers trained with straight-through sign; FP32 output layer",
            "binary_inference": "Python integer XNOR/popcount for two hidden layers; FP32 final layer",
            "input_binarization": "threshold normalized pixel intensities at 8 on the original 0..16 scale",
            "seeds": list(seeds),
            "epochs": epochs,
            "timing_repeats": timing_repeats,
            "no_test_tuning": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "logical_cpus": os.cpu_count(),
            "physical_memory_bytes": psutil.virtual_memory().total,
            "torch": version("torch"),
            "numpy": version("numpy"),
            "scikit_learn": version("scikit-learn"),
            "torch_threads": torch.get_num_threads(),
        },
        "runs": runs,
        "limitations": [
            "The sklearn digits set is a small 8x8 benchmark, not a modern vision workload.",
            "The binary network is not fully integer: biases and final layer stay FP32; training uses FP32 straight-through gradients.",
            "Python big-integer popcount is a correctness backend, not optimized SIMD/native code; its measured latency is implementation-specific.",
            "Three seeds and one fixed split are an exploratory comparison, not a broad accuracy estimate.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("evidence/02_real"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    args = parser.parse_args(argv)
    result = run_real_experiment(seeds=tuple(args.seeds), epochs=args.epochs, timing_repeats=10)
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = args.out / f"article-02-digits-binary-{stamp}.json"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
    print(path)
    for field in (
        "float32_test_accuracy",
        "binary_torch_test_accuracy",
        "binary_packed_test_accuracy",
        "float32_test_batch_seconds",
        "binary_packed_test_batch_seconds",
        "fp32_weight_bytes",
        "binary_total_weight_bytes",
    ):
        print(field, [round(row[field], 6) if isinstance(row[field], float) else row[field] for row in result["runs"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
