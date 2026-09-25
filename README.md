# laya-mlx-demo

Demo project for **[laya-mlx](https://huggingface.co/aac6fef/laya-mlx)**, a native MLX FP16 conversion of [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) optimized for Apple silicon.

This checkpoint uses **ModernBERT-large**, a **512-token total context**, and Laya's decision Transformer, scoring head, and action head. It supports `choice`, ordinal `score`, and boolean `noul` questions. All model computation runs in MLX; the runtime does not require PyTorch or Transformers.

## Installation

Run on an Apple silicon Mac with macOS 14+ and Python 3.11+:

```bash
uv sync
```

Or install with pip:

```bash
pip install laya-mlx
```

## Quick Start

You can run the demo directly with:

```bash
uv run laya-mlx-demo
```

### Python Example

```python
import laya_mlx as laya

agent = laya.load("aac6fef/laya-mlx")
result = agent.predict(
    "I was billed twice. Please refund the duplicate today.",
    {
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this request?",
            "criteria": ["billing", "technical", "sales"],
        },
        "refund": {
            "type": "noul",
            "instructions": "Does the customer ask for money back?",
        },
    },
)
print(result["answers"])
```

> **Tip:** Use `dtype="float32"` for closer agreement with upstream FP32 arithmetic. The source weights themselves are FP16. Question formatting, tokenizer behavior, calibration temperatures, and output schema are preserved.

## Validation & Benchmarks

Tested locally on Apple M3 Max, 40-core GPU, 128 GB unified memory, macOS 27.2, Python 3.12.13 and MLX 0.32.2.

- FP16 agrees with upstream PyTorch MPS FP32 on the argmax of **63/63** decision distributions across 16 cases.
- Maximum calibrated probability difference: **0.0054443**.
- **100 repeated calls** produced finite, deterministic public outputs; measured MLX active-memory growth after clearing caches was **0 bytes**.
- Every exported tensor was checked for exact equality with the corresponding source tensor cast to FP16.

For full performance details and raw timing samples, refer to the [performance report & benchmarks](https://github.com/mizorewww/laya-mlx/blob/main/BENCHMARKS.md).

## Provenance and Limits

- **Source checkpoint:** `convaiinnovations/laya` at `c5d78730f3493e4fe16d61507ef4b78eef7318cf`.
- **Upstream code:** [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), commit `6a5819129eb220570792e417e49723d697efd76f`.
- **Conversion:** Changes parameter names for MLX and preserves FP16 weights without retraining or quantizing to fewer bits.
- **Port scope:** Independent port. Model quality, calibration, and language/task limitations remain those of the original checkpoint. Questions and options share the context budget with the input state.
- **Language support:** The typed-decisions checkpoint is specialized for upstream workflows; the multilingual checkpoint is the intended choice for non-English text.

## License

Apache-2.0. Original Laya models and code are by Convai Innovations and contributors.
