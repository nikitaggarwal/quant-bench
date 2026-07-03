# quant-bench

A tool for benchmarking quantized Hugging Face models against their full-precision baselines.

## Problem

There's no simple open-source tool that takes a Hugging Face model, runs it through a
standard benchmark suite, and shows exactly how much accuracy was lost by quantizing it.
Claims are scattered across model cards, if they exist at all.

## Goal

Paste in a quantized model's Hugging Face ID → the tool finds its full-precision baseline →
runs both through the same benchmarks → shows a side-by-side comparison.

- **MMLU** — knowledge/reasoning
- **HellaSwag** — commonsense reasoning
- **WikiText-2 perplexity** — cheapest signal, no generation needed

For models too large to run locally (e.g. hundreds of billions of parameters), the tool
accepts manually-entered numbers from the organization's own published comparison — stored
in the exact same schema as a locally-computed result, so the comparison view doesn't care
which path the number came from.

## Status

🚧 Early development. Currently:
- [x] Load and run inference on a model given its Hugging Face ID
- [x] Run a real benchmark (HellaSwag) via `lm-evaluation-harness`
- [x] Structured `BenchmarkResult` schema (computed + manual entry, same shape)
- [ ] Persistent storage (SQLite) + lookup for existing comparisons
- [ ] Auto-detect baseline model from a quantized model's metadata
- [ ] Background job queue for long-running benchmark runs
- [ ] API layer
- [ ] Web frontend

## Example
zai-org/GLM-5.2          (BF16, original, full precision)
vs.
nvidia/GLM-5.2-NVFP4      (4-bit, quantized)
## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate lm-eval huggingface_hub
hf auth login
```

## Usage

```bash
python3 benchmark.py
```

Runs a benchmark against a model and saves structured results to `results.json`.

## Stack

- Python, `transformers`, `lm-evaluation-harness`
- Tested locally on Apple Silicon (MPS); larger models run on rented GPU compute
EOF