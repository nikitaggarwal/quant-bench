# quant-bench

A tool for benchmarking quantized Hugging Face models against their full-precision baselines.

## Problem

There's no simple open-source tool that takes a Hugging Face model, runs it through a
standard benchmark suite, and shows exactly how much accuracy was lost by quantizing it.
Claims are scattered across model cards, if they exist at all.

## Goal

Paste in a quantized model's Hugging Face ID → the tool finds its full-precision baseline →
runs both through the same benchmarks → shows a side-by-side comparison of accuracy, along
with runtime and memory use.

- **MMLU** — knowledge/reasoning
- **HellaSwag** — commonsense reasoning
- **WikiText-2 perplexity** — cheapest signal, no generation needed

For models too large to run locally (e.g. hundreds of billions of parameters), the tool
accepts manually-entered numbers from the organization's own published comparison — stored
in the exact same schema as a locally-computed result, so the comparison view doesn't care
which path the number came from.

## Status

Early development. Currently:
- [x] Load and run inference on a model given its Hugging Face ID
- [x] Run a real benchmark (HellaSwag) via `lm-evaluation-harness`
- [x] Structured result schema (computed + manual entry, same shape)
- [x] Precision control (fp32 / fp16 / int8 / int4) with runtime + peak-memory tracking
- [x] Auto-detect the full-precision baseline from a quantized model's metadata
- [x] Persistent storage in Postgres (Neon), with lookup before recompute
- [ ] Background job queue for long-running benchmark runs
- [ ] API layer
- [ ] Web frontend

## How quantization runs

Precision is selected per run:

- `fp32` / `fp16` — full precision, runs anywhere (including Apple Silicon / CPU).
- `int8` / `int4` — quantized via `bitsandbytes`, which requires a CUDA GPU. Running these
  locally raises a clear error; run them on rented GPU compute instead.

Each run records its accuracy metrics plus runtime and peak memory, so a comparison shows
not just the accuracy lost by quantizing, but the speed and footprint gained.

## Example

```
zai-org/GLM-5.2          (BF16, original, full precision)
vs.
nvidia/GLM-5.2-NVFP4     (4-bit, quantized)
```

## Setup

Results are stored in Postgres (Neon). Put your connection string in a `.env` file (it holds
a password, so keep it out of version control — it's already in `.gitignore`):

```
DATABASE_URL=postgresql://USER:PASSWORD@HOST/DBNAME?sslmode=require
```

Local (Apple Silicon / CPU — fp16/fp32 only):

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate lm-eval huggingface_hub psycopg2-binary
hf auth login          # for gated models
python3 storage.py     # create the tables in Postgres
```

GPU instance (adds int8/int4 support via bitsandbytes):

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 benchmark.py
```

Runs a benchmark and saves the results to Postgres. The two entry points:

- `run_benchmark(model_id, task, precision=...)` — evaluate one model at one precision.
- `run_comparison(quantized_model_id, task)` — detect the baseline, then benchmark both the
  quantized model and its full-precision baseline, skipping anything already stored.

## Stack

- Python, `transformers`, `lm-evaluation-harness`, `bitsandbytes`
- Postgres (Neon) for storage, via `psycopg2`
- Tested locally on Apple Silicon (MPS); quantized models run on rented GPU compute
