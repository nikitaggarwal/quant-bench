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
- [x] Web frontend — a leaderboard + per-model comparison pages (`app.py`)
- [x] On-demand requests: visitors request a model, it's queued, and a GPU worker
      benchmarks it and emails them a link (`bench_requests` queue + `worker.py`)
- [ ] On-demand GPU (Modal) + SMS notifications (Phase 2)
- [ ] API layer

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

## Live benchmarking (on-demand requests)

A visitor can request a model that isn't in the database yet. The web app can't run
benchmarks itself (it's a serverless function with no GPU), so the request and the
work are decoupled by a job queue in Postgres. See `docs/live-benchmarking-design.md`
for the full design.

> **What "on-demand" means today (Phase 1).** The *request* side is automatic: anyone
> can submit a model and it's instantly queued. The *GPU* side is **not** — nothing
> boots a GPU by itself. A benchmark only runs while **you have manually started a GPU
> box and left `worker.py` running** on it. Requests submitted while that worker is off
> just wait in the queue until the next time you run it. Automatic "a request arrives →
> a GPU spins up → it shuts down when idle" is **Phase 2** (see below), and is not built
> yet.

The flow:

1. Visitor submits a Hugging Face model id + email on the leaderboard page.
2. `POST /request` (in `app.py`) checks whether we already have the results (if so it
   just links there), then validates the model exists and is small enough
   (`intake.py`), rate-limits, and drops a row on the `bench_requests` queue.
3. `worker.py` — which **you** have started on a GPU box — claims the job, benchmarks
   the model against its baseline, saves the results, and emails the visitor a link
   (`notify.py`).
4. The visitor's status page (`/request/<token>`) flips to "done" with a link.

Run the worker yourself on the GPU box (after installing `requirements-bench.txt`):

```bash
python3 worker.py     # polls the queue; Ctrl-C to stop
```

Requests submitted while the worker is off simply wait in the queue until you run it.

### Making the GPU truly on-demand (Phase 2, not built)

To get "a request boots a GPU automatically," the worker's job body moves into a
serverless-GPU platform (**Modal** is the design doc's recommendation) that provisions
a GPU per job and shuts it down when idle — so you pay only while a benchmark actually
runs, instead of paying by the hour for a box that's mostly idle. The job-claiming and
result-saving logic in `worker.py` stays the same; only *where the GPU comes from*
changes. Until then, the box is manual and billed for every hour it's on.

### Environment variables

For local runs, put these in the git-ignored `.env` file (alongside `DATABASE_URL`);
`storage.load_env()` loads them automatically at startup. In production (Vercel / the
GPU box), set them as real environment variables — those always take precedence over
`.env`. Never commit secrets.

| Variable | Needed by | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | web + worker | Neon Postgres connection string (already required) |
| `FLASK_SECRET_KEY` | web | signs flash messages; set a random value in production |
| `RESEND_API_KEY` | worker | Resend key for sending result emails (unset = emails are skipped/logged) |
| `RESEND_FROM` | worker | sender address (default: Resend's `onboarding@resend.dev` test sender) |
| `PUBLIC_BASE_URL` | worker | site root for building links, e.g. `https://quant-bench.vercel.app` |
| `HF_TOKEN` | web + worker | optional; only needed to validate/run gated models |
| `BENCH_LIMIT` | worker | optional; cap eval examples per task for cheaper test runs (unset = full) |

### Email setup (Resend)

Email uses [Resend](https://resend.com). To start, just set `RESEND_API_KEY` — sending
falls back to Resend's shared test sender `onboarding@resend.dev`, which needs no domain
setup but **can only deliver to the email you signed up to Resend with** (fine for
testing, and messages may land in Spam/Promotions). To email real visitors, verify your
own domain in the Resend dashboard (add the DNS records it shows you), then set
`RESEND_FROM` to an address at that domain, e.g. `quant-bench <bench@yourdomain.com>`.

## Stack

- Python, `transformers`, `lm-evaluation-harness`, `bitsandbytes`
- Postgres (Neon) for storage, via `psycopg2`
- Tested locally on Apple Silicon (MPS); quantized models run on rented GPU compute
