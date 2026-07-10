from dataclasses import asdict
import json
import time

import torch
from transformers import BitsAndBytesConfig
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# RunResult / BenchmarkResult live in results.py so the storage layer and web UI
# can import them without pulling in this module's torch/transformers/lm_eval.
from results import BenchmarkResult, RunResult


# Precisions that require a CUDA GPU (bitsandbytes has no CPU/MPS kernel).
_CUDA_ONLY_PRECISIONS = {"int4", "int8"}
_VALID_PRECISIONS = {"fp32", "fp16", "int8", "int4"}


def _check_precision(precision: str):
    """Validate a precision label and that the current hardware can run it."""
    if precision not in _VALID_PRECISIONS:
        raise ValueError(
            f"Unknown precision '{precision}'. Expected one of {sorted(_VALID_PRECISIONS)}."
        )

    if precision in _CUDA_ONLY_PRECISIONS and not torch.cuda.is_available():
        raise RuntimeError(
            f"precision='{precision}' needs a CUDA GPU (bitsandbytes has no CPU/MPS "
            f"kernel). Run this on Colab or a rented GPU, or use fp16/fp32 locally."
        )


def _bnb_config(precision: str) -> BitsAndBytesConfig:
    """bitsandbytes quantization config for a quantized precision label."""
    if precision == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    # int4 — nf4, matching the Colab run
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )


def _build_lm(model_id: str, precision: str, device: str) -> HFLM:
    """
    Build the lm-eval model wrapper for a (model, precision).

    fp16/fp32 load straight through HFLM. For int4/int8 we build the quantized
    model with transformers ourselves and hand the ready model to HFLM: HFLM
    already passes its own quantization_config into the loader, so passing a
    second one via kwargs collides ("multiple values for quantization_config").
    """
    _check_precision(precision)

    if precision in ("fp32", "fp16"):
        dtype = "float32" if precision == "fp32" else "float16"
        return HFLM(pretrained=model_id, device=device, dtype=dtype)

    # Quantized: load and quantize the model ourselves, then wrap it.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=_bnb_config(precision),
        dtype=torch.float16,
        device_map={"": 0},  # place the whole model on GPU 0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return HFLM(pretrained=model, tokenizer=tokenizer, device=device)


def _reset_peak_memory(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb(device: str) -> float | None:
    """Peak allocated memory in MB, or None if the backend can't report it."""
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    if device == "mps" and torch.backends.mps.is_available():
        # MPS has no peak counter; current allocation is the best proxy available.
        return torch.mps.current_allocated_memory() / (1024 * 1024)
    return None


def run_benchmark(
    model_id: str,
    task: str,
    precision: str = "fp32",
    limit: int | None = 20,
    device: str = "mps",
    n_shot: int = 0,
) -> RunResult:
    # bitsandbytes precisions force CUDA; ignore whatever device was requested.
    if precision in _CUDA_ONLY_PRECISIONS:
        device = "cuda"

    lm = _build_lm(model_id, precision, device)

    _reset_peak_memory(device)
    start = time.perf_counter()
    results = simple_evaluate(model=lm, tasks=[task], limit=limit, num_fewshot=n_shot)
    runtime_seconds = time.perf_counter() - start
    peak_memory_mb = _peak_memory_mb(device)

    task_results = results["results"][task]
    skip_keys = {"alias", "sample_len", "sample_len_stderr"}

    metrics = []
    for key, value in task_results.items():
        if key in skip_keys or not isinstance(value, (int, float)):
            continue

        metric_name = key.split(",")[0]  # strip the ",none" filter suffix first

        if metric_name.endswith("_stderr"):
            continue  # skip for now — revisit once we decide how to display error bars

        metrics.append(BenchmarkResult(
            model_id=model_id,
            benchmark=task,
            metric_name=metric_name,
            value=value,
            precision=precision,
            source="computed",
            n_shot=n_shot,
            limit=limit,
        ))

    return RunResult(
        model_id=model_id,
        benchmark=task,
        precision=precision,
        device=device,
        runtime_seconds=runtime_seconds,
        peak_memory_mb=peak_memory_mb,
        n_shot=n_shot,
        limit=limit,
        source="computed",
        metrics=metrics,
    )


from detect_baseline import detect_baseline


def run_comparison(
    quantized_model_id: str,
    task: str,
    precision: str = "int4",
    baseline_precision: str = "fp16",
    limit: int | None = 20,
    device: str = "mps",
):
    """
    Given a quantized model ID, detect its baseline and run the benchmark on both.
    Returns (quantized_run, baseline_run, baseline_model_id).
    """
    from storage import has_results, save_run

    baseline_id = detect_baseline(quantized_model_id)
    if baseline_id is None:
        print(f"Could not auto-detect a baseline for {quantized_model_id}. "
              f"You'll need to provide one manually.")
        return None, None, None

    print(f"Detected baseline: {baseline_id}")

    # Quantized model
    if has_results(quantized_model_id, task):
        print(f"Already have results for {quantized_model_id} on {task}, skipping compute.")
        quantized_run = None
    else:
        quantized_run = run_benchmark(
            quantized_model_id, task, precision=precision, limit=limit, device=device
        )
        save_run(quantized_run)

    # Baseline model (full precision)
    if has_results(baseline_id, task):
        print(f"Already have results for {baseline_id} on {task}, skipping compute.")
        baseline_run = None
    else:
        baseline_run = run_benchmark(
            baseline_id, task, precision=baseline_precision, limit=limit, device=device
        )
        save_run(baseline_run)

    return quantized_run, baseline_run, baseline_id


def save_results_json(run: RunResult, path: str = "results.json"):
    """Append a run (with its metrics) to a JSON file. Debug/inspection helper."""
    try:
        with open(path, "r") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []

    existing.append(asdict(run))

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved run {run.model_id} [{run.precision}] to {path}")


if __name__ == "__main__":
    from storage import init_db, save_run
    init_db()

    # fp16 baseline locally; int4 requires CUDA and will raise a clear error here.
    run = run_benchmark("Qwen/Qwen2.5-0.5B-Instruct", "hellaswag", precision="fp16", limit=20)
    save_run(run)
    print(
        f"{run.model_id} [{run.precision}] on {run.benchmark}: "
        f"{run.runtime_seconds:.1f}s, peak {run.peak_memory_mb} MB"
    )
    for m in run.metrics:
        print(f"  {m.metric_name}: {m.value}")
