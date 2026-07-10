"""
Result data types for quant-bench.

These plain dataclasses describe the *shape* of a benchmark result and carry no
dependency on torch / transformers / lm_eval. Keeping them here (rather than in
benchmark.py, which imports the heavy ML stack) lets the storage layer and the
web UI import them without pulling in CUDA-only libraries — which is what makes
the read-only web app deployable to a lightweight serverless host like Vercel.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class BenchmarkResult:
    model_id: str
    benchmark: str
    metric_name: str
    value: float
    precision: str
    source: str
    source_url: str | None = None
    n_shot: int = 0
    limit: int | None = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class RunResult:
    """
    One evaluation run of a (model, benchmark, precision) triple.

    Holds run-level facts (runtime, peak memory, device) once, plus the list of
    per-metric BenchmarkResult rows the run produced (acc, acc_norm, ...). This
    two-level shape maps directly onto the Postgres runs/metrics schema.
    """
    model_id: str
    benchmark: str
    precision: str
    device: str
    runtime_seconds: float
    peak_memory_mb: float | None
    n_shot: int
    limit: int | None
    source: str
    metrics: list[BenchmarkResult] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
