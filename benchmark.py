from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


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


def run_benchmark(model_id: str, task: str, limit: int | None = 20, device: str = "mps") -> list[BenchmarkResult]:
    lm = HFLM(pretrained=model_id, dtype="float32", device=device)
    results = simple_evaluate(model=lm, tasks=[task], limit=limit, num_fewshot=0)

    task_results = results["results"][task]
    skip_keys = {"alias", "sample_len", "sample_len_stderr"}

    output = []
    for key, value in task_results.items():
        if key in skip_keys or not isinstance(value, (int, float)):
            continue

        metric_name = key.split(",")[0]  # strip the ",none" filter suffix first

        if metric_name.endswith("_stderr"):
            continue  # skip for now — revisit once we decide how to display error bars

        output.append(BenchmarkResult(
            model_id=model_id,
            benchmark=task,
            metric_name=metric_name,
            value=value,
            precision="fp32",
            source="computed",
            n_shot=0,
            limit=limit,
        ))
    return output


def save_results(results: list[BenchmarkResult], path: str = "results.json"):
    try:
        with open(path, "r") as f:
            existing = json.load(f)
    except FileNotFoundError:
        existing = []

    existing.extend([asdict(r) for r in results])

    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"Saved {len(results)} result(s) to {path}")


if __name__ == "__main__":
    from storage import init_db, save_results, has_results

    init_db()

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    task = "hellaswag"

    if has_results(model_id, task):
        print(f"Already have results for {model_id} on {task}, skipping compute.")
    else:
        results = run_benchmark(model_id, task, limit=20)
        for r in results:
            print(r)
        save_results(results)
