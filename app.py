"""
quant-bench web UI.

A small Flask app that reads benchmark results from Neon (via the existing
storage layer) and shows, for each model, how its quantized variants compare to
the full-precision (fp16) baseline: what accuracy you lose, and what memory and
speed you gain.

Run it with:
    python3 app.py
then open http://127.0.0.1:5001

Read-only: it never writes to the database.
"""
from collections import defaultdict

from flask import Flask, render_template, abort

import storage

app = Flask(__name__)

# Lowest number = "most full precision"; used to order rows and pick a baseline.
PRECISION_ORDER = {"fp32": 0, "fp16": 1, "int8": 2, "int4": 3}


def _fetch_all() -> list[dict]:
    """Every row of the benchmark_results view, as a list of dicts."""
    conn = storage.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM benchmark_results")
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def list_models() -> list[dict]:
    """Distinct models that have any results, with the tasks they've been run on."""
    by_repo: dict[str, dict] = {}
    for r in _fetch_all():
        entry = by_repo.setdefault(
            r["hf_repo"], {"hf_repo": r["hf_repo"], "model": r["model"], "tasks": set()}
        )
        entry["tasks"].add(r["task"])
    models = [
        {"hf_repo": e["hf_repo"], "model": e["model"], "tasks": sorted(e["tasks"])}
        for e in by_repo.values()
    ]
    return sorted(models, key=lambda m: m["model"])


def build_comparison(hf_repo: str) -> dict | None:
    """
    Assemble the comparison view for one model: for each task, a set of
    per-precision runs with their metrics, plus deltas against the fp16 baseline.
    """
    rows = storage.get_results(hf_repo)
    if not rows:
        return None

    model_name = rows[0]["model"]

    # Group metric rows by task.
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    task_views = []
    for task, trows in sorted(by_task.items()):
        # If a model was run at several sample sizes, show the largest (most signal).
        limits = [r["eval_limit"] for r in trows if r["eval_limit"] is not None]
        chosen_limit = max(limits) if limits else None
        if chosen_limit is not None:
            trows = [r for r in trows if r["eval_limit"] == chosen_limit]

        # Collapse the metric rows into one entry per precision.
        runs: dict[str, dict] = {}
        for r in trows:
            run = runs.setdefault(r["quant_method"], {
                "precision": r["quant_method"],
                "bits": r["bits"],
                "runtime_seconds": r["runtime_seconds"],
                "peak_memory_mb": r["peak_memory_mb"],
                "device": r["device"],
                "metrics": {},
            })
            run["metrics"][r["metric_name"]] = r["value"]

        # Baseline = fp16 if present, else fp32, else the highest-bit run.
        baseline = runs.get("fp16") or runs.get("fp32")
        if baseline is None:
            baseline = max(runs.values(), key=lambda x: (x["bits"] or 0))

        metric_names = sorted({m for run in runs.values() for m in run["metrics"]})
        ordered = sorted(runs.values(),
                         key=lambda x: PRECISION_ORDER.get(x["precision"], 99))

        for run in ordered:
            run["is_baseline"] = run is baseline
            # Accuracy deltas vs baseline (per metric).
            run["deltas"] = {}
            for m in metric_names:
                v, bv = run["metrics"].get(m), baseline["metrics"].get(m)
                if v is not None and bv is not None:
                    run["deltas"][m] = v - bv
            # Memory saved and speedup vs baseline.
            run["mem_pct_saved"] = None
            if run["peak_memory_mb"] and baseline["peak_memory_mb"]:
                run["mem_pct_saved"] = (
                    1 - run["peak_memory_mb"] / baseline["peak_memory_mb"]
                ) * 100
            run["speedup"] = None
            if run["runtime_seconds"] and baseline["runtime_seconds"]:
                run["speedup"] = baseline["runtime_seconds"] / run["runtime_seconds"]

        task_views.append({
            "task": task,
            "eval_limit": chosen_limit,
            "metric_names": metric_names,
            "runs": ordered,
            "baseline_precision": baseline["precision"],
        })

    return {"model": model_name, "hf_repo": hf_repo, "tasks": task_views}


@app.route("/")
def index():
    return render_template("index.html", models=list_models())


@app.route("/model/<path:hf_repo>")
def model_page(hf_repo):
    data = build_comparison(hf_repo)
    if data is None:
        abort(404)
    return render_template("compare.html", data=data)


if __name__ == "__main__":
    # Port 5001 to avoid macOS AirPlay, which grabs 5000.
    app.run(host="127.0.0.1", port=5001, debug=True)
