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

# Metadata for every task the leaderboard knows about. Tasks without data yet
# still appear (as "coming soon" tabs), so adding mmlu/wikitext is drop-in:
#   - primary_metric: the column the leaderboard ranks on.
#   - lower_is_better: True for perplexity-style metrics (wikitext), where a
#     SMALLER number is better — ranking and delta colouring flip accordingly.
#   - value_kind: how a metric value is rendered ("accuracy" -> percentage,
#     "perplexity" -> raw number). Drives the template's number formatting.
TASK_META = {
    "hellaswag": {
        "label": "HellaSwag",
        "blurb": "Commonsense sentence completion.",
        "primary_metric": "acc_norm",
        "lower_is_better": False,
        "value_kind": "accuracy",
    },
    "mmlu": {
        "label": "MMLU",
        "blurb": "Knowledge & reasoning across 57 subjects.",
        "primary_metric": "acc",
        "lower_is_better": False,
        "value_kind": "accuracy",
    },
    "wikitext": {
        "label": "WikiText-2",
        "blurb": "Language-modelling perplexity — lower is better.",
        "primary_metric": "word_perplexity",
        "lower_is_better": True,
        "value_kind": "perplexity",
    },
}
# Display order of task tabs (available ones sorted first at render time).
TASK_ORDER = ["hellaswag", "mmlu", "wikitext"]


def _task_meta(task: str) -> dict:
    """Metadata for a task, with a sensible default for anything unregistered."""
    return TASK_META.get(task, {
        "label": task, "blurb": "", "primary_metric": None,
        "lower_is_better": False, "value_kind": "accuracy",
    })


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


def build_leaderboard() -> list[dict]:
    """
    One ranked table per task. Rows are model x precision entries (so the
    leaderboard directly shows how int8/int4 stack up against fp16), ranked by
    the task's primary metric with lower-is-better handled correctly.

    Every task in TASK_ORDER is returned; ones with no data yet come back with
    available=False so the page can render a "coming soon" tab.
    """
    all_rows = _fetch_all()
    tasks_with_data = {r["task"] for r in all_rows}

    tasks_out = []
    for task in TASK_ORDER:
        meta = _task_meta(task)
        primary = meta["primary_metric"]
        available = task in tasks_with_data

        rows_out, columns = [], []
        if available:
            trows = [r for r in all_rows if r["task"] == task]

            # Prefer the largest sample size per model (most signal).
            repo_limit: dict[str, int] = {}
            for r in trows:
                if r["eval_limit"] is not None:
                    repo_limit[r["hf_repo"]] = max(
                        repo_limit.get(r["hf_repo"], 0), r["eval_limit"])

            # Collapse metric rows into one entry per (model, precision).
            runs: dict[tuple, dict] = {}
            for r in trows:
                lim = repo_limit.get(r["hf_repo"])
                if lim is not None and r["eval_limit"] != lim:
                    continue
                key = (r["hf_repo"], r["quant_method"])
                run = runs.setdefault(key, {
                    "model": r["model"], "hf_repo": r["hf_repo"],
                    "precision": r["quant_method"], "bits": r["bits"],
                    "peak_memory_mb": r["peak_memory_mb"],
                    "runtime_seconds": r["runtime_seconds"],
                    "metrics": {},
                })
                run["metrics"][r["metric_name"]] = r["value"]

            present = sorted({m for run in runs.values() for m in run["metrics"]})
            # Primary metric first, remaining metric columns after it.
            columns = ([primary] if primary in present else []) + \
                      [m for m in present if m != primary]
            rank_metric = primary if primary in present else (columns[0] if columns else None)

            ranked = [run for run in runs.values()
                      if rank_metric is not None and run["metrics"].get(rank_metric) is not None]
            ranked.sort(key=lambda x: x["metrics"][rank_metric],
                        reverse=not meta["lower_is_better"])

            # Per-model full-precision baseline (fp16 preferred, else fp32), used
            # to express each quantized row's memory/speed *improvement*.
            baseline_of: dict[str, dict] = {}
            for run in runs.values():
                if run["precision"] in ("fp16", "fp32"):
                    cur = baseline_of.get(run["hf_repo"])
                    if cur is None or run["precision"] == "fp16":
                        baseline_of[run["hf_repo"]] = run

            # Heat = where this row's primary metric sits in the task's range,
            # 0 (worst) -> 1 (best). Drives the green heatmap wash. Flip for
            # lower-is-better metrics so "best" is always the deepest green.
            vals = [run["metrics"][rank_metric] for run in ranked]
            vmin, vmax = (min(vals), max(vals)) if vals else (0, 0)
            span = vmax - vmin

            for i, run in enumerate(ranked, 1):
                run["rank"] = i
                run["primary_value"] = run["metrics"].get(rank_metric)
                run["is_baseline"] = run["precision"] in ("fp16", "fp32")

                if span == 0:
                    run["heat"] = 1.0
                else:
                    norm = (run["metrics"][rank_metric] - vmin) / span
                    run["heat"] = (1 - norm) if meta["lower_is_better"] else norm

                # Memory saved (%) and speedup vs this model's fp16 baseline.
                base = baseline_of.get(run["hf_repo"])
                run["mem_pct_saved"] = None
                run["speedup"] = None
                if base is not None and base is not run:
                    if run["peak_memory_mb"] and base["peak_memory_mb"]:
                        run["mem_pct_saved"] = (
                            1 - run["peak_memory_mb"] / base["peak_memory_mb"]) * 100
                    if run["runtime_seconds"] and base["runtime_seconds"]:
                        run["speedup"] = base["runtime_seconds"] / run["runtime_seconds"]
            rows_out = ranked

        tasks_out.append({
            "id": task,
            "label": meta["label"],
            "blurb": meta["blurb"],
            "available": available,
            "lower_is_better": meta["lower_is_better"],
            "value_kind": meta["value_kind"],
            "primary_metric": primary,
            "columns": columns,
            "rows": rows_out,
            "model_count": len({r["hf_repo"] for r in rows_out}),
        })

    # Available tasks first, otherwise keep declared order.
    tasks_out.sort(key=lambda t: (not t["available"], TASK_ORDER.index(t["id"])))
    return tasks_out


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

        meta = _task_meta(task)
        # Order metric columns with the task's primary metric first.
        primary = meta["primary_metric"]
        ordered_metrics = ([primary] if primary in metric_names else []) + \
                          [m for m in metric_names if m != primary]

        task_views.append({
            "task": task,
            "label": meta["label"],
            "blurb": meta["blurb"],
            "lower_is_better": meta["lower_is_better"],
            "value_kind": meta["value_kind"],
            "eval_limit": chosen_limit,
            "metric_names": ordered_metrics,
            "runs": ordered,
            "baseline_precision": baseline["precision"],
        })

    return {"model": model_name, "hf_repo": hf_repo, "tasks": task_views}


@app.route("/")
def index():
    return render_template("index.html", tasks=build_leaderboard())


@app.route("/model/<path:hf_repo>")
def model_page(hf_repo):
    data = build_comparison(hf_repo)
    if data is None:
        abort(404)
    return render_template("compare.html", data=data)


if __name__ == "__main__":
    # Port 5001 to avoid macOS AirPlay, which grabs 5000.
    app.run(host="127.0.0.1", port=5001, debug=True)
