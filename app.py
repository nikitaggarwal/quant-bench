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
import os
from collections import defaultdict

from flask import Flask, render_template, abort, request, redirect, url_for, flash

import storage
import intake

app = Flask(__name__)
# Needed to show flash() messages (e.g. a rejected request). Set FLASK_SECRET_KEY
# in the Vercel env; the dev fallback is fine locally but not for production.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-insecure-key")

# Rate limits for on-demand requests (design doc §6.4). Every accepted request is
# a potential GPU run, so cap how many one person can trigger.
MAX_REQUESTS_PER_IP_PER_HOUR = 3
MAX_REQUESTS_PER_EMAIL_PER_DAY = 5

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


def build_model_groups() -> dict:
    """
    The unified leaderboard, grouped by MODEL rather than split into per-task
    tabs. Returns:

        {
          "tasks":  [ {id, label, blurb, lower_is_better, value_kind,
                       primary_metric}, ... ]   # one column per benchmark
          "models": [ {model, hf_repo, hardware, base_precision,
                       rows: [ per-precision variant, ... ]}, ... ]
        }

    Each model section's rows are its precision variants (fp16 baseline first,
    then int8/int4 and any other quant that shows up), and each row carries that
    variant's primary-metric score for EVERY benchmark at once, plus memory
    saved and speedup versus the model's own full-precision baseline. This lets
    you eyeball quantization degradation across all benchmarks in a single row.

    The green heatmap wash is normalised per benchmark across ALL runs (every
    model + precision), so a cell's depth reflects where that value sits in the
    benchmark's global range — with perplexity flipped (deeper green = lower).
    """
    all_rows = _fetch_all()
    tasks_with_data = {r["task"] for r in all_rows}

    # Column set: the registered tasks that actually have data, in declared order.
    tasks = []
    for tid in TASK_ORDER:
        if tid in tasks_with_data:
            m = _task_meta(tid)
            tasks.append({
                "id": tid, "label": m["label"], "blurb": m["blurb"],
                "lower_is_better": m["lower_is_better"],
                "value_kind": m["value_kind"], "primary_metric": m["primary_metric"],
            })

    # Prefer the largest sample size per (model, task) — most signal.
    limit_of: dict[tuple, int] = {}
    for r in all_rows:
        if r["eval_limit"] is not None:
            k = (r["hf_repo"], r["task"])
            limit_of[k] = max(limit_of.get(k, 0), r["eval_limit"])

    # Collapse metric rows into one run per (model, precision, task).
    runs: dict[tuple, dict] = {}
    for r in all_rows:
        lim = limit_of.get((r["hf_repo"], r["task"]))
        if lim is not None and r["eval_limit"] != lim:
            continue
        key = (r["hf_repo"], r["quant_method"], r["task"])
        run = runs.setdefault(key, {
            "hf_repo": r["hf_repo"], "model": r["model"],
            "precision": r["quant_method"], "bits": r["bits"], "task": r["task"],
            "peak_memory_mb": r["peak_memory_mb"],
            "runtime_seconds": r["runtime_seconds"],
            "hardware": r["hardware"], "source": r["source"], "metrics": {},
        })
        run["metrics"][r["metric_name"]] = r["value"]

    # Global per-benchmark min/max on the primary metric -> heatmap range.
    task_range: dict[str, tuple] = {}
    for t in tasks:
        vals = [run["metrics"].get(t["primary_metric"])
                for run in runs.values() if run["task"] == t["id"]]
        vals = [v for v in vals if v is not None]
        if vals:
            task_range[t["id"]] = (min(vals), max(vals))

    # Group runs by model.
    by_model: dict[str, list] = defaultdict(list)
    for run in runs.values():
        by_model[run["hf_repo"]].append(run)

    models_out = []
    for hf_repo, mruns in by_model.items():
        model_name = mruns[0]["model"]
        hardware = sorted({r["hardware"] for r in mruns if r["hardware"]})

        # Runs of this model grouped by precision.
        by_prec: dict[str, list] = defaultdict(list)
        for run in mruns:
            by_prec[run["precision"]].append(run)

        def prec_bits(p: str):
            b = next((r["bits"] for r in by_prec[p] if r["bits"] is not None), None)
            return b if b is not None else -1

        # Baseline = fp16 if present, else fp32, else the highest-bit precision.
        # (Robust to a "reported"/published baseline row that wasn't measured.)
        if "fp16" in by_prec:
            base_prec = "fp16"
        elif "fp32" in by_prec:
            base_prec = "fp32"
        else:
            base_prec = max(by_prec, key=prec_bits)

        base_runs = by_prec[base_prec]
        base_peak = max((r["peak_memory_mb"] for r in base_runs
                         if r["peak_memory_mb"]), default=None)
        base_rt_by_task = {r["task"]: r["runtime_seconds"] for r in base_runs}
        base_by_task = {r["task"]: r for r in base_runs}
        base_val = {t["id"]: (base_by_task[t["id"]]["metrics"].get(t["primary_metric"])
                              if t["id"] in base_by_task else None)
                    for t in tasks}

        rows_out = []
        for prec in sorted(by_prec, key=lambda p: PRECISION_ORDER.get(p, 99)):
            pruns = by_prec[prec]
            by_task = {r["task"]: r for r in pruns}
            is_baseline = prec == base_prec

            # One cell per benchmark: the variant's primary-metric value + heat.
            cells = {}
            for t in tasks:
                run = by_task.get(t["id"])
                val = run["metrics"].get(t["primary_metric"]) if run else None
                heat = None
                if val is not None and t["id"] in task_range:
                    vmin, vmax = task_range[t["id"]]
                    span = vmax - vmin
                    if span == 0:
                        heat = 1.0
                    else:
                        norm = (val - vmin) / span
                        heat = (1 - norm) if t["lower_is_better"] else norm
                # Degradation versus this model's own baseline on this benchmark.
                delta = good = None
                bv = base_val.get(t["id"])
                if not is_baseline and val is not None and bv is not None:
                    delta = val - bv
                    good = (delta > 0) != t["lower_is_better"]
                cells[t["id"]] = {"value": val, "heat": heat,
                                  "delta": delta, "good": good}

            # Peak memory = footprint ceiling across this variant's runs.
            peak = max((r["peak_memory_mb"] for r in pruns
                        if r["peak_memory_mb"]), default=None)
            mem_pct_saved = None
            if not is_baseline and peak and base_peak:
                mem_pct_saved = (1 - peak / base_peak) * 100

            # Speedup over shared tasks only, so the ratio is apples-to-apples.
            speedup = None
            if not is_baseline:
                shared = [tk for tk, r in by_task.items()
                          if r["runtime_seconds"] and base_rt_by_task.get(tk)]
                if shared:
                    vt = sum(by_task[tk]["runtime_seconds"] for tk in shared)
                    bt = sum(base_rt_by_task[tk] for tk in shared)
                    if vt:
                        speedup = bt / vt

            total_rt = sum(r["runtime_seconds"] for r in pruns
                           if r["runtime_seconds"]) or None
            src = pruns[0]["source"]
            rows_out.append({
                "precision": prec, "bits": prec_bits(prec),
                "is_baseline": is_baseline,
                "is_reported": bool(src and src != "computed"),
                "source": src, "cells": cells,
                "peak_memory_mb": peak, "mem_pct_saved": mem_pct_saved,
                "speedup": speedup, "runtime_seconds": total_rt,
            })

        models_out.append({
            "model": model_name, "hf_repo": hf_repo, "hardware": hardware,
            "base_precision": base_prec, "rows": rows_out,
        })

    models_out.sort(key=lambda m: m["model"])
    return {"tasks": tasks, "models": models_out}


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
    return render_template("index.html", data=build_model_groups())


@app.route("/model/<path:hf_repo>")
def model_page(hf_repo):
    data = build_comparison(hf_repo)
    if data is None:
        abort(404)
    return render_template("compare.html", data=data)


@app.route("/request", methods=["POST"])
def request_benchmark():
    """
    Accept an on-demand benchmark request (design doc §1, steps 1-3). This is the
    only *write* path in the web app, and it never runs a benchmark itself — it
    validates cheaply, then drops a job on the queue for the GPU worker to pick up.

    Order matters: reject junk before spending anything, cheapest checks first.
    """
    hf_repo = (request.form.get("hf_repo") or "").strip()
    email = (request.form.get("email") or "").strip()
    task = (request.form.get("task") or "hellaswag").strip()

    if not hf_repo or not email:
        flash("Please enter both a Hugging Face model id and an email.", "error")
        return redirect(url_for("index"))
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        flash("That doesn't look like a valid email address.", "error")
        return redirect(url_for("index"))

    # 1. Do we already have it? Skip the queue and the GPU entirely.
    if storage.has_results(hf_repo, task):
        return redirect(url_for("model_page", hf_repo=hf_repo))

    # 2. Rate limits (cheap DB counts, before any Hub call or GPU spend).
    #    On Vercel the real client IP is in X-Forwarded-For, not remote_addr.
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr) or "").split(",")[0].strip()
    if ip and storage.count_recent_requests(requester_ip=ip, since_hours=1) >= MAX_REQUESTS_PER_IP_PER_HOUR:
        flash("You've made a few requests recently — please try again in a little while.", "error")
        return redirect(url_for("index"))
    if storage.count_recent_requests(contact_email=email, since_hours=24) >= MAX_REQUESTS_PER_EMAIL_PER_DAY:
        flash("That email has reached its daily request limit. Try again tomorrow.", "error")
        return redirect(url_for("index"))

    # 3. Validate existence + size on the Hub (free, still pre-queue).
    try:
        param_count = intake.validate_request(hf_repo)
    except intake.IntakeError as e:
        flash(str(e), "error")
        return redirect(url_for("index"))

    # 4. Enqueue. Dedup is DB-enforced: a second in-flight request for the same
    #    (model, task) returns the existing token instead of a second GPU run.
    token = storage.enqueue_request(
        hf_repo, task, contact_email=email,
        param_count=param_count, requester_ip=ip or None,
    )
    return redirect(url_for("request_status", token=token))


@app.route("/request/<token>")
def request_status(token):
    """Status page for a queued request (design doc §4). Polls itself while the
    job is pending/running, and links to the results once done."""
    req = storage.get_request_by_token(token)
    if req is None:
        abort(404)
    return render_template("request_status.html", req=req)


if __name__ == "__main__":
    # Port 5001 to avoid macOS AirPlay, which grabs 5000.
    app.run(host="127.0.0.1", port=5001, debug=True)
