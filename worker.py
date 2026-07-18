"""
GPU worker for on-demand live benchmarking (design doc §3 Option A, Phase 1).

Runs on the GPU box (e.g. an Exla A10). A simple poll loop:

    claim a pending request  ->  run the benchmark on the GPU  ->  results are
    saved to Neon by run_comparison  ->  mark the request done/failed  ->  email
    the requester a link.

This is the ONLY piece that needs the heavy GPU stack (requirements-bench.txt);
the web app never imports it. Job-claiming and result-saving are identical no
matter where the GPU lives, so this same logic moves onto Modal in Phase 2.

Run it with:
    python3 worker.py
Stop with Ctrl-C. Requests submitted while it's off just wait in the queue.

Env (plus the storage.py / notify.py vars):
    POLL_SECONDS   how often to check the queue when idle (default 10)
    BENCH_LIMIT    cap eval examples per task for cheaper/faster runs; unset = full
    MAX_ATTEMPTS   give up on a repeatedly-failing job after this many tries (default 3)
"""
import os
import time

import storage
import notify
from benchmark import run_comparison

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "10"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
_limit_env = os.environ.get("BENCH_LIMIT")
BENCH_LIMIT = int(_limit_env) if _limit_env else None


def process(req: dict):
    """Run one claimed request end-to-end and record its outcome."""
    hf_repo, task = req["hf_repo"], req["task"]
    email = req.get("contact_email")
    print(f"[worker] running {hf_repo} on {task} (attempt {req['attempts']})")

    # run_comparison detects the baseline, benchmarks both it and the quantized
    # model on the GPU, and saves both via storage.save_run — no results-schema
    # change. It returns baseline_id=None only when the baseline is undetectable.
    _quantized, _baseline, baseline_id = run_comparison(
        hf_repo, task,
        precision=req.get("precision") or "int4",
        limit=BENCH_LIMIT,
        device="cuda",
    )

    if baseline_id is None:
        reason = ("we couldn't automatically detect its full-precision baseline "
                  "from the model's metadata.")
        storage.mark_request(req["id"], "failed", error=reason, notified=bool(email))
        if email:
            notify.notify_failure(email, hf_repo, reason)
        print(f"[worker] failed {hf_repo}: no baseline detected")
        return

    storage.mark_request(req["id"], "done", notified=bool(email))
    if email:
        notify.notify_success(email, hf_repo, task, baseline_id=baseline_id)
    print(f"[worker] done {hf_repo} (baseline {baseline_id})")


def handle_error(req: dict, exc: Exception):
    """A run crashed. Retry until MAX_ATTEMPTS, then fail for good and notify."""
    if req["attempts"] < MAX_ATTEMPTS:
        storage.mark_request(req["id"], "pending")  # back on the queue
        print(f"[worker] error on {req['hf_repo']} "
              f"(attempt {req['attempts']}/{MAX_ATTEMPTS}), requeued: {exc}")
        return
    email = req.get("contact_email")
    storage.mark_request(req["id"], "failed", error=str(exc)[:500], notified=bool(email))
    if email:
        notify.notify_failure(email, req["hf_repo"], f"the benchmark run failed: {exc}")
    print(f"[worker] gave up on {req['hf_repo']} after {MAX_ATTEMPTS} attempts: {exc}")


def main():
    print(f"[worker] polling every {POLL_SECONDS}s "
          f"(limit={BENCH_LIMIT}, max_attempts={MAX_ATTEMPTS}). Ctrl-C to stop.")
    while True:
        req = storage.claim_next_request()
        if req is None:
            time.sleep(POLL_SECONDS)
            continue
        try:
            process(req)
        except Exception as exc:  # keep the worker alive across individual failures
            handle_error(req, exc)


if __name__ == "__main__":
    main()
