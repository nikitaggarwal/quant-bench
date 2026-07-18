"""
Storage layer for quant-bench results, backed by Postgres (Neon).

A benchmark RunResult maps onto two tables: `runs` holds one row of run-level
facts (model, quant, task, runtime, memory, device) and owns many `run_metrics`
rows (one per score: acc, acc_norm, ...). See quant_bench_schema.sql.

Connection string comes from the DATABASE_URL env var, or a DATABASE_URL= line
in a local .env file (never commit that file — it holds your password).
"""
import os
import secrets

import psycopg2

from results import RunResult
from metadata import model_meta, quant_meta

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "quant_bench_schema.sql")
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")


def load_env(path: str = ENV_PATH):
    """
    Load KEY=VALUE lines from the .env file into os.environ, so local runs can
    keep secrets (DATABASE_URL, RESEND_API_KEY, ...) in one git-ignored file.

    Real environment variables win: we only fill in keys that aren't already set,
    so production (Vercel/Modal env vars) always takes precedence over .env.
    """
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _load_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if dsn:
        return dsn
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("DATABASE_URL="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    raise RuntimeError(
        "No database connection string found. Set the DATABASE_URL environment "
        "variable or add a DATABASE_URL= line to .env (copy it from your Neon "
        "dashboard)."
    )


def get_connection():
    return psycopg2.connect(_load_dsn())


def init_db():
    """Create the tables/view if they don't exist. Idempotent — safe to re-run."""
    with open(SCHEMA_PATH) as f:
        schema_sql = f.read()
    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(schema_sql)
    conn.close()
    print("Schema ensured on Postgres.")


def _get_or_create(cur, table, unique_col, unique_val, extra_cols=None):
    """Return the id of the row where unique_col=unique_val, inserting it first if absent."""
    cur.execute(f"SELECT id FROM {table} WHERE {unique_col} = %s", (unique_val,))
    row = cur.fetchone()
    if row:
        return row[0]

    extra_cols = extra_cols or {}
    cols = [unique_col] + list(extra_cols.keys())
    vals = [unique_val] + list(extra_cols.values())
    placeholders = ", ".join(["%s"] * len(vals))
    cur.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders}) RETURNING id",
        vals,
    )
    return cur.fetchone()[0]


def _model_id(cur, hf_repo: str) -> int:
    meta = model_meta(hf_repo)
    return _get_or_create(cur, "models", "hf_repo", hf_repo, {
        "display_name": meta["display_name"],
        "param_count": meta["param_count"],
        "family": meta["family"],
    })


def _quant_method_id(cur, precision: str) -> int:
    meta = quant_meta(precision)
    return _get_or_create(cur, "quant_methods", "name", precision, {
        "method_family": meta["method_family"],
        "bits": meta["bits"],
        "description": meta["description"],
    })


def _task_id(cur, task: str) -> int:
    return _get_or_create(cur, "tasks", "name", task)


def _upsert_run(cur, run: RunResult, model_id: int, quant_id: int, task_id: int) -> int:
    """
    Find the run for this (model, quant, task, n_shot, limit) and update it, or
    insert a fresh one. Uses IS NOT DISTINCT FROM so a NULL limit matches a NULL
    limit (plain = would treat two NULLs as different and duplicate the run).
    """
    cur.execute(
        """
        SELECT id FROM runs
        WHERE model_id = %s AND quant_method_id = %s AND task_id = %s
          AND n_shot = %s AND eval_limit IS NOT DISTINCT FROM %s
        """,
        (model_id, quant_id, task_id, run.n_shot, run.limit),
    )
    existing = cur.fetchone()
    if existing:
        run_id = existing[0]
        cur.execute(
            """
            UPDATE runs
            SET runtime_seconds = %s, peak_memory_mb = %s, device = %s,
                hardware = %s, source = %s
            WHERE id = %s
            """,
            (run.runtime_seconds, run.peak_memory_mb, run.device, run.hardware,
             run.source, run_id),
        )
        return run_id

    cur.execute(
        """
        INSERT INTO runs
            (model_id, quant_method_id, task_id, runtime_seconds, peak_memory_mb,
             device, hardware, n_shot, eval_limit, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (model_id, quant_id, task_id, run.runtime_seconds, run.peak_memory_mb,
         run.device, run.hardware, run.n_shot, run.limit, run.source),
    )
    return cur.fetchone()[0]


def save_run(run: RunResult):
    """
    Persist a RunResult and its metrics to Postgres.

    Re-running the same (model, quant, task, n_shot, limit) updates the existing
    run and its scores in place instead of duplicating them.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            model_id = _model_id(cur, run.model_id)
            quant_id = _quant_method_id(cur, run.precision)
            task_id = _task_id(cur, run.benchmark)
            run_id = _upsert_run(cur, run, model_id, quant_id, task_id)

            for m in run.metrics:
                cur.execute(
                    """
                    INSERT INTO run_metrics (run_id, metric_name, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (run_id, metric_name)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    (run_id, m.metric_name, m.value),
                )
        conn.commit()
    finally:
        conn.close()
    print(f"Saved run {run.model_id} [{run.precision}] on {run.benchmark} "
          f"({len(run.metrics)} metric(s)) to Postgres.")


def get_results(model_id: str, benchmark: str | None = None) -> list[dict]:
    """
    Look up stored results for a model as a list of dict rows (one per metric),
    read from the benchmark_results view. Optionally filter to one benchmark.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if benchmark:
                cur.execute(
                    "SELECT * FROM benchmark_results WHERE hf_repo = %s AND task = %s",
                    (model_id, benchmark),
                )
            else:
                cur.execute(
                    "SELECT * FROM benchmark_results WHERE hf_repo = %s",
                    (model_id,),
                )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def has_results(model_id: str, benchmark: str) -> bool:
    """The core 'lookup vs compute' check the website will call."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 1
                FROM runs r
                JOIN models m ON m.id = r.model_id
                JOIN tasks t ON t.id = r.task_id
                WHERE m.hf_repo = %s AND t.name = %s
                LIMIT 1
                """,
                (model_id, benchmark),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Job queue for on-demand live benchmarking (see docs/live-benchmarking-design.md).
# These operate purely on the bench_requests table — the *request lifecycle* — and
# never touch the results tables. save_run() above is still the only path results
# take into the database.
# ---------------------------------------------------------------------------

def enqueue_request(hf_repo: str, task: str = "hellaswag", *,
                    contact_email: str | None = None,
                    contact_phone: str | None = None,
                    precision: str | None = None,
                    param_count: int | None = None,
                    requester_ip: str | None = None) -> str:
    """
    Add a benchmark request to the queue and return its public token (the id used
    in a status/results URL).

    Dedup: if a request for this same (hf_repo, task) is already pending or running,
    no new row is created and the existing token is returned instead. The database
    enforces this via the uniq_bench_requests_inflight partial index, so two people
    asking for the same model at once can't spawn two GPU runs.
    """
    token = secrets.token_urlsafe(16)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bench_requests
                    (hf_repo, task, precision, contact_email, contact_phone,
                     param_count, requester_ip, public_token, result_hf_repo)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (hf_repo, task) WHERE status IN ('pending','running')
                DO NOTHING
                RETURNING public_token
                """,
                (hf_repo, task, precision, contact_email, contact_phone,
                 param_count, requester_ip, token, hf_repo),
            )
            row = cur.fetchone()
            if row is None:
                # Dedup hit: an in-flight request already exists — return its token.
                cur.execute(
                    """
                    SELECT public_token FROM bench_requests
                    WHERE hf_repo = %s AND task = %s
                      AND status IN ('pending','running')
                    ORDER BY created_at
                    LIMIT 1
                    """,
                    (hf_repo, task),
                )
                row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    return row[0] if row else token


def claim_next_request() -> dict | None:
    """
    Atomically claim the oldest pending request for a worker to run, flipping it to
    'running' and bumping its attempt count. Returns the claimed row as a dict, or
    None if the queue is empty.

    Uses FOR UPDATE SKIP LOCKED so several workers can poll at once and never grab
    the same job — the standard 'Postgres-as-a-queue' primitive.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bench_requests
                SET status = 'running', started_at = now(), attempts = attempts + 1
                WHERE id = (
                    SELECT id FROM bench_requests
                    WHERE status = 'pending'
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                )
                RETURNING *
                """
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if row else None
        conn.commit()
    finally:
        conn.close()
    return dict(zip(cols, row)) if row else None


def mark_request(request_id: int, status: str, *, error: str | None = None,
                 notified: bool = False):
    """
    Transition a request to a new lifecycle status ('running'/'done'/'failed'/
    'rejected'). Stamps finished_at for terminal states and notified_at when the
    contact has just been notified.
    """
    terminal = status in ("done", "failed", "rejected")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE bench_requests
                SET status = %s,
                    error = %s,
                    finished_at = CASE WHEN %s THEN now() ELSE finished_at END,
                    notified_at = CASE WHEN %s THEN now() ELSE notified_at END
                WHERE id = %s
                """,
                (status, error, terminal, notified, request_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_request_by_token(token: str) -> dict | None:
    """Look up a single request by its public token — powers the status page."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM bench_requests WHERE public_token = %s", (token,)
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description] if row else None
    finally:
        conn.close()
    return dict(zip(cols, row)) if row else None


def count_recent_requests(*, requester_ip: str | None = None,
                          contact_email: str | None = None,
                          contact_phone: str | None = None,
                          since_hours: int = 24) -> int:
    """
    Count how many requests a given IP / email / phone made in the last
    `since_hours` hours. Powers the rate limits in the design doc's §6. Pass exactly
    one of requester_ip / contact_email / contact_phone.
    """
    if requester_ip is not None:
        col, val = "requester_ip", requester_ip
    elif contact_email is not None:
        col, val = "contact_email", contact_email
    elif contact_phone is not None:
        col, val = "contact_phone", contact_phone
    else:
        raise ValueError(
            "Pass exactly one of requester_ip, contact_email, or contact_phone."
        )
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # `col` is one of three fixed literals above, never user input.
            cur.execute(
                f"""
                SELECT COUNT(*) FROM bench_requests
                WHERE {col} = %s
                  AND created_at > now() - make_interval(hours => %s)
                """,
                (val, since_hours),
            )
            return cur.fetchone()[0]
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized on Postgres.")
