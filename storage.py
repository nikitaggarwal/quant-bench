"""
Storage layer for quant-bench results, backed by Postgres (Neon).

A benchmark RunResult maps onto two tables: `runs` holds one row of run-level
facts (model, quant, task, runtime, memory, device) and owns many `run_metrics`
rows (one per score: acc, acc_norm, ...). See quant_bench_schema.sql.

Connection string comes from the DATABASE_URL env var, or a DATABASE_URL= line
in a local .env file (never commit that file — it holds your password).
"""
import os

import psycopg2

from benchmark import RunResult
from metadata import model_meta, quant_meta

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "quant_bench_schema.sql")


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
            SET runtime_seconds = %s, peak_memory_mb = %s, device = %s, source = %s
            WHERE id = %s
            """,
            (run.runtime_seconds, run.peak_memory_mb, run.device, run.source, run_id),
        )
        return run_id

    cur.execute(
        """
        INSERT INTO runs
            (model_id, quant_method_id, task_id, runtime_seconds, peak_memory_mb,
             device, n_shot, eval_limit, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (model_id, quant_id, task_id, run.runtime_seconds, run.peak_memory_mb,
         run.device, run.n_shot, run.limit, run.source),
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


if __name__ == "__main__":
    init_db()
    print("Database initialized on Postgres.")
