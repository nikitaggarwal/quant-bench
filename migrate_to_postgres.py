"""
Migrate quant_bench.db (SQLite) into the Postgres schema in quant_bench_schema.sql.
"""

import argparse
import sqlite3

import psycopg2

MODEL_METADATA = {
    "Qwen/Qwen2.5-0.5B-Instruct": {"display_name": "Qwen2.5 0.5B Instruct", "param_count": 500_000_000, "family": "Qwen"},
    "Qwen/Qwen2.5-1.5B-Instruct": {"display_name": "Qwen2.5 1.5B Instruct", "param_count": 1_500_000_000, "family": "Qwen"},
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"display_name": "TinyLlama 1.1B Chat", "param_count": 1_100_000_000, "family": "TinyLlama"},
}

QUANT_METADATA = {
    "fp16": {"method_family": "none", "bits": 16, "description": "Full precision baseline (fp16)"},
    "fp32": {"method_family": "none", "bits": 32, "description": "Full precision baseline (fp32)"},
    "int4": {"method_family": "bitsandbytes", "bits": 4, "description": "bitsandbytes 4-bit (nf4), dequantized on the fly per matmul"},
    "int8": {"method_family": "bitsandbytes", "bits": 8, "description": "bitsandbytes 8-bit"},
    "gguf_q4_k_m": {"method_family": "gguf", "bits": 4, "description": "GGUF Q4_K_M via llama.cpp"},
}


def get_or_create(cur, table, unique_col, unique_val, extra_cols=None):
    cur.execute(f"SELECT id FROM {table} WHERE {unique_col} = %s", (unique_val,))
    row = cur.fetchone()
    if row:
        return row[0]

    extra_cols = extra_cols or {}
    cols = [unique_col] + list(extra_cols.keys())
    vals = [unique_val] + list(extra_cols.values())
    placeholders = ", ".join(["%s"] * len(vals))
    col_names = ", ".join(cols)
    cur.execute(
        f"INSERT INTO {table} ({col_names}) VALUES ({placeholders}) RETURNING id",
        vals,
    )
    return cur.fetchone()[0]


def migrate(sqlite_path: str, pg_dsn: str):
    sqlite_conn = sqlite3.connect(sqlite_path)
    sqlite_conn.row_factory = sqlite3.Row
    rows = sqlite_conn.execute(
        "SELECT model_name, precision, task, accuracy, runtime_seconds, peak_memory_mb, created_at FROM results"
    ).fetchall()
    sqlite_conn.close()

    print(f"Read {len(rows)} rows from {sqlite_path}")
    if not rows:
        print("Nothing to migrate.")
        return

    pg_conn = psycopg2.connect(pg_dsn)
    cur = pg_conn.cursor()

    model_ids = {}
    quant_method_ids = {}
    task_ids = {}
    migrated = 0
    skipped = 0

    for row in rows:
        model_name = row["model_name"]
        precision = row["precision"]
        task = row["task"]

        if model_name not in model_ids:
            meta = MODEL_METADATA.get(model_name, {})
            model_ids[model_name] = get_or_create(
                cur, "models", "hf_repo", model_name,
                {
                    "display_name": meta.get("display_name", model_name),
                    "param_count": meta.get("param_count"),
                    "family": meta.get("family"),
                },
            )

        if precision not in quant_method_ids:
            meta = QUANT_METADATA.get(precision)
            if meta is None:
                print(f"WARNING: unknown precision label '{precision}', inserting with no bits/family metadata")
                meta = {"method_family": "unknown", "bits": None, "description": None}
            quant_method_ids[precision] = get_or_create(
                cur, "quant_methods", "name", precision,
                {"method_family": meta["method_family"], "bits": meta["bits"], "description": meta["description"]},
            )

        if task not in task_ids:
            task_ids[task] = get_or_create(cur, "tasks", "name", task)

        try:
            cur.execute(
                """
                INSERT INTO benchmark_runs
                    (model_id, quant_method_id, task_id, accuracy, runtime_seconds, peak_memory_mb, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    model_ids[model_name],
                    quant_method_ids[precision],
                    task_ids[task],
                    row["accuracy"],
                    row["runtime_seconds"],
                    row["peak_memory_mb"],
                    row["created_at"],
                ),
            )
            migrated += 1
        except Exception as e:
            print(f"FAILED to insert row ({model_name}, {precision}, {task}): {e}")
            pg_conn.rollback()
            skipped += 1
            continue

    pg_conn.commit()
    cur.close()
    pg_conn.close()

    print(f"\nMigrated {migrated} rows, skipped {skipped}.")
    print("eval_limit and hardware were left NULL for all rows — backfill directly in Postgres if you want them set.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite-path", default="quant_bench.db")
    parser.add_argument("--pg-dsn", required=True)
    args = parser.parse_args()

    migrate(args.sqlite_path, args.pg_dsn)
