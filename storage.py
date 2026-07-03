import sqlite3
from benchmark import BenchmarkResult

DB_PATH = "quantbench.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            benchmark TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            value REAL NOT NULL,
            precision TEXT NOT NULL,
            source TEXT NOT NULL,
            source_url TEXT,
            n_shot INTEGER NOT NULL,
            limit_n INTEGER,
            timestamp TEXT NOT NULL,
            UNIQUE(model_id, benchmark, metric_name, precision)
        )
    """)
    conn.commit()
    conn.close()


def save_results(results: list[BenchmarkResult]):
    conn = get_connection()
    for r in results:
        conn.execute("""
            INSERT INTO benchmark_results
                (model_id, benchmark, metric_name, value, precision, source, source_url, n_shot, limit_n, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id, benchmark, metric_name, precision)
            DO UPDATE SET value=excluded.value, source=excluded.source,
                          source_url=excluded.source_url, n_shot=excluded.n_shot,
                          limit_n=excluded.limit_n, timestamp=excluded.timestamp
        """, (r.model_id, r.benchmark, r.metric_name, r.value, r.precision,
              r.source, r.source_url, r.n_shot, r.limit, r.timestamp))
    conn.commit()
    conn.close()
    print(f"Saved {len(results)} result(s) to {DB_PATH}")


def get_results(model_id: str, benchmark: str | None = None) -> list[sqlite3.Row]:
    """Look up existing results for a model, optionally filtered to one benchmark."""
    conn = get_connection()
    if benchmark:
        rows = conn.execute(
            "SELECT * FROM benchmark_results WHERE model_id = ? AND benchmark = ?",
            (model_id, benchmark)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM benchmark_results WHERE model_id = ?",
            (model_id,)
        ).fetchall()
    conn.close()
    return rows


def has_results(model_id: str, benchmark: str) -> bool:
    """The core 'lookup vs compute' check your website will call."""
    return len(get_results(model_id, benchmark)) > 0


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")