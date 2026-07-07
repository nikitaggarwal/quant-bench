-- quant-bench Postgres schema

CREATE TABLE IF NOT EXISTS models (
    id              SERIAL PRIMARY KEY,
    hf_repo         TEXT NOT NULL UNIQUE,
    display_name    TEXT NOT NULL,
    param_count     BIGINT,
    family          TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quant_methods (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    method_family   TEXT NOT NULL,
    bits            SMALLINT,
    description     TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    source          TEXT NOT NULL DEFAULT 'lm-evaluation-harness',
    description     TEXT
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id              SERIAL PRIMARY KEY,
    model_id        INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    quant_method_id INTEGER NOT NULL REFERENCES quant_methods(id) ON DELETE CASCADE,
    task_id         INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,

    accuracy        DOUBLE PRECISION,
    runtime_seconds DOUBLE PRECISION,
    peak_memory_mb  DOUBLE PRECISION,

    eval_limit      INTEGER,
    hardware        TEXT,
    notes           TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_runs_model ON benchmark_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_quant_method ON benchmark_runs(quant_method_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_runs_task ON benchmark_runs(task_id);

CREATE OR REPLACE VIEW benchmark_results AS
SELECT
    br.id,
    m.display_name  AS model,
    m.hf_repo,
    qm.name         AS quant_method,
    qm.bits,
    t.name          AS task,
    br.accuracy,
    br.runtime_seconds,
    br.peak_memory_mb,
    br.eval_limit,
    br.hardware,
    br.created_at
FROM benchmark_runs br
JOIN models m ON m.id = br.model_id
JOIN quant_methods qm ON qm.id = br.quant_method_id
JOIN tasks t ON t.id = br.task_id;
