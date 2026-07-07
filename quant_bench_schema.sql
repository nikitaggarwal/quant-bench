-- quant-bench Postgres schema
--
-- Shape mirrors benchmark.py's RunResult: one *run* (model + quant + task,
-- with run-level facts like runtime/memory/device) owns many *metrics*
-- (acc, acc_norm, ...). A run is stored once in `runs`; each score it
-- produced is one row in `run_metrics`. New metrics need no schema change.

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

-- One evaluation run of a (model, quant method, task) triple. Run-level facts
-- live here once; the individual scores live in run_metrics.
CREATE TABLE IF NOT EXISTS runs (
    id              SERIAL PRIMARY KEY,
    model_id        INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    quant_method_id INTEGER NOT NULL REFERENCES quant_methods(id) ON DELETE CASCADE,
    task_id         INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,

    runtime_seconds DOUBLE PRECISION,
    peak_memory_mb  DOUBLE PRECISION,
    device          TEXT,

    n_shot          INTEGER,
    eval_limit      INTEGER,
    source          TEXT NOT NULL DEFAULT 'computed',
    hardware        TEXT,
    notes           TEXT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- One run per (model, quant, task, n_shot, limit); re-running upserts.
    UNIQUE (model_id, quant_method_id, task_id, n_shot, eval_limit)
);

-- One score produced by a run (e.g. acc = 0.42, acc_norm = 0.55).
CREATE TABLE IF NOT EXISTS run_metrics (
    id              SERIAL PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    metric_name     TEXT NOT NULL,
    value           DOUBLE PRECISION NOT NULL,

    UNIQUE (run_id, metric_name)
);

CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_quant_method ON runs(quant_method_id);
CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_run_metrics_run ON run_metrics(run_id);

-- Flat, human-readable view: one row per metric with its run's context
-- joined back in. This is the convenient surface for the website to query.
CREATE OR REPLACE VIEW benchmark_results AS
SELECT
    rm.id,
    m.display_name  AS model,
    m.hf_repo,
    qm.name         AS quant_method,
    qm.bits,
    t.name          AS task,
    rm.metric_name,
    rm.value,
    r.runtime_seconds,
    r.peak_memory_mb,
    r.device,
    r.n_shot,
    r.eval_limit,
    r.source,
    r.hardware,
    r.created_at
FROM run_metrics rm
JOIN runs r    ON r.id = rm.run_id
JOIN models m  ON m.id = r.model_id
JOIN quant_methods qm ON qm.id = r.quant_method_id
JOIN tasks t   ON t.id = r.task_id;
