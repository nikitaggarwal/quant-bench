# On-Demand Live Benchmarking — Design & Implementation Plan

**Status:** Design proposal (not yet built)
**Author:** (draft)
**Date:** 2026-07-10

---

## 0. Plain-language summary (start here)

Today quant-bench is two disconnected halves:

- **The website** (`app.py` + `templates/`) is deployed to Vercel. Vercel runs it as a
  "serverless function" — a small program that wakes up to answer one web request and then
  disappears. It has **no GPU** and a **short time limit** (seconds, not the ~15+ minutes a
  benchmark takes). It can only *read* results out of the Neon Postgres database via
  `storage.py`. It cannot run a benchmark itself.
- **The benchmark runner** (`benchmark.py`) needs a real NVIDIA GPU (because `bitsandbytes`,
  which does the int4/int8 quantization, only runs on CUDA). Right now a human rents an "Exla"
  A10 GPU box by hand (~$1.31/hr), runs `python3 benchmark.py`, and `storage.save_run()` writes
  the numbers into Neon.

**The feature we want:** a visitor types in any Hugging Face model ID. If we already have its
numbers in Neon, show them instantly (the site already does this). If we *don't*, the visitor
leaves an email and/or phone number, we run the benchmark on a GPU in the background, and when
it finishes we text/email them a link to the results.

The whole design problem is a consequence of one fact: **the part of the system that receives
the request (Vercel) is physically unable to do the work (GPU eval).** So the request and the
work have to be decoupled by a **queue**, and something with a GPU has to pick jobs off that
queue. Everything below follows from that.

**Jargon glossary** (terms used in this doc):

| Term | Plain meaning |
| --- | --- |
| Serverless function | Code that runs on-demand per request, then shuts down. No always-on server. |
| Queue | A to-do list of jobs. Producers add jobs; workers take them off and do them. |
| Worker | A long-running program that pulls jobs off the queue and executes them. |
| Cold start | The delay when a GPU has to boot up from nothing before it can do work. |
| Idempotent | Safe to do twice — running it again doesn't create duplicates or corrupt state. |
| Polling | Repeatedly asking "any new jobs?" on a timer, instead of being pushed a notification. |
| Webhook | An HTTP call *into* your app to tell it something happened (e.g. "the run finished"). |

---

## 1. End-to-end request flow

```
                          ┌──────────────────────────────────────────────────────┐
                          │                    NEON POSTGRES                       │
                          │   models / quant_methods / tasks / runs / run_metrics  │
                          │   + NEW: bench_requests  (the job queue)               │
                          └──────────────────────────────────────────────────────┘
        (1) submit model ID                 ▲            ▲                 │
        + email/phone                       │            │                 │
   ┌────────────┐   POST /request   ┌───────┴──────┐     │ (6) save_run()  │ (3) claim job
   │  Browser   │ ───────────────►  │  Vercel app  │     │   write metrics │  (poll or push)
   │ (visitor)  │                   │  (app.py)    │     │                 ▼
   └────────────┘                   └───────┬──────┘     │           ┌───────────┐
        ▲                                   │            │           │ GPU WORKER│
        │                                   │ (2) has_results()?     │(benchmark │
        │  (8) click link                   │   yes → return page    │   .py)    │
        │      /model/<repo>                │   no  → INSERT request  └─────┬─────┘
        │                                   │       status='pending'       │
        │                                   ▼                              │ (7) notify
        │                          ┌─────────────────┐                     │
        └──────────────────────────┤ Resend / Twilio │◄────────────────────┘
           email/SMS with link     └─────────────────┘
```

**Numbered sequence:**

1. **Request.** Visitor enters an HF repo ID (e.g. `nvidia/GLM-5.2-NVFP4`) and an email and/or
   phone number, and submits. This hits a **new** `POST /request` route in `app.py`.
2. **"Do we have it?" check.** The route calls the existing `storage.has_results(model_id, task)`
   (already in `storage.py`, lines 194–212). This is the exact "lookup vs compute" check the
   docstring already anticipates.
   - **Hit** → respond with the results link (`/model/<hf_repo>`, the existing route at
     `app.py:134`). No GPU, no job, done.
   - **Miss** → go to step 3.
3. **Enqueue.** Insert a row into a new `bench_requests` table with `status='pending'`, the
   requested model, contact info, and a generated public token. (After validation + safety
   checks from §6.) Return a "we're on it, we'll notify you" page to the visitor. Vercel's job
   is now *over* — it never runs the eval.
4. **Claim.** A GPU worker (see §3 for which kind) finds the oldest `pending` row and atomically
   flips it to `status='running'` so no other worker grabs the same job.
5. **Run.** The worker runs the existing `benchmark.py` code path — effectively
   `run_comparison(model_id, task)` (which internally calls `run_benchmark` and
   `detect_baseline`). This is the expensive GPU step, minutes long.
6. **Persist.** The worker calls the existing `storage.save_run(run)` for the quantized model
   and its baseline — **no change to the results schema at all.** This is the key reuse: the
   website already knows how to render anything in `runs` / `run_metrics`.
7. **Notify.** The worker updates `bench_requests.status='done'` and sends the notification via
   Resend (email) and/or Twilio (SMS), including the results link.
8. **Land.** Visitor clicks the link → the existing `/model/<hf_repo>` route renders the
   comparison via `build_comparison()` (`app.py:54`), which now finds the freshly-saved rows.

**Design principle:** steps 1–3 and 8 touch only the lightweight web layer and reuse existing
functions. Steps 4–7 are the only genuinely new runtime. The results schema is untouched; the
new `bench_requests` table is purely a *job/queue/notification* concern, kept separate from the
*results* concern.

---

## 2. How a serverless web app enqueues work it can't run

Vercel can't run the job, so it must hand it off. Two broad options:

| Option | What it is | Fit for quant-bench |
| --- | --- | --- |
| **A. Jobs table in Neon** (RECOMMENDED) | A `bench_requests` table acts as the queue. Vercel `INSERT`s; the worker `SELECT ... FOR UPDATE SKIP LOCKED` to claim. | We **already** have Neon wired up through `storage.py`. Zero new infrastructure, zero new bill, one connection string we already own. |
| **B. Hosted queue** (SQS, Upstash QStash, Redis, Cloud Tasks) | A dedicated managed queue service; Vercel publishes, worker subscribes. | More "correct" at high scale, but adds a second vendor, second set of secrets, second thing to learn. Overkill for a spiky, low-volume hobby-scale workload. |

### Recommendation: **Option A — a jobs table in Neon.**

Rationale specific to this codebase:

- The data layer is *already* Postgres via `psycopg2` (`storage.py`). Adding a table is a
  one-file change and reuses `get_connection()`.
- Postgres is a perfectly good low-volume queue. The pattern is well-known and race-safe:

  ```sql
  -- Worker claims exactly one job, atomically, even if several workers run:
  UPDATE bench_requests
  SET status = 'running', started_at = now(), attempts = attempts + 1
  WHERE id = (
      SELECT id FROM bench_requests
      WHERE status = 'pending'
      ORDER BY created_at
      FOR UPDATE SKIP LOCKED     -- other workers skip this locked row
      LIMIT 1
  )
  RETURNING *;
  ```

  `FOR UPDATE SKIP LOCKED` is the standard "Postgres-as-a-queue" primitive: two workers can
  poll simultaneously and never grab the same job.

- We avoid a distributed-systems tax we don't need. If volume ever grows past what polling a
  Postgres table handles comfortably (many jobs/second — implausible here, since each job is
  minutes of GPU time), we can graduate to Option B later without changing the results schema.

**How the GPU side picks it up** depends on the provisioning choice in §3:

- **Manual Exla box** → a `worker.py` loop that polls the `UPDATE ... SKIP LOCKED` query every N
  seconds and processes any claimed job. Simple, but the box (and its $/hr) must be running for
  jobs to move.
- **On-demand serverless GPU (Modal/RunPod)** → Vercel, right after the `INSERT`, makes one
  lightweight HTTP call to trigger a GPU function (a "webhook"/`.spawn()`). No always-on poller;
  the GPU boots, drains any `pending` rows, and shuts down. (The jobs table is still the source
  of truth / dedup ledger; the HTTP call is just the doorbell.)

---

## 3. GPU execution / provisioning options

This is the most consequential decision. The workload is **spiky and unpredictable**: long idle
stretches punctuated by a burst when someone requests an un-benchmarked model. Each job is
minutes of GPU time.

### Option A — Keep the manual Exla A10 + a polling worker

Add `worker.py` to the existing box: a loop that claims jobs (§2) and runs them.

- **Cost:** ~$1.31/hr **for every hour the box is on, whether or not any job is running.** To
  serve on-demand requests 24/7 you'd pay ~$1.31 × 24 × 30 ≈ **$943/month** to mostly sit idle.
  If you only turn it on manually, requests submitted while it's off sit in `pending` until you
  remember to boot it — which defeats the "live, on-demand" promise.
- **Cold start:** effectively zero *while the box is up* (model download aside); but "is the box
  up?" is a manual, human-in-the-loop question.
- **Ops:** you babysit a Linux CUDA box — driver/CUDA version pinning (see the sharp warnings in
  `requirements-bench.txt` about matching the cu128 wheel to driver 570.x), disk filling with HF
  model caches, the process dying, restarts. All on you.
- **Verdict:** fine as a *starting point* because it's what exists today and needs no new vendor,
  but it does not economically support genuine 24/7 on-demand load.

### Option B — On-demand serverless GPU (Modal or RunPod) — RECOMMENDED

A platform that **boots a GPU per job and shuts it down when idle.** You wrap `run_benchmark` in
a GPU function; the platform provisions hardware on demand and bills per second of use.

- **Cost:** you pay **only while a job runs.** If nobody requests anything, you pay ~$0. An A10-
  class GPU is roughly $1–1.30/hr *of actual compute*; a handful of jobs a day costs cents to a
  few dollars. This matches the spiky load far better than a $943/month idle box.
- **Cold start:** the real tradeoff. A cold GPU boot + container pull + model download can add
  tens of seconds to a couple of minutes before the eval starts. **For this feature that's fine
  — the user is already being notified asynchronously by email/SMS, so a 60–120s startup is
  invisible to them.** (Contrast with a synchronous "wait on the page" UX, where cold start
  would hurt.) Modal keeps images warm and caches weights on a persistent volume to shrink this.
- **Ops:** the platform owns drivers, CUDA, autoscaling, and shutdown. You own a Python function
  and a couple of secrets. Much less to babysit than a raw box.
- **Modal vs RunPod (brief):**
  - **Modal** — Python-native: decorate a function with a GPU spec, `pip install` the
    `requirements-bench.txt` stack into the image, mount a volume for the HF cache, and either
    `.spawn()` it from Vercel or run it on a schedule that drains `pending`. Best developer
    ergonomics for this Python codebase. **Recommended.**
  - **RunPod** — cheaper per-GPU-hour and more GPU variety; "serverless" endpoints exist but the
    wrapper is a bit more manual. A good fallback if Modal's pricing or GPU availability disappoints.

### Recommendation

**Target Option B (Modal) as the production design; use Option A (Exla) only for the Phase 1
MVP** because it's already set up and lets us prove the queue + notification flow end-to-end
before taking on a new vendor. The migration is small: `worker.py`'s *job-claiming and
result-saving logic is identical* regardless of where the GPU lives — only the "how does the GPU
get triggered/provisioned" wrapper changes. So Phase 1 code is not throwaway.

> **Cost-safety note:** on-demand billing (Option B) removes the idle cost but *raises* the
> per-request cost's visibility — every request is real money. That makes §6 (abuse/cost safety)
> non-optional before this is publicly reachable.

---

## 4. Job / state model

Keep this **entirely separate** from the results tables (`runs` / `run_metrics`). Those stay
exactly as they are — the whole point is that once a job finishes, the results flow through the
unchanged `save_run()` path and the existing website renders them. `bench_requests` only tracks
the *lifecycle of a request*.

### New table: `bench_requests`

```sql
CREATE TABLE IF NOT EXISTS bench_requests (
    id              SERIAL PRIMARY KEY,

    -- What was asked for
    hf_repo         TEXT NOT NULL,              -- requested HF model id, e.g. 'nvidia/GLM-5.2-NVFP4'
    task            TEXT NOT NULL DEFAULT 'hellaswag',  -- benchmark to run (matches tasks.name)
    precision       TEXT,                        -- optional; usually inferred (int4 for quantized)

    -- Lifecycle
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','running','done','failed','rejected')),
    attempts        SMALLINT NOT NULL DEFAULT 0, -- retry counter; cap to avoid infinite GPU spend
    error           TEXT,                        -- failure reason if status='failed'/'rejected'

    -- Contact (at least one of the two must be present — enforced below)
    contact_email   TEXT,
    contact_phone   TEXT,                        -- E.164 format, e.g. +14155550123
    notified_at     TIMESTAMPTZ,                 -- when we successfully sent the notification

    -- Safety / provenance (see §6)
    param_count     BIGINT,                      -- resolved from HF at enqueue time; size gate
    requester_ip    TEXT,                        -- for rate limiting
    public_token    TEXT NOT NULL UNIQUE,        -- unguessable id for a status/results URL

    -- The link we hand back once done
    result_hf_repo  TEXT,                        -- usually = hf_repo; lets the link point at results

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,

    CONSTRAINT contact_present CHECK (contact_email IS NOT NULL OR contact_phone IS NOT NULL)
);

-- Fast queue claim (pending rows, oldest first) and dedup lookups.
CREATE INDEX IF NOT EXISTS idx_bench_requests_status  ON bench_requests(status, created_at);
CREATE INDEX IF NOT EXISTS idx_bench_requests_repo    ON bench_requests(hf_repo);
CREATE INDEX IF NOT EXISTS idx_bench_requests_token   ON bench_requests(public_token);

-- Dedup: at most one in-flight request per (model, task). A partial unique index
-- makes the DB itself refuse a duplicate while one is pending/running.
CREATE UNIQUE INDEX IF NOT EXISTS uniq_bench_requests_inflight
    ON bench_requests(hf_repo, task)
    WHERE status IN ('pending','running');
```

**Status lifecycle:**

```
pending ──claim──► running ──save_run + notify──► done
   │                   │
   │                   └── run raised / crashed ──► failed  (retry if attempts < N, else notify failure)
   │
   └── failed validation / safety gate at enqueue ──► rejected  (never touches a GPU)
```

**Notes on the design:**

- `public_token` (a random UUID/slug) lets us give the visitor a **status page**
  (`/request/<token>`) without exposing the integer PK or requiring accounts. The notification
  link can point either at this status page or straight at `/model/<hf_repo>` once done.
- The partial unique index `uniq_bench_requests_inflight` is the **dedup mechanism from §6**,
  enforced by the database rather than app logic — two people requesting the same model at once
  can't spawn two GPU runs.
- We do **not** duplicate result data here. `result_hf_repo` is just a pointer so the notifier
  knows which `/model/<repo>` link to send; the numbers themselves live in `runs`/`run_metrics`.
- Add a matching `models.param_count` is already present (`quant_bench_schema.sql:12`); the
  `param_count` column here is a snapshot captured at enqueue time for the size gate, before we
  necessarily have a `models` row.

### `storage.py` additions (new functions, same style as existing ones)

- `enqueue_request(hf_repo, task, contact_email, contact_phone, param_count, requester_ip) -> token`
  — INSERT with `ON CONFLICT DO NOTHING` against the in-flight unique index (returns existing
  token if a dup).
- `claim_next_request() -> dict | None` — the `UPDATE ... FOR UPDATE SKIP LOCKED` from §2.
- `mark_request(id, status, error=None)` — transition helper (`running`/`done`/`failed`).
- `get_request_by_token(token) -> dict | None` — powers the status page.
- `count_recent_requests(ip_or_email, since) -> int` — powers rate limiting (§6).

All reuse the existing `get_connection()` and `_load_dsn()` — no new infrastructure.

---

## 5. Notifications

The user explicitly wants **both** email and SMS as contact options. Use:

- **Email → [Resend](https://resend.com):** simple HTTP API, generous free tier, good
  deliverability. Send with a single POST; no SMTP server to run.
- **SMS → [Twilio](https://www.twilio.com):** the standard for programmatic SMS. Requires a
  Twilio phone number and (for many countries) some sender registration.

### Where notifications are sent from

The **GPU worker** sends them (step 7 in §1), *after* `save_run()` succeeds and
`bench_requests.status='done'`. Reasons:

- The worker is the only party that knows the run actually finished (success or failure).
- It keeps Vercel's request path fast and free of long-running/secret-heavy calls.
- On the on-demand-GPU design, the worker already has the secrets in its environment.

(If you'd rather keep all secrets on Vercel, an alternative is: worker flips status to `done`,
and a tiny Vercel cron route `GET /cron/notify` picks up `done AND notified_at IS NULL` rows and
sends. Slightly more moving parts; only do this if you don't want to put Resend/Twilio keys on
the GPU side.)

### Secrets handling

Store every secret as an **environment variable**, never in the repo (`.env` is already
git-ignored; `storage.py:_load_dsn()` already models the "env var, fall back to `.env`" pattern —
reuse that exact approach for the new keys):

| Secret | Where it's needed |
| --- | --- |
| `DATABASE_URL` | Vercel + worker (already exists) |
| `RESEND_API_KEY` | wherever notifications are sent (worker, or Vercel cron) |
| `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER` | same |
| `HF_TOKEN` | worker (gated models; `hf auth login` equivalent) |
| `PUBLIC_BASE_URL` | to build absolute links in messages (e.g. `https://quant-bench.vercel.app`) |

On Vercel these go in **Project → Settings → Environment Variables**. On Modal use
`modal.Secret`; on the Exla box, an `.env` on the box (never committed). **Never** expose any of
these to the browser — all notification sending is server/worker-side only.

### Message content

Keep it short, lead with the result, always include the link.

**Success — email (Resend):**

> **Subject:** Your quant-bench results for `nvidia/GLM-5.2-NVFP4` are ready
>
> We benchmarked **nvidia/GLM-5.2-NVFP4** (int4) against its full-precision baseline
> **zai-org/GLM-5.2** on HellaSwag.
>
> - Accuracy (acc_norm): 0.68 vs 0.71 baseline (−0.03)
> - Peak memory: 41% lower · Speed: 1.6× faster
>
> Full side-by-side: {PUBLIC_BASE_URL}/model/nvidia/GLM-5.2-NVFP4
>
> — quant-bench

**Success — SMS (Twilio):** SMS is length-limited, so trim hard:

> quant-bench: results for nvidia/GLM-5.2-NVFP4 are ready.
> acc_norm 0.68 vs 0.71 baseline. {PUBLIC_BASE_URL}/model/nvidia/GLM-5.2-NVFP4

**Failure** (e.g. baseline undetectable — `detect_baseline` returned `None`, or the model OOM'd
the GPU): tell them plainly and don't pretend it worked.

> We couldn't finish benchmarking `nvidia/GLM-5.2-NVFP4`: we couldn't automatically detect its
> full-precision baseline. Reply with the baseline model ID and we'll try again. — quant-bench

### The results link

Build absolute URLs from `PUBLIC_BASE_URL` + the existing route shape `/model/<path:hf_repo>`
(`app.py:134` — note it already uses `<path:...>`, so slashes in repo IDs like `nvidia/...` work).
Optionally point at the status page `/request/<public_token>` instead, which then redirects to
the model page once `done` — nicer if you want a single stable link you can send *before* the run
finishes.

---

## 6. Abuse / cost safety

**Every GPU run is real money**, and with on-demand billing (§3B) a malicious or careless user
could rack up a bill by requesting many large models. These controls are **required before the
`POST /request` route is publicly reachable**, layered cheapest-first so we reject junk before
spending anything.

1. **Validate the model exists (free, at enqueue).** Reuse `huggingface_hub` — `detect_baseline`
   already imports `HfApi` and calls `api.model_info(model_id)`. Wrap that: if `model_info`
   raises (404 / gated / typo), mark the request `rejected` and tell the user. No GPU touched.

2. **Size gate (free, at enqueue).** From the same `model_info` (or `safetensors` metadata /
   `param_count`), reject models above a threshold (e.g. **> ~14B params for an A10**, and a hard
   cap like 34B regardless). Large models OOM the GPU and/or run for hours = big bills. Store the
   resolved `param_count` on the request row. This is the single most important cost control.
   (`metadata.py` already has a `param_count` concept — reuse it.)

3. **Dedup identical requests (DB-enforced).** The partial unique index
   `uniq_bench_requests_inflight` (§4) means a second request for the same `(hf_repo, task)`
   while one is `pending`/`running` returns the existing token instead of spawning a second run.
   And step 1 of the flow (`has_results`) already short-circuits anything we've *ever* computed —
   we never re-pay for a known model.

4. **Rate limiting (per IP and per contact).** Before enqueue, `count_recent_requests(...)`:
   cap e.g. **3 new GPU jobs per IP per hour** and **per email/phone per day**. Store
   `requester_ip`. Cheap SELECT against the indexed table.

5. **Global concurrency / spend cap.** Cap in-flight jobs (`COUNT(*) WHERE status='running'`) at
   a small number (e.g. 2). Optionally a **daily budget kill-switch**: if
   `COUNT(*) WHERE status IN ('running','done') AND created_at > today` exceeds N, stop accepting
   new jobs and show "at capacity, try tomorrow." Protects against a runaway bill even if other
   layers are bypassed.

6. **Contact verification (Phase 2+).** To stop someone spamming *other* people's inboxes/phones
   and to make rate limits meaningful, verify the contact before running: email a confirm-link or
   SMS a 6-digit code, and only enqueue once confirmed. In the MVP (§7) a lighter touch is fine
   (rate-limit + you eyeball the queue), but this matters before wide public exposure.

7. **Retry cap.** `attempts` column (§4): a job that keeps failing must not retry forever and
   burn GPU each time. Cap at 2–3, then mark `failed` and notify.

**Defense-in-depth summary:** existence + size gate reject junk for free; dedup + `has_results`
prevent paying twice; rate limits + concurrency + daily cap bound the worst case; contact
verification (later) closes the "spam strangers" hole.

---

## 7. Phased build plan

### Phase 0 — Groundwork (no new vendors, no GPU cost)

- Add `bench_requests` table to `quant_bench_schema.sql` (idempotent `CREATE TABLE IF NOT
  EXISTS`, so `storage.init_db()` picks it up).
- Add the `storage.py` helpers from §4 (`enqueue_request`, `claim_next_request`, `mark_request`,
  `get_request_by_token`, `count_recent_requests`).
- **Requires from user:** nothing new — just Neon (already have it).

### Phase 1 — Minimal end-to-end MVP (email only, Exla box)

Goal: prove request → queue → GPU → results → notify with the least new surface area.

- **Web:** add `POST /request` (validate existence + size, rate-limit, enqueue) and a
  `GET /request/<token>` status page. Add the input form to `templates/`.
- **Worker:** `worker.py` on the **existing Exla A10** — a poll loop that `claim_next_request()`,
  runs `run_comparison()`, `save_run()`, then sends **email via Resend**, and `mark_request`.
- **Notify:** **email only** (Resend). Skip SMS for now.
- **Scope cuts:** single task (`hellaswag`, already the default); single small model-size band;
  no contact verification (rate-limit + manual eyeballing instead); you start the Exla box by
  hand when you expect load.
- **Requires from user:**
  - A **Resend account** + verified sending domain (or their test domain) → `RESEND_API_KEY`.
  - The **Exla box running** while testing; awareness that jobs queue while it's off.
  - Vercel env vars: `DATABASE_URL` (exists), `RESEND_API_KEY`, `PUBLIC_BASE_URL`.
  - **Budget:** just the Exla hourly rate while it's on (~$1.31/hr). No new per-job vendor cost.

### Phase 2 — On-demand GPU + SMS + hardening

- **GPU:** move `worker.py`'s job body into a **Modal** GPU function (image built from
  `requirements-bench.txt`, HF cache on a persistent volume). Trigger it via `.spawn()` from the
  Vercel `POST /request` route (or a short Modal schedule that drains `pending`). Retire the
  always-on Exla box. Job-claim + `save_run` logic is unchanged from Phase 1.
- **Notify:** add **SMS via Twilio** as a second channel; send to whichever of
  email/phone the user provided (or both).
- **Safety:** add contact verification (§6.6), the global concurrency + daily budget cap, and the
  retry cap.
- **Requires from user:**
  - **Modal account** (payment method; pay-per-second GPU).
  - **Twilio account** + a phone number + (region-dependent) sender registration →
    `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`.
  - **Budget decision:** a monthly GPU ceiling and the per-request size cap.

### Phase 3 — Nice-to-haves (optional)

- Multiple tasks per request (MMLU + WikiText-2 perplexity — perplexity is the cheapest signal
  per the README and needs no generation, a good low-cost default).
- Live status streaming on the status page (queued → running → done) via polling.
- Auto-detect precision from the model card instead of assuming int4.
- Let users pick which benchmarks to run (with cost shown up front).

---

## 8. Open questions / decisions for the user

1. **GPU provider commitment.** Ship Phase 1 on the manual Exla box, then migrate to Modal in
   Phase 2 as recommended? Or jump straight to Modal? (Affects whether the always-on box cost is
   ever incurred.)
2. **Size cap number.** What's the max model size we'll benchmark on demand (param count / GPU
   tier)? This is the single biggest cost lever. A10-friendly is roughly ≤ ~13B; bigger needs a
   bigger/pricier GPU or is refused.
3. **Monthly GPU budget + kill-switch threshold.** What daily/monthly spend triggers "at
   capacity, try later"? Needed to bound the worst case.
4. **Which benchmark(s) per request.** Just HellaSwag (fast, current default), or also MMLU /
   WikiText-2 perplexity? More benchmarks = more accurate picture but more GPU minutes per job.
5. **Contact verification in MVP or defer?** Deferring is faster to ship but leaves a "spam a
   stranger's phone" hole and weaker rate limits. Acceptable for a small private beta; not for a
   public launch.
6. **Link target.** Notify with a direct `/model/<repo>` link, or a stable `/request/<token>`
   status page that redirects when done? The latter lets one link work before *and* after the
   run finishes.
7. **Baseline-not-found UX.** `detect_baseline` returns `None` for models HF doesn't tag. Do we
   (a) fail the request and ask the user for the baseline, or (b) benchmark the requested model
   alone with no comparison? Affects the failure message and whether we need a "provide baseline"
   input.
8. **Where notifications are sent from.** Worker-side (simplest, secrets on GPU) vs a Vercel
   `/cron/notify` route (keeps all secrets on Vercel)? §5 recommends worker-side.
