# Exla On-Demand GPU — Integration Context & Feature Spec

**Audience:** whoever is working in the **Exla** codebase (e.g. a separate Claude Code session).
**Purpose:** implement the features Exla needs so that **quant-bench** can use Exla to run
benchmarks on-demand — instead of Modal. You do **not** need to know quant-bench's internals
beyond what's in §1; this doc is self-contained.

**Status:** proposal / handoff brief. Assumptions are called out explicitly in **[ASSUMPTION]**
tags — correct them where Exla actually differs.

---

## 0. Plain-language summary

quant-bench is a website where someone pastes a Hugging Face model id and gets back a
benchmark comparing the quantized ("shrunk") model to its full-precision original. If we've
already benchmarked it, the site answers instantly. If not, the request goes onto a **queue**
(a table in a Postgres database), and *something with a GPU* has to pick the job up, run the
benchmark (a few minutes of GPU work), save the numbers, and email the requester a link.

Today that "something with a GPU" is a **human manually starting an Exla box** and running a
script. We want to make it **automatic**: a request arrives → a GPU boots itself → it runs
the pending jobs → it shuts down when there's nothing left to do (so we don't pay for idle
time). That auto-boot / auto-shutdown is the piece Exla needs to provide.

**The one-sentence ask:** give quant-bench an authenticated HTTP endpoint it can call that
ensures a GPU worker is running, where that worker drains a job queue and the instance shuts
itself down when the queue is empty.

---

## 1. Background: how quant-bench works today (context you need)

- **Database:** Neon **Postgres**. One connection string, `DATABASE_URL`.
- **Job queue:** a table `bench_requests`. Each row is one benchmark request with a
  `status` in `pending → running → done | failed | rejected`. Relevant columns:
  `id`, `hf_repo` (the model), `task` (which benchmark, default `hellaswag`), `precision`,
  `status`, `attempts`, `error`, `contact_email`, `param_count`, `public_token`,
  `result_hf_repo`, timestamps.
- **The worker** (`worker.py` in the quant-bench repo) is the code that does the GPU work.
  Its loop, per job:
  1. `claim_next_request()` — atomically grabs the oldest `pending` row and flips it to
     `running` (uses Postgres `FOR UPDATE SKIP LOCKED`, so **multiple workers are safe** —
     they never grab the same job).
  2. Runs the benchmark on the GPU (`run_comparison(...)`, minutes long, needs CUDA +
     `bitsandbytes`).
  3. Saves results to Postgres and emails the requester (`notify.py`, via Resend).
  4. `mark_request(..., 'done'|'failed')`.
- **The website** runs on Vercel as a serverless function. It has **no GPU** and a short
  time limit, so it can only *enqueue* — it cannot run a benchmark. It's the piece that will
  call Exla's trigger endpoint.
- **GPU deps:** the benchmark stack (`requirements-bench.txt`) is heavy: a CUDA build of
  `torch`, `transformers`, `lm_eval`, `bitsandbytes`, `accelerate`. This must be installed on
  the GPU instance. The Exla box we've used is an **A10-class GPU (~$1.31/hr)**.

**What's manual today and must become automatic:** a human boots the Exla box and runs
`python3 worker.py`. Jobs submitted while it's off just wait. Exla's job is to remove the
human.

---

## 2. Target flow (Exla as the provisioner)

```
 visitor          Vercel (quant-bench web)         Exla                     GPU instance
   │  submit model     │                             │                          │
   ├──────────────────►│                             │                          │
   │                   │ enqueue row (status=pending)│                          │
   │                   │ in Neon Postgres            │                          │
   │                   │                             │                          │
   │                   │  POST /trigger (authed)     │                          │
   │                   ├────────────────────────────►│                          │
   │  "we'll email you"│                             │ GPU already running?     │
   │◄──────────────────┤ 202 Accepted (returns fast) │   no → boot one ─────────►│ (cold start)
   │                   │                             │                          │ run drain worker:
   │                   │                             │                          │  claim pending jobs,
   │                   │                             │                          │  benchmark, save,
   │                   │                             │                          │  email requester
   │                   │                             │  queue empty → shut down ◄┤ exit
   │  email w/ link    │                             │                          │
   │◄─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (sent by the worker, not Vercel) ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
```

The queue in Postgres is the **source of truth**. The trigger HTTP call is just a "doorbell";
if it's ever missed, a running worker still finds every `pending` row. This makes the whole
thing robust to lost triggers and safe against duplicate triggers.

---

## 3. Features Exla must implement

### 3.1 Trigger endpoint ("the doorbell") — **required**
- An HTTP endpoint, e.g. `POST /v1/benchmark/trigger`.
- **Authenticated** with a shared secret (see §3.6) — this endpoint can spend money by
  booting GPUs, so it must not be publicly callable.
- **Idempotent and fast.** It should return immediately (e.g. `202 Accepted`) after ensuring
  a worker is (or will be) running. It must **not** wait for the benchmark to finish.
- Body can be minimal or empty — the worker discovers work by reading the queue, not from the
  request body. (Optionally accept `{"request_token": "..."}` for logging/tracing.)
- **Behavior:** "ensure at least one drain worker is running." If one is already up, do
  nothing (return 202). If none, boot one (§3.2).

### 3.2 GPU instance provisioning — **required**
- Boot a GPU instance on demand (A10-class is our baseline; see §7 for sizing).
- The instance must come up with the benchmark environment ready:
  - The `requirements-bench.txt` stack installed (CUDA `torch` matching the driver, plus
    `transformers`, `lm_eval`, `bitsandbytes`, `accelerate`). **[ASSUMPTION]** you bake this
    into an image so cold starts don't `pip install` from scratch.
  - The quant-bench worker code available (clone the repo, or bake it into the image).
  - The secrets/env from §3.6 injected.
- **[ASSUMPTION]** Exla can start and stop GPU instances programmatically (that's the premise
  of "add an API to Exla"). If it can't, that capability is the real prerequisite here.

### 3.3 Run the drain worker — **required**
- On the booted instance, run quant-bench's worker in **drain-once** mode: keep claiming and
  running `pending` jobs until the queue is empty, then **exit** (so the instance can shut
  down). This differs from today's `worker.py`, which polls forever.
  - quant-bench will add a `--once` / drain-until-empty mode to `worker.py` for this. **You
    just need to run that command**; the claim/run/save/notify logic is already written and
    tested. (Command will be roughly `python3 worker.py --drain`.)
- The worker already handles per-job failure, retries (`attempts`), and notifications, so Exla
  does not need to reimplement any of that.

### 3.4 Idle auto-shutdown — **required (this is the cost control)**
- When the drain worker exits (queue empty), the instance must **shut down** promptly. Paying
  for an idle A10 is the whole thing we're avoiding.
- Recommended: a short idle grace period (e.g. keep the box warm 60–120s after the queue
  empties) so a burst of requests arriving back-to-back reuses one warm GPU instead of
  cold-booting repeatedly. Tune to your cold-start cost.

### 3.5 Single-flight / concurrency control — **required**
- Concurrent triggers must **not** boot a swarm of GPUs. One running worker can drain the
  whole queue. Enforce a small **max concurrent instances** cap (start with **1**; raise later
  if throughput matters). Because job-claiming uses `SKIP LOCKED`, multiple workers are
  *correctness-safe* — this cap is purely about cost, not safety.

### 3.6 Secrets / environment on the instance — **required**
The GPU instance needs these (injected securely, never baked into an image or logged):

| Var | Purpose |
| --- | --- |
| `DATABASE_URL` | Neon Postgres — read the queue, write results |
| `HF_TOKEN` | Hugging Face token — download models (esp. gated ones) |
| `RESEND_API_KEY` | send the "results ready" email |
| `RESEND_FROM` | sender address (defaults to Resend's test sender if unset) |
| `PUBLIC_BASE_URL` | quant-bench's public URL, so email links are absolute |

The trigger endpoint additionally needs a shared secret, e.g. `EXLA_TRIGGER_SECRET`, that
Vercel sends and Exla checks.

### 3.7 Persistent model cache — **strongly recommended**
- Mount a **persistent volume** for the Hugging Face cache (`HF_HOME` / `~/.cache/huggingface`)
  that survives instance shutdown. Model weights are gigabytes; re-downloading them on every
  cold boot dominates run time and cost. A warm cache turns a multi-minute download into
  seconds.

### 3.8 Observability & failure handling — **recommended**
- Logs from each run retrievable (the worker prints structured `[worker] ...` lines).
- If an instance dies mid-job, the row is left `running`. The worker's retry logic covers
  crashes it catches; a hard instance kill won't be caught, so consider a **stale-job reaper**
  (a `running` row older than N minutes → back to `pending`). quant-bench can own this if you
  prefer — flag it either way.
- Surface boot failures / GPU-unavailable so they don't silently strand the queue.

---

## 4. Division of responsibility

| Concern | Owner |
| --- | --- |
| Trigger endpoint, auth check | **Exla** |
| Boot / shut down GPU instance | **Exla** |
| GPU image with benchmark deps + HF cache volume | **Exla** |
| Concurrency cap, idle shutdown | **Exla** |
| Claim job, run benchmark, save results, email | **quant-bench** (`worker.py`, done) |
| `--drain` (run-until-empty-then-exit) mode | **quant-bench** (small change, todo) |
| Call Exla's trigger after enqueue | **quant-bench** (`POST /request`, todo) |
| The job queue / results schema | **quant-bench** (done, no changes needed) |
| Size cap / rate limits / abuse gates | **quant-bench** (done in web layer) |

**Net for the Exla side:** you're building an authenticated "boot a GPU that runs one command,
then shuts down when idle, at most N at once" service. You are *not* writing any benchmark or
database code.

---

## 5. Concrete API contract (proposed — adjust to Exla conventions)

**Request** (Vercel → Exla):
```
POST /v1/benchmark/trigger
Authorization: Bearer <EXLA_TRIGGER_SECRET>
Content-Type: application/json

{ "request_token": "zyWMq1Ps..." }      // optional, for tracing only
```

**Response** (fast, non-blocking):
```
202 Accepted
{ "status": "worker-running" | "worker-started", "instance_id": "..." }
```

**Errors:**
- `401` if the secret is missing/wrong.
- `429` / `503` if the concurrency cap is hit and no capacity — quant-bench treats this as
  "the job stays queued; a running worker will get to it," so this is non-fatal.

---

## 6. Minimal first version (MVP) vs later

**MVP — smallest thing that removes the human:**
1. One authenticated trigger endpoint.
2. It boots **exactly one** A10 instance if none is running (cap = 1).
3. That instance runs `python3 worker.py --drain` and shuts down when the queue empties.
4. Secrets injected via env; HF cache on a persistent volume.

That alone delivers the whole "request → auto GPU → results → email → shutdown" loop.

**Later:** raise the concurrency cap for throughput, add keep-warm tuning, a stale-job reaper,
and a status endpoint for the website to show "a GPU is currently running your job."

---

## 7. Open questions / decisions

1. **Can Exla start/stop GPU instances via API today?** If not, that's the true first task —
   everything else assumes it.
2. **Cold-boot time?** Boot + image pull + (cached) model load. Drives the keep-warm window
   (§3.4). We can tolerate 1–2 min because the user is notified async by email, not waiting on
   a page.
3. **GPU tier vs model-size cap.** quant-bench currently refuses models >14B (an A10 ceiling).
   If Exla offers bigger GPUs, we can raise that cap; if smaller, we lower it. Keep them in
   sync.
4. **Who holds a daily spend cap / kill-switch?** A hard "max GPU-hours or max jobs per day,
   then stop booting" backstop. Can live in Exla (refuse to boot past a budget) or quant-bench
   (stop enqueuing). Decide where.
5. **Stale-job reaper owner** (§3.8) — Exla or quant-bench?
6. **Secrets delivery mechanism** — how does Exla inject `DATABASE_URL` etc. into an instance
   securely?

---

## 8. What quant-bench will do on its side (so you can plan the handshake)

- Add a **`--drain`** mode to `worker.py`: claim + run until `claim_next_request()` returns
  `None`, then exit (instead of polling forever). This is the exact command Exla runs.
- Add a call to Exla's trigger endpoint at the end of the `POST /request` route (after a row is
  enqueued), sending `EXLA_TRIGGER_SECRET`. Failure to reach Exla is non-fatal — the job stays
  queued.
- Provide the exact `pip`/env setup (already in `requirements-bench.txt`) and the worker
  command for your image build.

Ping back with the trigger endpoint's final URL + shape and the secret mechanism, and
quant-bench will wire the `POST /request` → Exla call to match.
