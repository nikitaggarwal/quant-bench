"""
Request intake checks for on-demand live benchmarking (design doc §6).

These run in the *web layer* the moment someone submits a model, BEFORE anything
is queued or a GPU is ever touched. They are deliberately cheap and depend only
on huggingface_hub (no torch/transformers), so they fit in the Vercel serverless
function alongside app.py.

The single most important check is the size gate: every GPU run is real money, so
we refuse models too big to benchmark on the target GPU before spending anything.
"""
import os

from huggingface_hub import HfApi
from huggingface_hub.utils import (
    RepositoryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
)

# Size gate. An A10-class GPU comfortably benchmarks up to roughly 13B params;
# above this we refuse rather than risk an out-of-memory crash or a multi-hour
# run that burns money. This is the biggest single cost lever — see design §6.2
# and open question §8.2. Tune it to the GPU tier you actually run on.
MAX_PARAM_COUNT = 14_000_000_000  # 14B hard cap

_api = HfApi()


def _token():
    """
    Which credential to use for Hub lookups. We use an explicitly-configured
    HF_TOKEN (so gated models the org has access to still validate), but otherwise
    query anonymously (token=False) instead of letting huggingface_hub fall back
    to a possibly-stale cached login token that would 401 on public models.
    """
    return os.environ.get("HF_TOKEN") or False


class IntakeError(Exception):
    """A request that should be rejected before it ever reaches the queue.

    The message is safe to show the user (it explains why we can't run it).
    """


def resolve_param_count(model_id: str) -> int | None:
    """
    Best-effort parameter count for a model, read from its safetensors metadata
    on the Hub. Returns None when the Hub doesn't expose a count (e.g. a model
    published without safetensors weight metadata) — the caller decides how to
    treat "unknown".
    """
    try:
        info = _api.model_info(model_id, token=_token())
    except (RepositoryNotFoundError, GatedRepoError, HfHubHTTPError):
        return None
    st = getattr(info, "safetensors", None)
    if st is not None and getattr(st, "total", None):
        return int(st.total)
    return None


def validate_request(model_id: str) -> int | None:
    """
    Run the free, pre-queue safety checks on a requested model id. Returns the
    resolved param_count (possibly None if the Hub doesn't report one) on success.
    Raises IntakeError with a user-facing message if the request should be
    rejected.

    Checks, cheapest first:
      1. Existence / access — does this repo exist and can we read it?
      2. Size gate — is it small enough to benchmark on our GPU?
    """
    # 1. Existence / access. model_info raises for typos, 404s, and gated repos.
    try:
        info = _api.model_info(model_id, token=_token())
    except RepositoryNotFoundError:
        raise IntakeError(
            f"No model found at '{model_id}' on Hugging Face. Check the id "
            f"(it should look like 'org/model-name')."
        )
    except GatedRepoError:
        raise IntakeError(
            f"'{model_id}' is gated and we don't have access, so we can't "
            f"benchmark it automatically."
        )
    except HfHubHTTPError as e:
        raise IntakeError(
            f"Couldn't reach Hugging Face for '{model_id}' ({e}). Try again shortly."
        )

    # 2. Size gate. Reject only when we KNOW it's too big; an unknown count is
    #    allowed through (the worker's GPU is the backstop) — see module note.
    st = getattr(info, "safetensors", None)
    param_count = int(st.total) if st is not None and getattr(st, "total", None) else None
    if param_count is not None and param_count > MAX_PARAM_COUNT:
        raise IntakeError(
            f"'{model_id}' has ~{param_count / 1e9:.0f}B parameters, above our "
            f"current {MAX_PARAM_COUNT / 1e9:.0f}B limit for on-demand runs. "
            f"Larger models need a bigger GPU than we benchmark on."
        )

    return param_count
