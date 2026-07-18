"""
Notifications for finished benchmark requests (design doc §5).

MVP scope: email only, via Resend (https://resend.com) — a simple HTTP API, no
SMTP server to run. SMS (Twilio) is Phase 2.

Sent from the GPU worker after a run finishes, because the worker is the only
party that knows whether the run actually succeeded. Uses only stdlib urllib so
the GPU box needs no extra dependency.

Config via environment variables (never commit these):
    RESEND_API_KEY   - your Resend API key. If unset, sending is skipped (logged),
                       so the worker still runs end-to-end in local dev.
    RESEND_FROM      - verified sender, e.g. 'quant-bench <bench@yourdomain.com>'.
    PUBLIC_BASE_URL  - site root for building result links, e.g.
                       'https://quant-bench.vercel.app'.
"""
import json
import os
import urllib.error
import urllib.request

RESEND_ENDPOINT = "https://api.resend.com/emails"
DEFAULT_FROM = "quant-bench <onboarding@resend.dev>"  # Resend's shared test sender


def _public_base_url() -> str:
    return os.environ.get("PUBLIC_BASE_URL", "http://127.0.0.1:5001").rstrip("/")


def result_link(hf_repo: str) -> str:
    """Absolute URL to a model's comparison page."""
    return f"{_public_base_url()}/model/{hf_repo}"


def send_email(to: str, subject: str, html: str) -> bool:
    """
    Send one email via Resend. Returns True if sent, False if skipped or failed.
    Never raises — a notification failure must not crash the worker or lose the
    (already saved) benchmark result.
    """
    api_key = os.environ.get("RESEND_API_KEY")
    if not api_key:
        print(f"[notify] RESEND_API_KEY unset — would email {to!r}: {subject!r}")
        return False

    payload = json.dumps({
        "from": os.environ.get("RESEND_FROM", DEFAULT_FROM),
        "to": [to],
        "subject": subject,
        "html": html,
    }).encode()
    req = urllib.request.Request(
        RESEND_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            ok = 200 <= resp.status < 300
            print(f"[notify] emailed {to!r} ({resp.status}): {subject!r}")
            return ok
    except urllib.error.HTTPError as e:
        print(f"[notify] Resend rejected email to {to!r}: {e.code} {e.read()[:200]!r}")
    except urllib.error.URLError as e:
        print(f"[notify] couldn't reach Resend for {to!r}: {e.reason}")
    return False


def notify_success(email: str, hf_repo: str, task: str,
                   baseline_id: str | None = None) -> bool:
    """Tell the requester their results are ready, with a link to the comparison."""
    link = result_link(hf_repo)
    vs = f" against its baseline <b>{baseline_id}</b>" if baseline_id else ""
    html = (
        f"<p>We benchmarked <b>{hf_repo}</b>{vs} on <b>{task}</b>.</p>"
        f'<p><a href="{link}">See the full side-by-side comparison &rarr;</a></p>'
        f'<p style="color:#71767f;font-size:13px">{link}</p>'
        f'<p style="color:#71767f;font-size:13px">&mdash; quant-bench</p>'
    )
    return send_email(email, f"Your quant-bench results for {hf_repo} are ready", html)


def notify_failure(email: str, hf_repo: str, reason: str) -> bool:
    """Tell the requester we couldn't finish — plainly, without pretending it worked."""
    html = (
        f"<p>We couldn't finish benchmarking <b>{hf_repo}</b>:</p>"
        f"<p>{reason}</p>"
        f'<p style="color:#71767f;font-size:13px">&mdash; quant-bench</p>'
    )
    return send_email(email, f"We couldn't benchmark {hf_repo}", html)
