"""Token / credit usage tracking for limited-allowance APIs.

Every billable request in the app should start its result with a line showing
how much of a limited allowance it consumed and how much remains. This module
provides the plumbing:

  * `UsageTracker` snapshots vendor balances around a unit of work and reports
    the delta (consumed) plus the post-call balance (remaining).
  * Semrush exposes a free units-balance endpoint, so its numbers are exact
    (consumed = before - after, remaining = after). A cache hit makes no API
    call, so the delta is correctly 0.
  * Apollo only exposes per-account usage via a MASTER-key-only endpoint. We
    auto-detect: if it answers, we use the real billing-cycle credit delta;
    if it 403s (ordinary key), we fall back to a known-cost model (each
    billable Apollo call reports its credit cost) plus a session running
    total, and remember not to retry the endpoint this session.
  * Claude is pay-as-you-go (no fixed allowance), so we report tokens consumed
    this request and a session total, but no "remaining".
  * HubSpot / Google are rate-limited, not credit-limited — we report call
    counts only, noted as having no depleting balance.

Thin clients report into the *active* tracker via a ContextVar, so they don't
need the tracker threaded through every call. `record_*` are no-ops when no
tracker is active, so the clients stay usable outside a tracked request.
"""

from __future__ import annotations

import contextvars
from typing import Any

import streamlit as st

# The tracker currently collecting usage for this script run, if any. A
# ContextVar (not a module global) so concurrent Streamlit sessions — each its
# own thread — never clobber one another's tracker.
_active: contextvars.ContextVar["UsageTracker | None"] = contextvars.ContextVar(
    "seo_usage_active_tracker", default=None
)

# Session-state keys for cross-request running totals and the cached Apollo
# master-key detection.
_TOTALS_KEY = "_usage_session_totals"
_APOLLO_USAGE_SUPPORTED_KEY = "_usage_apollo_supported"  # None=unknown, bool=known
_PENDING_KEY = "_usage_pending_md"  # summary stashed to render after an st.rerun()


# ── Per-call recording hooks (called by the thin clients) ────────────────────


def record_claude(model: str, usage: Any) -> None:
    """Record Claude token usage from a response's `.usage` object (or None)."""
    tracker = _active.get()
    if tracker is None or usage is None:
        return
    tracker._claude_in += int(getattr(usage, "input_tokens", 0) or 0)
    tracker._claude_out += int(getattr(usage, "output_tokens", 0) or 0)
    tracker._claude_calls += 1


def record_apollo_credits(n: int) -> None:
    """Record the known credit cost of one billable Apollo call (cost model).

    Used as the consumed figure when the master-key usage endpoint isn't
    available; harmless otherwise (we prefer the real balance delta when we
    have it).
    """
    tracker = _active.get()
    if tracker is None:
        return
    tracker._apollo_estimate += int(n)


def record_call(provider: str, n: int = 1) -> None:
    """Record N calls to a rate-limited provider (e.g. 'hubspot', 'google')."""
    tracker = _active.get()
    if tracker is None:
        return
    tracker._calls[provider] = tracker._calls.get(provider, 0) + int(n)


# ── Balance readers (best-effort; never raise into the page) ─────────────────


def _semrush_balance(api_key: str) -> int | None:
    """Remaining Semrush API units, or None if the lookup failed."""
    try:
        import seo_semrush

        return seo_semrush.units_balance(api_key)
    except Exception:
        return None


def _apollo_cycle_credits(api_key: str) -> int | None:
    """Total Apollo credits consumed this billing cycle (master key only).

    Returns None when the endpoint is unavailable (ordinary key → 403, or any
    other error). Caches a negative result in session state so we don't pay the
    latency of re-probing a non-master key on every request.
    """
    supported = st.session_state.get(_APOLLO_USAGE_SUPPORTED_KEY)
    if supported is False:
        return None
    try:
        import seo_apollo

        consumed = seo_apollo.credits_consumed_this_cycle(api_key)
    except Exception:
        consumed = None
    # Distinguish "endpoint works" from "endpoint unavailable": a successful
    # call returns an int (possibly 0); an unavailable one returns None.
    st.session_state[_APOLLO_USAGE_SUPPORTED_KEY] = consumed is not None
    return consumed


# ── Formatting helpers ───────────────────────────────────────────────────────


def _n(x: int | float | None) -> str:
    if x is None:
        return "—"
    return f"{int(x):,}"


# ── The tracker ──────────────────────────────────────────────────────────────


class UsageTracker:
    """Snapshots vendor balances around a unit of work and summarizes usage.

    Typical use:

        slot = st.empty()                       # reserve top-of-result spot
        with seo_usage.track(slot, semrush_key=k, apollo_key=a):
            ... billable work, may early-return ...
        # on exit the consumed/remaining line is rendered into `slot`

    Or, when the work is followed by st.rerun() (so an inline render would be
    discarded), stash the summary and render it on the next run:

        u = seo_usage.start(apollo_key=a)
        ... billable work ...
        u.finish()
        seo_usage.stash_pending(u.summary_md())
        st.rerun()
    """

    def __init__(
        self,
        *,
        semrush_key: str | None = None,
        apollo_key: str | None = None,
        slot: Any | None = None,
    ) -> None:
        self._semrush_key = semrush_key
        self._apollo_key = apollo_key
        self._slot = slot

        self._token = None
        self._started = False
        self._finished = False

        # Semrush (exact balance delta).
        self._semrush_before: int | None = None
        self._semrush_after: int | None = None

        # Apollo: real cycle-credit balance (master key) if available …
        self._apollo_before: int | None = None
        self._apollo_after: int | None = None
        self._apollo_real = False
        # … otherwise the cost-model estimate accumulated from billable calls.
        self._apollo_estimate = 0

        # Claude tokens (this request).
        self._claude_in = 0
        self._claude_out = 0
        self._claude_calls = 0

        # Rate-limited providers (call counts only).
        self._calls: dict[str, int] = {}

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> "UsageTracker":
        if self._started:
            return self
        self._started = True
        self._token = _active.set(self)
        if self._semrush_key:
            self._semrush_before = _semrush_balance(self._semrush_key)
        if self._apollo_key:
            self._apollo_before = _apollo_cycle_credits(self._apollo_key)
        return self

    def finish(self) -> "UsageTracker":
        if self._finished:
            return self
        self._finished = True
        if self._semrush_key:
            self._semrush_after = _semrush_balance(self._semrush_key)
        if self._apollo_key and self._apollo_before is not None:
            self._apollo_after = _apollo_cycle_credits(self._apollo_key)
            self._apollo_real = self._apollo_after is not None
        if self._token is not None:
            _active.reset(self._token)
            self._token = None
        self._update_session_totals()
        return self

    def __enter__(self) -> "UsageTracker":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.finish()
        if self._slot is not None:
            self.render(self._slot)
        return False  # never suppress (lets st.rerun()/exceptions propagate)

    # -- derived figures -----------------------------------------------------

    @property
    def semrush_consumed(self) -> int | None:
        if self._semrush_before is None or self._semrush_after is None:
            return None
        return max(0, self._semrush_before - self._semrush_after)

    @property
    def apollo_consumed(self) -> int:
        if self._apollo_real and self._apollo_before is not None and self._apollo_after is not None:
            return max(0, self._apollo_after - self._apollo_before)
        return self._apollo_estimate

    @property
    def claude_tokens(self) -> int:
        return self._claude_in + self._claude_out

    def _update_session_totals(self) -> None:
        totals = st.session_state.setdefault(
            _TOTALS_KEY,
            {"semrush_units": 0, "apollo_credits": 0, "claude_tokens": 0},
        )
        if self.semrush_consumed:
            totals["semrush_units"] += self.semrush_consumed
        if self.apollo_consumed:
            totals["apollo_credits"] += self.apollo_consumed
        if self.claude_tokens:
            totals["claude_tokens"] += self.claude_tokens

    # -- rendering -----------------------------------------------------------

    def _segments(self) -> list[str]:
        totals = st.session_state.get(_TOTALS_KEY, {})
        segs: list[str] = []

        if self._semrush_key:
            consumed = self.semrush_consumed
            remaining = self._semrush_after
            if consumed is None and remaining is None:
                segs.append("**Semrush** units balance unavailable")
            else:
                used = _n(consumed)
                cached = " (cached)" if consumed == 0 else ""
                segs.append(
                    f"**Semrush** {used} units used{cached} · {_n(remaining)} left"
                )

        if self._apollo_key:
            used = _n(self.apollo_consumed)
            if self._apollo_real:
                # Real billing-cycle figures from the master-key endpoint.
                segs.append(
                    f"**Apollo** {used} credits used · "
                    f"{_n(self._apollo_after)} used this billing cycle"
                )
            else:
                session_total = totals.get("apollo_credits", 0)
                est = "" if self.apollo_consumed == 0 else " (est.)"
                segs.append(
                    f"**Apollo** {used} credits used{est} · "
                    f"{_n(session_total)} this session "
                    "(live balance needs a master API key)"
                )

        if self.claude_tokens:
            sess = totals.get("claude_tokens", 0)
            segs.append(
                f"**Claude** {_n(self.claude_tokens)} tokens "
                f"({_n(self._claude_in)} in + {_n(self._claude_out)} out) · "
                f"{_n(sess)} this session"
            )

        for provider, count in self._calls.items():
            label = provider.capitalize()
            plural = "call" if count == 1 else "calls"
            segs.append(
                f"**{label}** {count} {plural} (rate-limited; no credit balance)"
            )

        return segs

    def summary_md(self) -> str:
        segs = self._segments()
        if not segs:
            return ""
        return "📊 **Usage this request** — " + "  ·  ".join(segs)

    def render(self, slot: Any | None = None) -> None:
        """Render the consumed/remaining line into `slot` (an st.empty()) or
        inline. Clears the slot when there's nothing to report."""
        md = self.summary_md()
        target = slot if slot is not None else st
        if not md:
            if slot is not None:
                slot.empty()
            return
        target.info(md)


# ── Module-level conveniences ────────────────────────────────────────────────


def start(
    *, semrush_key: str | None = None, apollo_key: str | None = None
) -> UsageTracker:
    """Create and start a tracker (snapshots balances immediately)."""
    return UsageTracker(semrush_key=semrush_key, apollo_key=apollo_key).start()


def track(
    slot: Any | None = None,
    *,
    semrush_key: str | None = None,
    apollo_key: str | None = None,
) -> UsageTracker:
    """Context manager that finishes and renders into `slot` on exit."""
    return UsageTracker(
        semrush_key=semrush_key, apollo_key=apollo_key, slot=slot
    ).start()


def stash_pending(md: str) -> None:
    """Stash a usage summary to render after an imminent st.rerun()."""
    if md:
        st.session_state[_PENDING_KEY] = md


def render_pending(slot: Any | None = None) -> None:
    """Render and clear a usage summary stashed before a prior st.rerun()."""
    md = st.session_state.pop(_PENDING_KEY, None)
    if not md:
        return
    target = slot if slot is not None else st
    target.info(md)
