"""Budget Tracking and Reconciliation — planned budget vs. actual spend.

Upload two spreadsheets: the planned marketing budget and a feed of actual
expenses from finance. Because the two sheets are laid out differently, Claude
first reads each layout (which row is the header, which columns are the category
and the money — seo_claude.infer_layouts); pandas then rolls each up to exact
category totals (seo_budget.rollup); and Claude reconciles the two — matching
differently-named categories and flagging over/under-spend
(seo_claude.reconcile_budget).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import seo_budget
import seo_usage
from seo_claude import ClaudeError, infer_layouts, reconcile_budget
from seo_keys import get_anthropic_key

# Session-state keys.
_SIG_KEY = "_budget_sig"           # signature of the current file pair
_LAYOUTS_KEY = "_budget_layouts"   # Claude's per-sheet layout read
_DETECT_USAGE_KEY = "_budget_detect_usage"
_RESULT_KEY = "_budget_result"     # last reconciliation (survives reruns)

_STATUS_LABEL = {
    "over": "🔴 Over",
    "under": "🟢 Under",
    "on_track": "✅ On track",
    "unbudgeted": "⚠️ Unbudgeted",
    "unspent": "⚪ Unspent",
}


def _money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):,.2f}"


def _esc(text: str) -> str:
    """Escape '$' so Streamlit markdown doesn't treat amounts as LaTeX math."""
    return text.replace("$", "\\$")


def _resolve(df: pd.DataFrame, indices: list[int]) -> list[str]:
    """Map 0-based column indices to column names present in `df`."""
    return [df.columns[i] for i in indices if 0 <= i < len(df.columns)]


def _render_result(result: dict) -> None:
    """Render a stored reconciliation: usage line, totals, table, flags."""
    if result.get("usage_md"):
        st.info(result["usage_md"])

    planned_roll = result.get("planned_roll", {})
    actual_roll = result.get("actual_roll", {})
    total_planned = round(sum(planned_roll.values()), 2)
    total_actual = round(sum(actual_roll.values()), 2)
    total_var = round(total_actual - total_planned, 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Planned", _money(total_planned))
    c2.metric("Actual", _money(total_actual))
    c3.metric(
        "Variance",
        _money(total_var),
        delta=f"{'over' if total_var > 0 else 'under'} budget",
        delta_color="inverse",
    )

    if result.get("summary"):
        st.markdown(_esc(result["summary"]))

    flags = result.get("flags") or []
    if flags:
        st.markdown("**Flags**")
        for f in flags:
            st.markdown(f"- {_esc(f)}")

    rows = result["categories"]
    table = pd.DataFrame(
        [
            {
                "Category": r["category"],
                "Status": _STATUS_LABEL.get(r["status"], r["status"]),
                "Planned": r["planned"],
                "Actual": r["actual"],
                "Variance": r["variance"],
                "Variance %": (
                    round(r["variance"] / r["planned"] * 100, 1) if r["planned"] else None
                ),
                "Note": r["note"],
            }
            for r in rows
        ]
    ).sort_values("Variance", ascending=False, ignore_index=True)

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Planned": st.column_config.NumberColumn(format="$%.2f"),
            "Actual": st.column_config.NumberColumn(format="$%.2f"),
            "Variance": st.column_config.NumberColumn(format="$%.2f"),
            "Variance %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
    st.download_button(
        "⬇️ Download reconciliation (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name="budget_reconciliation.csv",
        mime="text/csv",
    )


def _mapping_controls(kind: str, label: str, df: pd.DataFrame, layout: dict,
                      prefer: str, period_default: list[str] | None) -> dict:
    """Editable column mapping for one sheet, defaulting to Claude's read.

    When the sheet is a monthly matrix (period columns detected), the amount
    control becomes a month picker so you can limit the comparison to the months
    your actuals cover. Returns {"cat": [cols], "amt": [cols], "excl": [pats]}.
    """
    cols = list(df.columns)
    note = layout.get("note") or f"header row {layout.get('header_row', 0)}"
    st.markdown(f"**{label}** — {_esc(note)}")

    # Surface the data-row span so a misread trailing-totals block is visible.
    first = layout.get("first_data_row")
    last = layout.get("last_data_row")
    if first is not None and last is not None:
        st.caption(
            f"Reading spreadsheet rows {int(first) + 1}–{int(last) + 1} as line "
            f"items ({len(df)} rows); anything below is treated as summary/totals. "
            "Re-detect if that's wrong."
        )

    default_cat = _resolve(df, layout.get("category_columns", [])) or [
        seo_budget.guess_category_column(df)
    ]
    cat = st.multiselect(
        "Category column(s)", cols,
        default=[c for c in default_cat if c in cols], key=f"{kind}_cat",
    )

    period_cols = _resolve(df, layout.get("period_columns", []))
    if period_cols:
        default_months = period_default if period_default is not None else period_cols
        amt = st.multiselect(
            "Months / periods to include",
            period_cols,
            default=[c for c in default_months if c in period_cols] or period_cols,
            key=f"{kind}_amt",
            help="Pick only the months your actuals cover so the two sides line up.",
        )
    else:
        default_amt = _resolve(df, layout.get("amount_columns", [])) or [
            seo_budget.guess_amount_column(df, prefer=prefer)
        ]
        amt = st.multiselect(
            "Amount column(s) — summed", cols,
            default=[c for c in default_amt if c in cols], key=f"{kind}_amt",
        )

    excl = st.text_input(
        "Exclude rows whose category contains (comma-separated)",
        value=", ".join(layout.get("exclude_patterns", [])), key=f"{kind}_excl",
    )
    return {
        "cat": cat,
        "amt": amt,
        "excl": [p.strip() for p in excl.split(",") if p.strip()],
    }


def render() -> None:
    st.header("💰 Budget Tracking and Reconciliation")
    st.caption(
        "Upload your **planned budget** and the **actual marketing spend** feed "
        "from finance (CSV or Excel). Claude reads how each sheet is laid out, "
        "then reconciles them — matching differently-named categories, computing "
        "variance, and flagging where you're over- or under-spending."
    )

    anthropic_key = get_anthropic_key()
    if not anthropic_key:
        st.error(
            "`ANTHROPIC_API_KEY` is not set. Add it to Streamlit Cloud Secrets "
            "or to `.streamlit/secrets.toml` locally."
        )
        st.stop()

    c1, c2 = st.columns(2)
    planned_file = c1.file_uploader(
        "Planned budget", type=["csv", "xlsx", "xls"], key="_budget_planned"
    )
    actual_file = c2.file_uploader(
        "Actual expenses (finance feed)", type=["csv", "xlsx", "xls"],
        key="_budget_actual",
    )

    def _show_last() -> None:
        if st.session_state.get(_RESULT_KEY):
            st.divider()
            st.subheader("Last reconciliation")
            _render_result(st.session_state[_RESULT_KEY])

    if not (planned_file and actual_file):
        st.info("Upload both spreadsheets to begin.")
        _show_last()
        return

    planned_bytes = planned_file.getvalue()
    actual_bytes = actual_file.getvalue()
    sig = f"{planned_file.name}:{len(planned_bytes)}|{actual_file.name}:{len(actual_bytes)}"
    if st.session_state.get(_SIG_KEY) != sig:
        # New file pair — drop any stale layout read and result.
        for k in (_LAYOUTS_KEY, _DETECT_USAGE_KEY, _RESULT_KEY):
            st.session_state.pop(k, None)
        st.session_state[_SIG_KEY] = sig

    try:
        planned_raw = seo_budget.read_raw(planned_file.name, planned_bytes)
        actual_raw = seo_budget.read_raw(actual_file.name, actual_bytes)
    except seo_budget.BudgetError as e:
        st.error(f"Couldn't read a file — {e}")
        return

    # ── Stage 1: Claude reads each sheet's layout ────────────────────────────
    if st.button("① Detect layout with Claude", type="primary"):
        tracker = seo_usage.start()
        with st.spinner("Reading how your sheets are laid out…"):
            try:
                layouts = infer_layouts(
                    seo_budget.preview_grid(planned_raw),
                    seo_budget.preview_grid(actual_raw),
                    anthropic_key,
                )
            except ClaudeError as e:
                tracker.finish()
                st.error(f"Claude — {e}")
                return
        tracker.finish()
        st.session_state[_LAYOUTS_KEY] = layouts
        st.session_state[_DETECT_USAGE_KEY] = tracker.summary_md()
        st.session_state.pop(_RESULT_KEY, None)  # stale once layout changes

    layouts = st.session_state.get(_LAYOUTS_KEY)
    if not layouts:
        st.info("Click **① Detect layout** to have Claude read how your two sheets are structured.")
        _show_last()
        return

    pl_layout, ac_layout = layouts["planned"], layouts["actual"]
    try:
        planned_df = seo_budget.build_frame(
            planned_raw, pl_layout.get("header_row", 0),
            pl_layout.get("first_data_row"), pl_layout.get("last_data_row"),
        )
        actual_df = seo_budget.build_frame(
            actual_raw, ac_layout.get("header_row", 0),
            ac_layout.get("first_data_row"), ac_layout.get("last_data_row"),
        )
    except Exception as e:  # noqa: BLE001 — surface, don't white-screen the app
        st.error(f"Couldn't structure the sheets after layout detection: {type(e).__name__}: {e}")
        st.exception(e)  # full traceback (rendered by us, so Cloud won't redact it)
        if st.button("Clear detected layout and retry"):
            for k in (_LAYOUTS_KEY, _DETECT_USAGE_KEY):
                st.session_state.pop(k, None)
            st.rerun()
        return

    # If both sheets are monthly matrices, default the budget's months to the
    # ones the actuals actually cover — so a 6-month finance feed compares
    # against the same 6 budget months, not the full year.
    planned_periods = _resolve(planned_df, pl_layout.get("period_columns", []))
    actual_periods = _resolve(actual_df, ac_layout.get("period_columns", []))
    planned_month_default = planned_periods
    if planned_periods and actual_periods:
        actual_months = {
            seo_budget.month_number(c) for c in actual_periods
            if seo_budget.month_number(c)
        }
        aligned = [c for c in planned_periods if seo_budget.month_number(c) in actual_months]
        if aligned:
            planned_month_default = aligned

    # ── Review / adjust the mapping ──────────────────────────────────────────
    with st.expander("How Claude read your sheets — adjust if needed", expanded=True):
        if st.session_state.get(_DETECT_USAGE_KEY):
            st.caption(st.session_state[_DETECT_USAGE_KEY])
        if planned_periods and actual_periods and planned_month_default != planned_periods:
            st.caption(
                "📅 Your actuals look like they cover "
                f"{len(planned_month_default)} month(s); I've defaulted the budget "
                "to those months so the totals are comparable. Adjust below if needed."
            )
        mapping = {
            "planned": _mapping_controls(
                "planned", "Planned budget", planned_df, pl_layout, "budget",
                planned_month_default,
            ),
        }
        st.divider()
        mapping["actual"] = _mapping_controls(
            "actual", "Actual expenses", actual_df, ac_layout, "actual",
            actual_periods or None,
        )

    # ── Stage 2: roll up + reconcile ─────────────────────────────────────────
    if st.button("② Reconcile", type="primary"):
        if not mapping["planned"]["amt"] or not mapping["actual"]["amt"]:
            st.error("Select at least one amount column for each sheet.")
            return
        tracker = seo_usage.start()
        with st.spinner("Reconciling budget vs. actuals…"):
            try:
                planned_roll = seo_budget.rollup(
                    planned_df, mapping["planned"]["cat"], mapping["planned"]["amt"],
                    mapping["planned"]["excl"],
                )
                actual_roll = seo_budget.rollup(
                    actual_df, mapping["actual"]["cat"], mapping["actual"]["amt"],
                    mapping["actual"]["excl"],
                )
            except seo_budget.BudgetError as e:
                tracker.finish()
                st.error(f"Couldn't roll up the sheets — {e}")
                return
            if not planned_roll and not actual_roll:
                tracker.finish()
                st.warning(
                    "Both sheets rolled up to zero. Check that the Amount "
                    "columns hold numbers, not text."
                )
                return
            try:
                analysis = reconcile_budget(planned_roll, actual_roll, anthropic_key)
            except ClaudeError as e:
                tracker.finish()
                st.error(f"Claude — {e}")
                return
        tracker.finish()
        st.session_state[_RESULT_KEY] = {
            **analysis,
            "planned_roll": planned_roll,
            "actual_roll": actual_roll,
            "usage_md": tracker.summary_md(),
        }

    if st.session_state.get(_RESULT_KEY):
        st.divider()
        _render_result(st.session_state[_RESULT_KEY])


render()
