"""Spreadsheet parsing + category rollup for Budget Tracking and Reconciliation.

Finance and budget sheets are laid out every which way — title rows above the
header, twelve monthly columns that need summing, subtotal rows, hierarchical
categories. So we don't assume a clean shape: Claude reads a preview of the raw
grid and tells us how to read it (seo_claude.infer_layouts), and the helpers
here apply that mapping deterministically. The arithmetic (parsing money,
summing the chosen columns, grouping) stays here where it's exact; the *layout
interpretation* is Claude's.

Nothing here calls an API, so it's cheap and testable in isolation.
"""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

# Column-name hints, most-specific first — used only for the fallback guess when
# Claude's layout read is unavailable or invalid.
_CATEGORY_HINTS = [
    "category", "line item", "line-item", "channel", "campaign", "budget line",
    "expense type", "type", "vendor", "account", "description", "memo", "name",
    "item",
]
_AMOUNT_HINTS = [
    "actual", "spend", "planned", "budget", "amount", "total", "cost", "price",
    "usd", "value", "debit",
]


class BudgetError(Exception):
    """Raised when an uploaded file can't be read or has no usable columns."""


def read_raw(name: str, data: bytes) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX into a header-less all-string grid.

    We read with header=None so Claude — not pandas — decides which row is the
    header; every cell is a string and blanks are "" so the preview is faithful
    to the sheet. Raises BudgetError on any parse failure.
    """
    lower = (name or "").lower()
    buf = io.BytesIO(data)
    try:
        if lower.endswith((".xlsx", ".xlsm", ".xls")):
            raw = pd.read_excel(buf, engine="openpyxl", header=None, dtype=str)
        else:
            raw = pd.read_csv(
                buf, header=None, dtype=str, encoding="utf-8-sig",
                keep_default_na=False,
            )
    except Exception as e:  # noqa: BLE001 — surface any parser error uniformly
        raise BudgetError(f"couldn't read “{name}”: {e}") from None

    raw = raw.fillna("")
    raw = raw.map(lambda v: str(v).strip())
    # Drop trailing all-blank columns/rows that widen the preview for nothing.
    raw = raw.loc[:, ~(raw == "").all(axis=0)]
    raw = raw.loc[~(raw == "").all(axis=1)]
    if raw.empty or not raw.shape[1]:
        raise BudgetError(f"“{name}” has no rows or columns to read")
    # Reset to a clean 0..n integer column/row index after the drops.
    raw.columns = range(raw.shape[1])
    return raw.reset_index(drop=True)


def preview_grid(raw: pd.DataFrame, *, max_rows: int = 25, max_cols: int = 40,
                 max_cell: int = 60) -> list[list[str]]:
    """A compact list-of-rows view of `raw` for Claude, capped in every
    dimension. Outer index = row, inner index = 0-based column."""
    out: list[list[str]] = []
    for _, row in raw.iloc[:max_rows].iterrows():
        cells = [str(v)[:max_cell] for v in row.tolist()[:max_cols]]
        out.append(cells)
    return out


def build_frame(raw: pd.DataFrame, header_row: int) -> pd.DataFrame:
    """Turn the raw grid into a real table, using `header_row` as the header.

    Column names come from that row (blanks become column_N; duplicates get a
    numeric suffix); the data is everything below it, blank rows dropped.
    """
    n_rows = raw.shape[0]
    header_row = max(0, min(int(header_row), n_rows - 1))

    header = raw.iloc[header_row].tolist()
    cols: list[str] = []
    seen: dict[str, int] = {}
    for i, h in enumerate(header):
        name = str(h).strip() or f"column_{i + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}.{seen[name]}"
        else:
            seen[name] = 0
        cols.append(name)

    data = raw.iloc[header_row + 1:].copy()
    data.columns = cols
    data = data.loc[~(data == "").all(axis=1)]  # drop fully-blank rows
    return data.reset_index(drop=True)


def _score_column(col: str, hints: list[str]) -> int:
    low = col.lower()
    for i, hint in enumerate(hints):
        if hint in low:
            return i
    return len(hints) + 1


def guess_category_column(df: pd.DataFrame) -> str:
    """Fallback best-guess for the category column (leftmost text-ish column)."""
    cols = list(df.columns)
    text_cols = [c for c in cols if not _looks_numeric(df[c])]
    pool = text_cols or cols
    return min(pool, key=lambda c: (_score_column(c, _CATEGORY_HINTS), cols.index(c)))


def guess_amount_column(df: pd.DataFrame, *, prefer: str | None = None) -> str:
    """Fallback best-guess for the money column."""
    cols = list(df.columns)
    numeric_cols = [c for c in cols if _looks_numeric(df[c])]
    pool = numeric_cols or cols

    def key(c: str) -> tuple:
        low = c.lower()
        pref_rank = 0 if (prefer and prefer in low) else 1
        return (pref_rank, _score_column(c, _AMOUNT_HINTS), cols.index(c))

    return min(pool, key=key)


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _to_number(value: Any) -> float:
    """Parse a cell into a float, tolerating $, commas and (accounting parens)."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value) if pd.notna(value) else 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    match = _NUM_RE.search(text.replace("−", "-"))  # unicode minus → ascii
    if not match:
        return 0.0
    num = float(match.group().replace(",", ""))
    return -num if negative and num > 0 else num


def _looks_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    sample = [str(v) for v in series.tolist() if str(v).strip()][:50]
    if not sample:
        return False
    hits = sum(1 for v in sample if _NUM_RE.search(v.replace("−", "-")))
    return hits >= max(1, len(sample) * 0.6)


def rollup(df: pd.DataFrame, category_columns: list[str],
           amount_columns: list[str], exclude_patterns: list[str] | None = None
           ) -> dict[str, float]:
    """Group `df` by (possibly multi-column) category and sum the chosen amount
    column(s) per category.

    - `category_columns` are joined with " · " to form each row's label; a row
      with all-blank category cells is bucketed under "(uncategorized)".
    - `amount_columns` are money-parsed and summed per row (handles a budget's
      Jan…Dec spread). Pass the single annual/total column instead when the
      sheet already totals the periods, to avoid double counting.
    - Rows whose label contains any `exclude_patterns` substring (case-
      insensitive) are dropped — subtotal/total lines.

    Returns {category: total}, totals rounded to cents, zero-total rows dropped.
    """
    cat_cols = [c for c in category_columns if c in df.columns]
    amt_cols = [c for c in amount_columns if c in df.columns]
    if not amt_cols:
        raise BudgetError("no valid amount column was selected")
    patterns = [p.lower() for p in (exclude_patterns or []) if p.strip()]

    totals: dict[str, float] = {}
    for _, row in df.iterrows():
        parts = [str(row[c]).strip() for c in cat_cols if str(row[c]).strip()]
        label = " · ".join(parts) if parts else "(uncategorized)"
        low = label.lower()
        if any(p in low for p in patterns):
            continue
        amt = sum(_to_number(row[c]) for c in amt_cols)
        totals[label] = totals.get(label, 0.0) + amt

    return {k: round(v, 2) for k, v in totals.items() if round(v, 2) != 0.0}
