"""Shared helpers used across view pages: API-key lookup and URL normalization.

Extracted out of views/full_site_status.py so other view modules can import
them without triggering the Full Site Status page render() side effect.
"""

from __future__ import annotations

from urllib.parse import urlparse

import streamlit as st


def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name)
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def get_semrush_key() -> str | None:
    return _secret("SEMRUSH_API_KEY")


def get_apollo_key() -> str | None:
    return _secret("APOLLO_API_KEY")


def normalize_domain(raw: str) -> str:
    """Accept either a bare domain or a URL; return the bare host."""
    raw = raw.strip().lower()
    if "://" in raw:
        raw = urlparse(raw).netloc or raw
    return raw.lstrip("www.")
