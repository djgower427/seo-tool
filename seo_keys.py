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


def get_anthropic_key() -> str | None:
    return _secret("ANTHROPIC_API_KEY")


def get_hubspot_token() -> str | None:
    """HubSpot Private App access token (used as a Bearer token)."""
    return _secret("HUBSPOT_ACCESS_TOKEN")


def get_google_oauth_creds() -> dict[str, str] | None:
    """OAuth client + refresh token for Google APIs (Search Console, Ads).

    Returns None unless all three secrets are present.
    """
    cid = _secret("GOOGLE_OAUTH_CLIENT_ID")
    cs = _secret("GOOGLE_OAUTH_CLIENT_SECRET")
    rt = _secret("GOOGLE_OAUTH_REFRESH_TOKEN")
    if cid and cs and rt:
        return {"client_id": cid, "client_secret": cs, "refresh_token": rt}
    return None


def get_gsc_site() -> str | None:
    """Default Search Console property URL (e.g. 'sc-domain:example.com' or
    'https://example.com/'). Optional — the agent can also discover/choose it."""
    return _secret("GSC_SITE_URL")


def get_google_ads_config() -> dict[str, str] | None:
    """Google Ads developer token + customer id (and optional manager id).

    Returns None unless the developer token and customer id are both set.
    """
    dev = _secret("GOOGLE_ADS_DEVELOPER_TOKEN")
    cust = _secret("GOOGLE_ADS_CUSTOMER_ID")
    if dev and cust:
        return {
            "developer_token": dev,
            "customer_id": cust,
            "login_customer_id": _secret("GOOGLE_ADS_LOGIN_CUSTOMER_ID"),
        }
    return None


def normalize_domain(raw: str) -> str:
    """Accept either a bare domain or a URL; return the bare host."""
    raw = raw.strip().lower()
    if "://" in raw:
        raw = urlparse(raw).netloc or raw
    return raw.lstrip("www.")
