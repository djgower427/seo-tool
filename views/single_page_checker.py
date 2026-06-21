"""Single Page Checker — review a single URL for broken links, copy, and on-page SEO."""

from __future__ import annotations

import html
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import httpx
import markdown as md
import streamlit as st
from anthropic import Anthropic
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fpdf import FPDF

import seo_usage
from seo_review import (
    MODEL,
    build_prompt,
    check_links,
    extract_page_data,
    fetch_page,
)

load_dotenv()


def get_api_key() -> str | None:
    """Look up the Anthropic API key from env (local .env) or Streamlit secrets (Cloud)."""
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key:
        return env_key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except (FileNotFoundError, KeyError, AttributeError):
        return None


def normalize_url(raw: str) -> str:
    raw = raw.strip()
    if not urlparse(raw).scheme:
        raw = "https://" + raw
    return raw


def stream_review(client: Anthropic, prompt: str, usage_out: dict | None = None):
    """Yield text chunks from a streamed Claude response.

    After the stream is fully consumed, stashes the final message's token usage
    into `usage_out` (if given) so the caller can report consumption.
    """
    with client.messages.stream(
        model=MODEL,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for text in stream.text_stream:
            yield text
        if usage_out is not None:
            usage_out["usage"] = stream.get_final_message().usage


def run_review(url: str, api_key: str) -> dict:
    """Run the full pipeline and return everything needed to render the report."""
    client = Anthropic(api_key=api_key)

    with st.status(f"Reviewing {url}", expanded=True) as status:
        st.write("Fetching page…")
        final_url, html = fetch_page(url)

        st.write(f"Parsing HTML (final URL: {final_url})…")
        page = extract_page_data(final_url, html)
        st.write(
            f"Found **{len(page['links'])}** links, "
            f"**{len(page['images'])}** images, "
            f"**{page['word_count']}** words."
        )

        st.write(f"Checking {len(page['links'])} links…")
        link_progress = st.progress(0.0, text="Starting link checks…")

        def _on_progress(i: int, total: int, href: str) -> None:
            link_progress.progress(i / max(total, 1), text=f"[{i}/{total}] {href}")

        link_results = check_links(page["links"], final_url, on_progress=_on_progress)
        link_progress.empty()

        st.write("Streaming report from Claude…")
        prompt = build_prompt(page, link_results)

        # Live stream into a temporary placeholder so the user sees tokens arrive,
        # then clear it so the final tabbed render_result takes over.
        stream_placeholder = st.empty()
        usage_holder: dict = {}
        with stream_placeholder.container():
            full_text = st.write_stream(stream_review(client, prompt, usage_holder))
        stream_placeholder.empty()
        # Report Claude token consumption to the active usage tracker (if any).
        seo_usage.record_claude(MODEL, usage_holder.get("usage"))

        status.update(label=f"Review complete — {final_url}", state="complete")

    return {
        "url": url,
        "final_url": final_url,
        "page": page,
        "link_results": link_results,
        "report": full_text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def render_history_sidebar() -> dict | None:
    """Render past reviews in the sidebar. Return a selected one if clicked."""
    st.sidebar.header("History")
    history = st.session_state.get("history", [])
    if not history:
        st.sidebar.caption("No past reviews this session.")
        return None

    selected = None
    for idx, item in enumerate(reversed(history)):
        label = f"{item['timestamp'].split('T')[1]} — {item['final_url']}"
        if st.sidebar.button(label, key=f"hist-{idx}", use_container_width=True):
            selected = item
    return selected


def _safe_filename(final_url: str, timestamp: str) -> str:
    domain = urlparse(final_url).netloc or "report"
    safe_domain = re.sub(r"[^a-zA-Z0-9.-]", "_", domain)
    safe_ts = re.sub(r"[^0-9]", "", timestamp)[:14]  # YYYYMMDDHHMMSS
    return f"seo-review-{safe_domain}-{safe_ts}.pdf"


HEADING_COLOR = "#16a34a"

_LATIN1_REPLACEMENTS = {
    "—": "--",   # em dash
    "–": "-",    # en dash
    "‘": "'",    # left single quote
    "’": "'",    # right single quote
    "“": '"',    # left double quote
    "”": '"',    # right double quote
    "…": "...",  # ellipsis
    "•": "-",    # bullet
    "→": "->",   # right arrow
}


def _sanitize_for_latin1(text: str) -> str:
    """fpdf2's built-in Helvetica is latin-1; substitute common smart chars, drop the rest."""
    for src, dst in _LATIN1_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _prepare_report_html(html_str: str) -> str:
    """Flatten table cells (fpdf2 limitation) and tint headings green."""
    soup = BeautifulSoup(html_str, "html.parser")
    for cell in soup.find_all(["td", "th"]):
        text = cell.get_text(" ", strip=True)
        cell.clear()
        cell.append(text)
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading["color"] = HEADING_COLOR
    return str(soup)


def report_to_pdf_bytes(result: dict) -> bytes:
    """Render a review result as a PDF and return the bytes."""
    page = result["page"]
    link_results = result["link_results"]
    broken = sum(
        1 for r in link_results
        if r["error"] or (r["status"] is not None and r["status"] >= 400)
    )
    missing_alt = sum(1 for i in page["images"] if not i["alt"])
    report_html = md.markdown(result["report"], extensions=["extra", "sane_lists"])

    body_html = f"""
<h1 color="{HEADING_COLOR}">SEO Review</h1>
<p><b>URL:</b> {html.escape(result["final_url"])}<br>
<b>Generated:</b> {html.escape(result["timestamp"])}</p>
<p><b>Word count:</b> {page["word_count"]}<br>
<b>Links:</b> {len(link_results)} ({broken} broken)<br>
<b>Images:</b> {len(page["images"])} ({missing_alt} missing alt)<br>
<b>Title:</b> {html.escape(page["title"] or "(none)")}<br>
<b>Meta description:</b> {html.escape(page["meta_description"] or "(none)")}</p>
<hr>
{_prepare_report_html(report_html)}
"""

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.write_html(_sanitize_for_latin1(body_html))

    out = pdf.output()  # fpdf2 returns bytearray
    return bytes(out)


def render_result(result: dict) -> None:
    """Render a saved result (already-completed review)."""
    st.subheader(f"Report — {result['final_url']}")
    st.caption(f"Run at {result['timestamp']}")
    tabs = st.tabs(["Report", "Page data", "Link results"])

    with tabs[0]:
        try:
            pdf_bytes = report_to_pdf_bytes(result)
            st.download_button(
                "📄 Download as PDF",
                data=pdf_bytes,
                file_name=_safe_filename(result["final_url"], result["timestamp"]),
                mime="application/pdf",
            )
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")
        st.markdown(result["report"])

    with tabs[1]:
        page = result["page"]
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Words", page["word_count"])
            st.metric("Links", len(page["links"]))
        with col2:
            st.metric("Images", len(page["images"]))
            missing_alt = sum(1 for i in page["images"] if not i["alt"])
            st.metric("Images missing alt", missing_alt)

        st.markdown("**Title**")
        st.code(page["title"] or "(none)", language=None)
        st.markdown("**Meta description**")
        st.code(page["meta_description"] or "(none)", language=None)
        st.markdown("**Canonical**")
        st.code(page["canonical"] or "(none)", language=None)
        st.markdown("**Robots**")
        st.code(page["robots"] or "(none)", language=None)

        st.markdown("**Headings**")
        for level in ("h1", "h2", "h3"):
            with st.expander(f"{level} ({len(page['headings'][level])})"):
                for h in page["headings"][level]:
                    st.markdown(f"- {h}")

    with tabs[2]:
        results = result["link_results"]
        broken = [r for r in results if r["error"] or (r["status"] is not None and r["status"] >= 400)]
        ok = [r for r in results if not r["error"] and r["status"] is not None and r["status"] < 400]
        st.write(f"**{len(results)}** links · **{len(broken)}** broken · **{len(ok)}** ok")
        st.dataframe(
            [
                {
                    "status": r["status"] if r["status"] is not None else f"ERR: {r['error']}",
                    "internal": r["internal"],
                    "text": r["text"],
                    "href": r["href"],
                }
                for r in results
            ],
            use_container_width=True,
            hide_index=True,
        )


def render() -> None:
    st.title("🔎 Single Page Checker")
    st.caption("Enter a URL to get a full review: broken links, grammar, copywriting frameworks, and on-page SEO.")

    api_key = get_api_key()
    if not api_key:
        st.error(
            "`ANTHROPIC_API_KEY` is not set. "
            "Locally, add it to `.env`. On Streamlit Cloud, add it under **Settings → Secrets**."
        )
        st.stop()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    selected_from_history = render_history_sidebar()

    with st.form("review-form", clear_on_submit=False):
        url_input = st.text_input("URL", placeholder="https://example.com")
        submitted = st.form_submit_button("Run review", type="primary")

    if submitted and url_input.strip():
        url = normalize_url(url_input)
        # Reserve the top-of-result spot, then measure Claude token usage for
        # this review (pay-as-you-go, so consumed-only — no fixed balance).
        usage_slot = st.empty()
        tracker = seo_usage.start()
        try:
            result = run_review(url, api_key)
        except httpx.HTTPError as e:
            tracker.finish()
            st.error(f"Failed to fetch the page: {e}")
            return
        tracker.finish()
        tracker.render(usage_slot)
        st.session_state["history"].append(result)
        st.divider()
        render_result(result)
    elif selected_from_history is not None:
        render_result(selected_from_history)


render()
