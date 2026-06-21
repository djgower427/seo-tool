"""Magic Eight Ball — a chat where Claude answers from the app's data sources.

Claude is given a toolbox wired to the app's data sources (Apollo, Semrush,
HubSpot, Google), picks the relevant one(s), retrieves the data, and answers.
The conversation is kept in session state and fed back into the agent, so
follow-up questions have the full context of earlier answers and tool results.
See seo_agent.answer_question for the tool-use loop.
"""

from __future__ import annotations

import streamlit as st

import seo_usage
from seo_agent import answer_question
from seo_claude import ClaudeError
from seo_keys import (
    get_anthropic_key,
    get_apollo_key,
    get_google_ads_config,
    get_google_oauth_creds,
    get_gsc_site,
    get_hubspot_token,
    get_semrush_key,
)


def _render_answer(turn: dict) -> None:
    """Render one assistant turn: usage line first, then the answer and the
    sources it consulted."""
    if turn.get("usage_md"):
        st.info(turn["usage_md"])
    # Escape '$' so Streamlit's markdown doesn't treat dollar amounts as LaTeX
    # math (which silently strips spaces and stacks characters).
    st.markdown(turn["answer"].replace("$", "\\$"))
    tools_used = turn.get("tools_used") or []
    if tools_used:
        with st.expander(f"Sources consulted ({len(tools_used)})"):
            for t in tools_used:
                st.markdown(f"- `{t['name']}` — {t['input']}")


def render() -> None:
    st.header("Ask me a question")

    anthropic_key = get_anthropic_key()
    apollo_key = get_apollo_key()
    semrush_key = get_semrush_key()
    hubspot_token = get_hubspot_token()
    google_oauth = get_google_oauth_creds()
    gsc_site = get_gsc_site()
    google_ads_config = get_google_ads_config()

    if not anthropic_key:
        st.error(
            "`ANTHROPIC_API_KEY` is not set. Add it to Streamlit Cloud Secrets "
            "or to `.streamlit/secrets.toml` locally."
        )
        st.stop()

    sources = []
    if apollo_key:
        sources.append("Apollo (market companies & contacts)")
    if hubspot_token:
        sources.append("HubSpot (our CRM & campaigns)")
    if google_oauth:
        sources.append("Google Search Console (our organic search)")
    if google_oauth and google_ads_config:
        sources.append("Google Ads (our paid search)")
    if semrush_key:
        sources.append("Semrush (SEO & keywords)")
    if sources:
        st.caption(
            "I can pull from: " + ", ".join(sources) + ". Ask about a company's "
            "size/revenue/tech, SEO traffic and keywords, our CRM "
            "contacts/companies/deals, our website traffic, marketing campaign "
            "metrics, our organic Google clicks/queries, or our ad spend & "
            "performance. Answering may consume Apollo/Semrush credits."
        )
    else:
        st.warning(
            "No data sources are connected (set `APOLLO_API_KEY`, "
            "`SEMRUSH_API_KEY`, `HUBSPOT_ACCESS_TOKEN`, and/or Google OAuth "
            "secrets). I can only answer from general reasoning, not live data, "
            "until one is added."
        )

    # ── Chat state ───────────────────────────────────────────────────────────
    # _meb_chat holds display turns ({role, content/answer, usage_md, tools_used});
    # _meb_messages is the Anthropic-format conversation fed back to the agent so
    # follow-ups carry the full context of earlier answers and tool results.
    if "_meb_chat" not in st.session_state:
        st.session_state["_meb_chat"] = []
    if "_meb_messages" not in st.session_state:
        st.session_state["_meb_messages"] = []

    if st.session_state["_meb_chat"]:
        if st.button("🗑️ Clear chat"):
            st.session_state["_meb_chat"] = []
            st.session_state["_meb_messages"] = []
            st.rerun()

    # Replay the conversation so far.
    for turn in st.session_state["_meb_chat"]:
        with st.chat_message(turn["role"]):
            if turn["role"] == "assistant":
                _render_answer(turn)
            else:
                st.markdown(turn["content"])

    prompt = st.chat_input("Ask a question — or a follow-up on the last answer…")
    if not prompt or not prompt.strip():
        return
    prompt = prompt.strip()

    with st.chat_message("user"):
        st.markdown(prompt)

    # The agent may touch several allowances in one turn: Semrush units and
    # Apollo credits (balance deltas), Claude tokens (recorded in the agent
    # loop), plus rate-limited HubSpot/Google calls. Measure them around the run.
    usage_tracker = seo_usage.start(semrush_key=semrush_key, apollo_key=apollo_key)
    with st.chat_message("assistant"):
        with st.spinner("Consulting the eight ball…"):
            try:
                result = answer_question(
                    prompt,
                    anthropic_key=anthropic_key,
                    history=st.session_state["_meb_messages"],
                    apollo_key=apollo_key,
                    semrush_key=semrush_key,
                    hubspot_token=hubspot_token,
                    google_oauth=google_oauth,
                    gsc_site=gsc_site,
                    google_ads_config=google_ads_config,
                )
            except ClaudeError as e:
                usage_tracker.finish()
                st.error(f"Claude — {e}")
                return
        # Count the rate-limited providers the agent actually called.
        for t in result.get("tools_used") or []:
            name = t.get("name", "")
            if name.startswith("hubspot"):
                seo_usage.record_call("hubspot")
            elif name.startswith("gsc") or name.startswith("google_ads"):
                seo_usage.record_call("google")
        usage_tracker.finish()
        assistant_turn = {
            "role": "assistant",
            "answer": result["answer"],
            "usage_md": usage_tracker.summary_md(),
            "tools_used": result.get("tools_used") or [],
        }
        _render_answer(assistant_turn)

    # Persist the agent conversation and the display log for follow-ups/reruns.
    st.session_state["_meb_messages"] = result["messages"]
    st.session_state["_meb_chat"].append({"role": "user", "content": prompt})
    st.session_state["_meb_chat"].append(assistant_turn)


render()
