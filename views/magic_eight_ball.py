"""Magic Eight Ball — ask a free-text question, Claude answers from app data.

Claude is given a toolbox wired to the app's data sources (Apollo, Semrush),
picks the relevant one(s), retrieves the data, and returns a brief answer. See
seo_agent.answer_question for the tool-use loop.
"""

from __future__ import annotations

import streamlit as st

from seo_agent import answer_question
from seo_claude import ClaudeError
from seo_keys import (
    get_anthropic_key,
    get_apollo_key,
    get_hubspot_token,
    get_semrush_key,
)


def render() -> None:
    st.header("Ask me a question")

    anthropic_key = get_anthropic_key()
    apollo_key = get_apollo_key()
    semrush_key = get_semrush_key()
    hubspot_token = get_hubspot_token()

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
    if semrush_key:
        sources.append("Semrush (SEO & keywords)")
    if sources:
        st.caption(
            "I can pull from: " + ", ".join(sources) + ". Ask about a company's "
            "size/revenue/tech, its SEO traffic and keywords, our CRM "
            "contacts/companies/deals, our website traffic by source, marketing "
            "campaign metrics, or companies matching criteria. Answering may "
            "consume Apollo/Semrush credits."
        )
    else:
        st.warning(
            "No data sources are connected (set `APOLLO_API_KEY`, "
            "`SEMRUSH_API_KEY`, and/or `HUBSPOT_ACCESS_TOKEN`). I can only "
            "answer from general reasoning, not live data, until one is added."
        )

    with st.form("magic-eight-ball-form"):
        question = st.text_input(
            "Your question",
            label_visibility="collapsed",
            placeholder="e.g. How much organic traffic does stripe.com get?",
        )
        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        if not question.strip():
            st.error("Type a question first.")
            return
        with st.spinner("Consulting the eight ball…"):
            try:
                result = answer_question(
                    question.strip(),
                    anthropic_key=anthropic_key,
                    apollo_key=apollo_key,
                    semrush_key=semrush_key,
                    hubspot_token=hubspot_token,
                )
            except ClaudeError as e:
                st.error(f"Claude — {e}")
                return
        st.session_state["_meb_result"] = {"question": question.strip(), **result}

    result = st.session_state.get("_meb_result")
    if not result:
        return

    st.markdown(f"**Q:** {result['question']}")
    st.markdown("#### 🎱 Answer")
    with st.container(border=True):
        st.markdown(result["answer"])

    tools_used = result.get("tools_used") or []
    if tools_used:
        with st.expander(f"Sources consulted ({len(tools_used)})"):
            for t in tools_used:
                st.markdown(f"- `{t['name']}` — {t['input']}")


render()
