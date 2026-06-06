"""Streamlit dashboard for the SEO review tool.

Run with: `streamlit run app.py`
"""

import streamlit as st

st.set_page_config(page_title="SEO Review", page_icon="🔎", layout="wide")

SITE_TITLE = "Dan's mid marketing app"

# Site header in the sidebar (above the nav) and as an H1 atop every page.
# Rendered here, before nav.run(), so it appears once across all pages.
st.sidebar.title(SITE_TITLE)
st.title(SITE_TITLE)

nav = st.navigation(
    [
        st.Page(
            "views/single_page_checker.py",
            title="Single Page Checker",
            icon="🔎",
            default=True,
        ),
        st.Page(
            "views/full_site_status.py",
            title="Full Site Status",
            icon="🌐",
        ),
        st.Page(
            "views/find_contacts.py",
            title="Find Contacts",
            icon="👥",
        ),
        st.Page(
            "views/target_finder.py",
            title="Target Finder",
            icon="🎯",
        ),
        st.Page(
            "views/competitor_mapping.py",
            title="Competitor / Market Mapping",
            icon="🗺️",
        ),
    ]
)
nav.run()
