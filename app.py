"""Streamlit dashboard for the SEO review tool.

Run with: `streamlit run app.py`
"""

import streamlit as st

st.set_page_config(page_title="SEO Review", page_icon="🔎", layout="wide")

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
    ]
)
nav.run()
