
import streamlit as st

from companion.constants import DEFAULTS

st.set_page_config(
    page_title="Reading Companion",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, default in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

from companion.ui import render_app

render_app()
