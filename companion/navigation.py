import streamlit as st

from .constants import DEFAULTS


def go_to_chunk(new_idx: int):
    total = len(st.session_state.pdf_chunks)
    st.session_state.current_chunk_idx = max(0, min(new_idx, total - 1))
    st.session_state.ai_commentary = ""
    st.session_state.ai_question = ""
    st.session_state.question_answered = False
    st.session_state.question_feedback = ""
    st.session_state.section_summary = ""
    st.session_state.tts_audio = b""
    st.session_state.tts_source = ""


def reset_session():
    for key, default in DEFAULTS.items():
        st.session_state[key] = default
