import streamlit as st

from .constants import fresh_defaults


def go_to_chunk(new_idx: int):
    from .tts import clear_progressive_tts, stop_speech

    total = len(st.session_state.pdf_chunks)
    st.session_state.current_chunk_idx = max(0, min(new_idx, total - 1))
    st.session_state.ai_commentary = ""
    st.session_state.ai_question = ""
    st.session_state.question_answered = False
    st.session_state.question_feedback = ""
    st.session_state.section_summary = ""
    stop_speech()
    st.session_state.ll_summary = ""
    st.session_state.ll_vocab = []
    st.session_state.ll_vocab_parse_fail = None


def reset_session():
    for key, default in fresh_defaults().items():
        st.session_state[key] = default
