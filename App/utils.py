import streamlit as st

def initialize_session_state():
    if "clicked_project" not in st.session_state:
        st.session_state.clicked_project = None
    if "show_create_modal" not in st.session_state:
        st.session_state.show_create_modal = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_tool" not in st.session_state:
        st.session_state.current_tool = "finder"
    if "project_processed" not in st.session_state:
        st.session_state.project_processed = False