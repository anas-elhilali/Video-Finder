import re
import streamlit as st
import sys
import os
import time

# Setup path for rag_query
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # agentic/
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)

import rag_query as rg

# --- CSS Styling ---
st.markdown("""
<style>
.user{ background-color : #004f99 ; display : flex ; margin-left : 30% ; padding : 10px ; border-radius : 20px ; margin-bottom : 10px;}

</style>
""", unsafe_allow_html=True)

# --- Setup RAG Agent ---
PROMPT = rg.get_prompt()
qa, llm = rg.build_retriever(PROMPT)
tools = rg.build_tools(qa)

st.title("Agentic RAG")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Chat Input ---
user_input = st.chat_input("search for a clip")

if user_input:
    # Append user message immediately
    st.session_state.chat_history.append({"user": user_input, "bot": ""})

    # Display all messages so far
    chat_container = st.container()
    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='user'>{chat['user']}</div>", unsafe_allow_html=True)
        if chat["bot"]:
            st.markdown(f"<div class='bot'>{chat['bot']}</div>", unsafe_allow_html=True)
        else:
            # placeholder for bot typing
            chat["placeholder"] = st.empty()

    # Show "Bot is typing..."
    chat["placeholder"].markdown("<div class='bot'>Thinking...</div>", unsafe_allow_html=True)
    pattern = r"Scene\s+\d+\s*\|\s*[\d:–]+\s*\|\s*([\w\d_]+\.txt)\s*→"
    # --- Generate bot response ---
    response , source = rg.run_agent(llm, tools, user_input)
    match = re.match(pattern , source)
    doc_name = match.group(1)
            # Replace placeholder with actual bot message
    chat["bot"] = f"{response}'\n'{doc_name}"
    chat["placeholder"].markdown(f"<div class='bot'>{response}'\n'{doc_name}</div>", unsafe_allow_html=True)
