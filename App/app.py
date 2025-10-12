import re
import streamlit as st
import sys
import os
import time
from langchain_community.callbacks.streamlit import (StreamlitCallbackHandler)
# Setup path for rag_query
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # agentic/
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)

import rag.rag_query as rg


# --- CSS Styling ---
st.markdown("""
<style>
.user{ background-color : #004f99 ; display : flex ; margin-left : 30% ; padding : 10px ; border-radius : 20px ; margin-bottom : 10px;}
.stButton{
    display: inline-block;
    padding-left : 20px ; 
    cursor:pointer;
    border-radius: 8px;
    width : 200px ;
    padding-top : 10px;
    padding-bottom : 10px; 
    font-size :18px ;
    border: none;           /* remove border */
        outline: none;          /* remove focus outline */
        box-shadow: none; 

}
.stbutton[data-baseweb="button"]{    
    background-color: rgba(255, 255, 255, 0.1);
}
.stElementContainer{
    postition: absolute;
}
</style>
""", unsafe_allow_html=True)

tools_list = ["RagFinder" , "Scraper" , "Describer" , "WriterGPT"]
select_tools = st.sidebar.selectbox("",tools_list)





# --- Setup RAG Agent ---


if select_tools == "RagFinder":
    new_chat_button = st.sidebar.button("New Chat", key="new_chat")

    st.sidebar.markdown("Chats" )   
    PROMPT = rg.get_prompt()
    qa, llm = rg.build_retriever(PROMPT)
    rag_tool = rg.build_tools(qa)

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

        pattern = r"Scene\s+\d+\s*\|\s*[\d:–]+\s*\|\s*([\w\d_]+\.txt)\s*→"
        # --- Generate bot response ---
        st_callback = StreamlitCallbackHandler(parent_container=chat["placeholder"])

        response = rg.run_rag(user_input , rag_tool,st_callback)

                # Replace placeholder with actual bot message
        chat["bot"] = f"{response}"
        chat["placeholder"].markdown(f"<div class='bot'>{response}</div>", unsafe_allow_html=True)
if select_tools == "Scraper":
    st.title("Scrap Youtube channel")

    channel_url = st.text_input("Add youtube channel URL")
    number_videos = st.number_input("number of videos " , format="%d" , step = 1 ,  max_value=600 , min_value=1)
    languages = ['ar' , 'en' , 'fr']
    lang = st.selectbox("Transcript language" , languages)
    scrap = st.button("scrap")
    if channel_url and scrap: 
        status = st.empty()
        status.text("Getting video IDs...")
        video_ids  , number_videos = yt.get_videoids(channel_url , number_videos)
        status.text(f"Found {len(video_ids)} videos!")

        channel_metadata = []
        for i, vid in enumerate(video_ids, start=1):
            status.text(f"Processing video {i}/{len(video_ids)}...")
            transcript = yt.get_transcript([vid], lang)
            channel_metadata.extend(transcript)
            time.sleep(0.1)
        df = pd.DataFrame(channel_metadata)
            
        st.success("Scraping done!")
        st.dataframe(df)  # Show table in Streamlit
            
            # Let user download CSV
        status.text("")
        csv = df.to_csv(index=False)
        pattern = r"\@([^\/]+)"
        match = re.search(pattern , channel_url)
        channel_name = "channel"
        if match:
            channel_name = match.group(1)
            print(channel_name)
        else:
            print("No match found!")
        st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{channel_name}_transcription.csv",
                mime="text/csv"
            )
if select_tools == "WriterGPT":
    new_chat_button = st.sidebar.button("New Chat", key="new_chat")

    st.sidebar.markdown("Chats" )   
    PROMPT = rg.get_prompt()
    qa, llm = rg.build_retriever(PROMPT)
    rag_tool = rg.build_tools(qa)

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

        pattern = r"Scene\s+\d+\s*\|\s*[\d:–]+\s*\|\s*([\w\d_]+\.txt)\s*→"
        # --- Generate bot response ---
        st_callback = StreamlitCallbackHandler(parent_container=chat["placeholder"])

        response = rg.run_rag(user_input , rag_tool,st_callback)

                # Replace placeholder with actual bot message
        chat["bot"] = f"{response}"
        chat["placeholder"].markdown(f"<div class='bot'>{response}</div>", unsafe_allow_html=True)