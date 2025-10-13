import re
import streamlit as st
import sys
import os
import time
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_community.callbacks.streamlit import (StreamlitCallbackHandler)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Src.VideoToText import video_to_text as vtt
import pandas as pd
from Src.Scrap import scrap_channel_transcripts as yt
# Setup path for rag_query
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # agentic/
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)

import rag.rag_query as rg
# os.environ["HTTP_PROXY"] = "http://10.8.18.23:8089"
# os.environ["HTTPS_PROXY"] = "http://10.8.18.23:8089"


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
elif select_tools == "Scraper":
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
elif select_tools == "WriterGPT":
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
        
elif select_tools == "Describer":
    file_uploader = st.sidebar.file_uploader("Upload Videos : " , type=['mp4' , "mov" , "avi" , "wmv"] , accept_multiple_files=True)
    DESCRIPTION_FOLDER = "./agentic/Data/kitty_milk_descriptions"
    base_path = "./agentic/Video"
    list_Project = [
        item for item in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, item))
    ]
    
    list_Project =  list_Project    + ["➕ Create New Project"]
    project_name = st.sidebar.selectbox("Choose or Add A Project Name" , list_Project)
    if project_name == "➕ Create New Project":
        new_project = st.sidebar.text_input("Enter new project name:")
        if new_project:
            project_name = new_project
            os.makedirs(os.path.join(base_path, new_project), exist_ok=True)
            st.success(f"Project '{new_project}' created ✅")
    
    if file_uploader and project_name:
        videos_folder = os.path.join("./agentic/Video" , project_name)
        os.makedirs(videos_folder , exist_ok=True)
        analyze_button = st.sidebar.button("analyze")
        if analyze_button:
        # 1. Save Uploaded Files
            for video_files in file_uploader:
                file_path = os.path.join(videos_folder, video_files.name)
                with open(file_path, "wb") as f:
                    f.write(video_files.getbuffer())

            PROJECT_ID = os.getenv("PROJECT_ID") 
            LOCATION = os.getenv("LOCATION")  
            SERVICE_ACCOUNT_KEY_FILE = os.getenv("SERVICE_ACCOUNT_KEY_FILE")  
            GEMINI_MODEL = "gemini-2.5-flash"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE

            with st.spinner("🚀 Initializing Gemini model..."):
                vertexai.init(project=PROJECT_ID, location=LOCATION)
                gemini_model = GenerativeModel(GEMINI_MODEL)
            st.success(f"✅ Successfully initialized **{GEMINI_MODEL}**")

            if not os.path.isdir(videos_folder):
                st.error(f"❌ Error: Folder not found at **'{videos_folder}'**.")
            else:
                st.info(f"📂 Ready to process videos in: **'{videos_folder}'**")

                supported_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
                video_files = [f for f in os.listdir(videos_folder) if f.lower().endswith(supported_extensions)]
                total_videos = len(video_files)

                if total_videos == 0:
                    st.warning("⚠️ No supported videos found in this folder.")
                else:
                    st.success(f"🎥 Found **{total_videos}** video(s) to process.")

                    # Progress UI
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, filename in enumerate(video_files):
                        current_video = os.path.join(videos_folder, filename)
                        status_text.markdown(f"🛠️ **Processing video {i+1}/{total_videos}:** `{filename}`")

                        try:
                            vtt.analyze_and_create_description(current_video, gemini_model, DESCRIPTION_FOLDER)
                            st.toast(f"✅ Finished: {filename}", icon="🎯")
                        except Exception as e:
                            st.error(f"❌ Error processing `{filename}`: {e}")

                        # Update progress bar
                        progress = (i + 1) / total_videos
                        progress_bar.progress(progress)

                        # Wait between calls if needed
                        if i < total_videos - 1:
                            delay_seconds = 2
                            status_text.markdown(f"⏳ Waiting **{delay_seconds}s** before next video...")
                            time.sleep(delay_seconds)

                    status_text.markdown("🎉 All videos have been processed successfully!")
                    st.balloons()


            # except Exception as e:
            #     print(f"❌ A setup or authentication error occurred: {e}")
            #     print("Please ensure your Project ID, Location, and Key File are correct.")
    