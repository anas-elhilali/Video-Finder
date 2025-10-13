import os

import re
import streamlit as st
import sys
import time
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Src.VideoToText import video_to_text as vtt
import pandas as pd
from Src.Scrap import scrap_channel_transcripts as yt
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)

import rag.rag_query as rg

st.set_page_config(
    page_title="Content Automation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache expensive imports and initialization
@st.cache_resource
def init_vertexai():
    """Initialize Vertex AI once"""
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION")
    SERVICE_ACCOUNT_KEY_FILE = os.getenv("SERVICE_ACCOUNT_KEY_FILE")
    
    if PROJECT_ID and LOCATION and SERVICE_ACCOUNT_KEY_FILE:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        return GenerativeModel("gemini-2.5-flash")
    return None

st.markdown("""
<style>
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Project Cards */
.cards-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 20px;
    width: 100%;
    padding: 20px 0;
}

.cards-container button {
    background: rgba(30, 42, 58, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    color: white !important;
    transition: all 0.3s ease !important;
    height: auto !important;
    min-height: 180px !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    text-align: center !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 12px !important;
    font-weight: 500 !important;
}

.cards-container button:hover {
    background: rgba(42, 63, 95, 0.8) !important;
    border-color: rgba(255, 255, 255, 0.25) !important;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2) !important;
}

.cards-container button:first-of-type {
    background: rgba(30, 42, 58, 0.4) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.cards-container button:first-of-type:hover {
    background: rgba(42, 63, 95, 0.6) !important;
    border-color: rgba(255, 255, 255, 0.2) !important;
}

/* Modal Styles */
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
}

.modal-content {
    background: linear-gradient(135deg, #1a2332 0%, #232e3f 100%);
    border-radius: 20px;
    padding: 40px;
    max-width: 600px;
    width: 90%;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-header {
    font-size: 28px;
    font-weight: 700;
    color: white;
    margin-bottom: 30px;
}

.form-group {
    margin-bottom: 20px;
}

.form-label {
    display: block;
    color: rgba(255, 255, 255, 0.8);
    margin-bottom: 8px;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.form-input {
    width: 100%;
    padding: 12px 16px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: white;
    font-size: 14px;
    transition: all 0.3s ease;
}

.form-input:focus {
    background: rgba(255, 255, 255, 0.1);
    border-color: rgba(100, 200, 255, 0.5);
    outline: none;
}

.modal-buttons {
    display: flex;
    gap: 12px;
    margin-top: 30px;
}

.btn-primary {
    flex: 1;
    padding: 12px 24px;
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    border: none;
    border-radius: 8px;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(0, 212, 255, 0.3);
}

.btn-secondary {
    flex: 1;
    padding: 12px 24px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    color: white;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.15);
}

/* Chat Interface */
.chat-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
}

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px;
}

.message {
    margin-bottom: 16px;
    display: flex;
    gap: 12px;
}

.message.user {
    justify-content: flex-end;
}

.message.bot {
    justify-content: flex-start;
}

.message-bubble {
    max-width: 70%;
    padding: 12px 16px;
    border-radius: 12px;
    word-wrap: break-word;
}

.message.user .message-bubble {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    color: white;
}

.message.bot .message-bubble {
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.chat-input-area {
    padding: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    background: linear-gradient(135deg, #1a2332 0%, #232e3f 100%);
}

.tool-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}

.tool-tab {
    padding: 8px 16px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    color: rgba(255, 255, 255, 0.7);
    cursor: pointer;
    transition: all 0.3s ease;
    font-size: 12px;
    font-weight: 600;
}

.tool-tab.active {
    background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
    color: white;
    border-color: transparent;
}

.tool-tab:hover {
    background: rgba(100, 200, 255, 0.2);
    border-color: rgba(100, 200, 255, 0.5);
}

/* Progress Bar */
.progress-container {
    padding: 20px;
    background: rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    margin: 16px 0;
}

.progress-bar {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 8px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d4ff 0%, #0099cc 100%);
    border-radius: 3px;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

def set_project(project_name):
    st.session_state.clicked_project = project_name
    st.session_state.current_tool = "finder"
    st.session_state.project_processed = True

def clear_project():
    st.session_state.clicked_project = None
    st.session_state.current_tool = None

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

initialize_session_state()

def get_project_icon(project_name):
    icons = {
        "ai": "🤖", "data": "📊", "video": "🎬", "chat": "💬",
        "rag": "🔍", "ml": "🧠", "nlp": "📝",
    }
    for key, icon in icons.items():
        if key.lower() in project_name.lower():
            return icon
    return "📁"

def get_project_metadata(base_path, project_name):
    project_path = os.path.join(base_path, project_name)
    try:
        mod_time = os.path.getmtime(project_path)
        date_str = datetime.fromtimestamp(mod_time).strftime("%b %d, %Y")
        file_count = len([f for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f))])
        return date_str, file_count
    except:
        return "Unknown", 0

# === MAIN APP FLOW ===

if st.session_state.show_create_modal:
    # CREATE PROJECT PAGE (FULL PAGE)
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .main {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 1.5, 1])
    
    with col_center:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a2332 0%, #232e3f 100%);
                    border-radius: 20px;
                    padding: 50px;
                    border: 1px solid rgba(255, 255, 255, 0.1);">
            <h2 style="color: white; text-align: center; margin-bottom: 30px;">🎬 New Project</h2>
        """, unsafe_allow_html=True)
        
        base_path = "./agentic/Video"
        os.makedirs(base_path, exist_ok=True)
        
        project_name = st.text_input("Project Name", placeholder="Enter project name...", key="create_project_name")
        scrape_url = st.text_input("YouTube URL (Optional)", placeholder="youtube.com/...", key="create_youtube_url")
        uploaded_files = st.file_uploader("Upload Videos", type=['mp4', 'mov', 'avi', 'wmv'], accept_multiple_files=True, key="create_upload_videos")
        
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Create & Process", use_container_width=True, key="create_btn"):
                if project_name:
                    project_path = os.path.join(base_path, project_name)
                    os.makedirs(project_path, exist_ok=True)
                    
                    if uploaded_files:
                        for video_file in uploaded_files:
                            file_path = os.path.join(project_path, video_file.name)
                            with open(file_path, "wb") as f:
                                f.write(video_file.getbuffer())
                    
                    if scrape_url:
                        st.info("🔄 Scraping YouTube videos...")
                    
                    st.session_state.clicked_project = project_name
                    st.session_state.show_create_modal = False
                    st.session_state.project_processed = False
                    st.rerun()
                else:
                    st.error("❌ Please enter a project name")
        
        with col_btn2:
            if st.button("Cancel", use_container_width=True, key="cancel_btn"):
                st.session_state.show_create_modal = False
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.clicked_project is None:
    # PROJECT SELECTION VIEW
    base_path = "./agentic/Video"
    os.makedirs(base_path, exist_ok=True)
    
    list_projects_path = os.listdir(base_path)
    list_projects = [p for p in list_projects_path if os.path.isdir(os.path.join(base_path, p))]
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("## 🎬")
    with col2:
        st.markdown("## Recent Notebooks")
    
    st.markdown('<div class="cards-container">', unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    # Create new project button
    with cols[0]:
        if st.button("➕\n\nCreate New\nProject", key="new_project", use_container_width=True):
            st.session_state.show_create_modal = True
            st.rerun()
    
    # Project cards
    for idx, project_name in enumerate(list_projects):
        col_idx = (idx + 1) % 4
        col = cols[col_idx]
        
        date_str, file_count = get_project_metadata(base_path, project_name)
        icon = get_project_icon(project_name)
        
        with col:
            if st.button(
                f"{icon}\n\n{project_name}\n\n{date_str} • {file_count} files",
                key=project_name,
                on_click=set_project,
                args=(project_name,),
                use_container_width=True
            ):
                pass
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # PROJECT VIEW WITH CHAT (ChatGPT-style layout)
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"<h3 style='color: white;'>{get_project_icon(st.session_state.clicked_project)} {st.session_state.clicked_project}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("**Tools**")
        if st.button("🔍 RAG Finder", key="tool_finder", use_container_width=True):
            st.session_state.current_tool = "finder"
            st.session_state.chat_history = []
            st.rerun()
        
        if st.button("✍️ Writer GPT", key="tool_writer", use_container_width=True):
            st.session_state.current_tool = "writer"
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        if st.button("⬅️ Back to Projects", use_container_width=True):
            clear_project()
            st.rerun()
    
    # Main content area
    base_path = "./agentic/Video"
    project_path = os.path.join(base_path, st.session_state.clicked_project)
    
    # Process videos if not done yet
    if not st.session_state.project_processed:
        st.markdown("<div style='text-align: center; padding: 60px 20px;'>", unsafe_allow_html=True)
        st.markdown("### 🎬 Processing Videos...")
        
        supported_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        video_files = [f for f in os.listdir(project_path) if f.lower().endswith(supported_extensions)]
        
        if video_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                PROJECT_ID = os.getenv("PROJECT_ID")
                LOCATION = os.getenv("LOCATION")
                SERVICE_ACCOUNT_KEY_FILE = os.getenv("SERVICE_ACCOUNT_KEY_FILE")
                GEMINI_MODEL = "gemini-2.5-flash"
                
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE
                vertexai.init(project=PROJECT_ID, location=LOCATION)
                gemini_model = GenerativeModel(GEMINI_MODEL)
                
                DESCRIPTION_FOLDER = "./agentic/Data/descriptions"
                os.makedirs(DESCRIPTION_FOLDER, exist_ok=True)
                
                for i, filename in enumerate(video_files):
                    current_video = os.path.join(project_path, filename)
                    status_text.markdown(f"🛠️ **Processing:** `{filename}`")
                    
                    try:
                        vtt.analyze_and_create_description(current_video, gemini_model, DESCRIPTION_FOLDER)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    
                    progress = (i + 1) / len(video_files)
                    progress_bar.progress(progress)
                    
                    if i < len(video_files) - 1:
                        time.sleep(2)
                
                st.session_state.project_processed = True
                st.success("✅ Videos processed successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
        else:
            st.warning("⚠️ No videos found in this project")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    else:
        # CHAT INTERFACE
        if st.session_state.current_tool == "finder":
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Display chat history centered
            if len(st.session_state.chat_history) == 0:
                st.markdown("""
                <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; 
                            height: 70vh; text-align: center;'>
                    <h1 style='color: rgba(255, 255, 255, 0.8); font-size: 48px; margin-bottom: 20px;'>
                        🔍 RAG Finder
                    </h1>
                    <p style='color: rgba(255, 255, 255, 0.6); font-size: 18px;'>
                        Search and find clips from your videos
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for i, chat in enumerate(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.write(chat["user"])
                    with st.chat_message("assistant"):
                        st.write(chat["bot"])
            
            # Chat input at bottom
            if user_input := st.chat_input("Search for a clip or ask a question..."):
                st.session_state.chat_history.append({"user": user_input, "bot": ""})
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Searching..."):
                        try:
                            # Lazy load RAG components only when needed
                            PROMPT = rg.get_prompt()
                            qa, llm = rg.build_retriever(PROMPT)
                            
                            st_callback = StreamlitCallbackHandler(parent_container=st.container())
                            response = rg.run_rag(qa , user_input,  st_callback)
                            
                            st.session_state.chat_history[-1]["bot"] = response
                            st.write(response)
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        elif st.session_state.current_tool == "writer":
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            if len(st.session_state.chat_history) == 0:
                st.markdown("""
                <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; 
                            height: 70vh; text-align: center;'>
                    <h1 style='color: rgba(255, 255, 255, 0.8); font-size: 48px; margin-bottom: 20px;'>
                        ✍️ Writer GPT
                    </h1>
                    <p style='color: rgba(255, 255, 255, 0.6); font-size: 18px;'>
                        Generate descriptions, summaries, or content based on your videos
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for i, chat in enumerate(st.session_state.chat_history):
                    with st.chat_message("user"):
                        st.write(chat["user"])
                    with st.chat_message("assistant"):
                        st.write(chat["bot"])
            
            if user_input := st.chat_input("Ask Writer GPT to generate content..."):
                st.session_state.chat_history.append({"user": user_input, "bot": ""})
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.chat_message("assistant"):
                    st.info("💡 Writer GPT feature coming soon...")