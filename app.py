import json
import os
import re
import streamlit as st
import sys
import time
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from Src.VideoToText import video_to_text as vtt
import pandas as pd
from dotenv import load_dotenv
from App import logic
load_dotenv()
from Src.rag import rag_query as rg
from Src.rag import create_store_embeddings as cse
from Src.rag import load_and_chunks as lc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)


from App import utils
st.set_page_config(
    page_title="Content Automation",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)



style_file = os.path.join("App" , "style.css")
with open(style_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

utils.initialize_session_state()
if st.session_state.show_create_modal:
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
        
        base_path = "Video"
        os.makedirs(base_path, exist_ok=True)
        gemini_key_env = None
        project_name = st.text_input("Project Name", placeholder="Enter project name...", key="create_project_name")

        if os.getenv("GEMINI_API_KEY"):
            gemini_key_env = os.getenv("GEMINI_API_KEY")
        else:
            gemini_key_env = st.text_input("Gemini Api Key" , placeholder="Enter Gemini Api key" , key="set_api_key" , type="password")
            

        uploaded_files = st.file_uploader("Upload Videos", type=['mp4', 'mov', 'avi', 'wmv'], accept_multiple_files=True, key="create_upload_videos")
        
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Create & Process", use_container_width=True, key="create_btn"):
                if project_name and gemini_key_env:
                    project_path = os.path.join(base_path, project_name)
                    os.makedirs(project_path, exist_ok=True)
                    with open(".env" , "a") as f:
                        f.write(f"GEMINI_API_KEY='{gemini_key_env}'\n")
                    if uploaded_files:
                        for video_file in uploaded_files:
                            file_path = os.path.join(project_path, video_file.name)
                            with open(file_path, "wb") as f:
                                f.write(video_file.getbuffer())
                    
                    
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
    base_path = "Video"
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
    
    with cols[0]:
        if st.button("➕\n\nCreate New\nProject", key="new_project", use_container_width=True):
            st.session_state.show_create_modal = True
            st.rerun()
    
    for idx, project_name in enumerate(list_projects):
        col_idx = (idx + 1) % 4
        col = cols[col_idx]
        
        date_str, file_count = logic.get_project_metadata(base_path, project_name)
        icon = logic.get_project_icon(project_name)
        
        with col:
            if st.button(
                f"{icon}\n\n{project_name}\n\n{date_str} • {file_count} files",
                key=project_name,
                on_click=logic.set_project,
                args=(project_name,),
                use_container_width=True
            ):
                pass
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    
    with st.sidebar:
        st.markdown(f"<h3 style='color: white;'>{logic.get_project_icon(st.session_state.clicked_project)} {st.session_state.clicked_project}</h3>", unsafe_allow_html=True)
        st.markdown("---")
        
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        
       
        
        st.markdown("---")
        
        if st.button("⬅️ Back to Projects", use_container_width=True):
            logic.clear_project()
            st.session_state.chat_history = [] 
            st.rerun()
    
    base_path = "Video"
    project_path = os.path.join(base_path, st.session_state.clicked_project)
    
    if not st.session_state.project_processed:
        st.markdown("<div style='text-align: center; padding: 60px 20px;'>", unsafe_allow_html=True)
        st.markdown("### 🎬 Processing Videos...")
        
        supported_extensions = ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        video_files = [f for f in os.listdir(project_path) if f.lower().endswith(supported_extensions)]
        
        if video_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                project_data_path = f"Data/Projects/{st.session_state.clicked_project}"
                RAW_FOLDER = f"{project_data_path}/raw"
                PROCESSED_FOLDER = f"{project_data_path}/processed"
                FAISS_FOLDER = f"{project_data_path}/faiss"
                
                os.makedirs(RAW_FOLDER, exist_ok=True)
                os.makedirs(PROCESSED_FOLDER, exist_ok=True)
                os.makedirs(FAISS_FOLDER, exist_ok=True)
                
                for i, filename in enumerate(video_files):
                    current_video = os.path.join(project_path, filename)
                    status_text.markdown(f"🛠️ **Processing:** `{filename}`")
                    
                    try:
                        vtt.analyze_and_create_description(current_video, RAW_FOLDER)
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    
                    progress = (i + 1) / len(video_files)
                    progress_bar.progress(progress)
                    
                    if i < len(video_files) - 1:
                        time.sleep(2)
                
                status_text.markdown("📊 **Indexing videos...**")
                progress_bar.progress(0)
                
                raw_folder_path = RAW_FOLDER  
                raw_documents, file_names = lc.load_raw_docs(raw_folder_path)
                
                progress_bar.progress(0.33)
                status_text.markdown("📝 **Chunking documents...**")
                
                processed_docs = lc.chunking_docs(raw_documents, file_names, st.session_state.clicked_project)
                
                progress_bar.progress(0.66)
                status_text.markdown("🔍 **Creating FAISS index...**")
                
                cse.save_faiss(st.session_state.clicked_project)
                
                progress_bar.progress(1.0)
                status_text.markdown("✅ **Indexing complete!**")
                
                st.session_state.project_processed = True
                st.success("✅ Videos processed and indexed successfully!")
                time.sleep(1)
                st.rerun()    
            except Exception as e:
                st.error(f"❌ Processing error: {e}")
            
        else:
            st.warning("⚠️ No videos found in this project")
        
    else:
        if st.session_state.current_tool == "finder":
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            if len(st.session_state.chat_history) == 0:
                st.markdown("""
                <div style='display: flex; flex-direction: column; align-items: center; justify-content: center; 
                            height: 70vh; text-align: center;'>
                    <h1 style='color: rgba(255, 255, 255, 0.8); font-size: 48px; margin-bottom: 20px;'>
                        🔍 ClipFinder
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
                        st.markdown(chat["bot"] , unsafe_allow_html=True)
            
            if user_input := st.chat_input("Search for a clip or ask a question..."):
                st.session_state.chat_history.append({"user": user_input, "bot": ""})
                
                with st.chat_message("user"):
                    st.write(user_input)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Searching..."):
                        try:
                            PROMPT = rg.get_prompt()
                            qa, llm = rg.build_retriever(PROMPT, st.session_state.clicked_project)
                            response = rg.run_rag(qa, user_input)

                            
                                
                            def create_link(video_path ,time_span ,  video_id):
                                return f'<a href="http://localhost:8000/{video_path}" data-timespan="{time_span}">[{video_id}]</a>'
                            
                            pattern = r"\[(\d+)\]"
                            matches = re.findall(pattern , response)

                            base_data_folder = "Data/Projects"
                            json_file = f"Data/Projects/{st.session_state.clicked_project}\processed\processed_docs.json"
                            with open(json_file , "r" , encoding="utf-8") as f:
                                data = json.load(f)  
                            for key , key_values in data.items():
                                for match_id in matches:
                                    if match_id == key:
                                        response = response.replace(f'[{match_id}]' , create_link(key_values['video_path'] , key_values['scene_timespan'] , match_id))
                            response = re.sub(r"(</a>\s*\.)", r"\1<br>", response)
                            st.session_state.chat_history[-1]["bot"] = response 
                            st.markdown(response , unsafe_allow_html=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
        
        