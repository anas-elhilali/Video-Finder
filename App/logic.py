from datetime import datetime
import os 
import streamlit as st
from vertexai.generative_models import GenerativeModel, Part
import vertexai

def set_project(project_name):
    st.session_state.clicked_project = project_name
    st.session_state.current_tool = "finder"
    st.session_state.project_processed = True

def clear_project():
    st.session_state.clicked_project = None
    st.session_state.current_tool = None

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

def init_vertexai():
    """Initialize Vertex AI once"""
    PROJECT_ID = os.getenv("PROJECT_ID")
    LOCATION = os.getenv("LOCATION")
    SERVICE_ACCOUNT_KEY_FILE = "inductive-gift-465720-v6-c7ccb4800921.json"
    
    if PROJECT_ID and LOCATION and SERVICE_ACCOUNT_KEY_FILE:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        return GenerativeModel("gemini-2.5-flash")
    return None