import os
from langchain_huggingface import HuggingFaceEmbeddings
import json 
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")

def save_faiss(project_name):
    with open(f'Data/Projects/{project_name}/processed/processed_docs.json', 'r', encoding='utf-8') as f:
        processed_doc = json.load(f)
    
    docs = [
        Document(
            page_content=f"{scene['description']}\n source_id : {scene_id} \n source : {scene['video_path']} , timespan : {scene['scene_timespan']}",
            metadata={
                "scene" : scene , 
                "scene_num": scene['scene_num'],
                "scene_timespan": scene['scene_timespan'],
                "doc": scene['doc'],
                "video_path" :scene['video_path']
            }
        )
        for scene_id, scene in processed_doc.items()
    ]
    docsearch = FAISS.from_documents(docs , embeddings) 
    faiss_folder = f"Data/Projects/{project_name}/faiss"
    os.makedirs(faiss_folder, exist_ok=True)
    
    docsearch.save_local(f"{faiss_folder}/faiss_index")

def load_faiss(project_name):
    faiss_path = f"Data/Projects/{project_name}/faiss/faiss_index"
    
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at: {faiss_path}")
    
    docsearch = FAISS.load_local(
        faiss_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ FAISS index loaded from: {faiss_path}")
    return docsearch
if __name__ == "__main__":
    save_faiss("kitty_milk")