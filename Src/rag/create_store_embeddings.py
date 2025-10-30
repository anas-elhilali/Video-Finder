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
# vec_a = embeddings.embed_query(processed_doc[0]['description'])
# vec_b = embeddings.embed_query("kittens inside a box")

# def cosine_similarity(vec_a , vec_b):
#     return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
# similarity = cosine_similarity(vec_a , vec_b)
# print(similarity)
    
    docs = [
        Document(
            page_content=f"{scene['description']}\n source : {scene['video_path']} , timespan : {scene['scene_timespan']}",
            metadata={
                "scene_num": scene['scene_num'],
                "scene_timespan": scene['scene_timespan'],
                "doc": scene['doc'],
                "video_path" :scene['video_path']
            }
        )
        for scene in processed_doc
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