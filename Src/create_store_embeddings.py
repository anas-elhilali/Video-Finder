from langchain_huggingface import HuggingFaceEmbeddings
import json 
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"  , model_kwargs={"device": "cuda"})

def save_faiss():
    with open('agentic/Data/processed/processed_docs.json' , 'r' , encoding='utf-8') as f:
        processed_doc = json.load(f)
# vec_a = embeddings.embed_query(processed_doc[0]['description'])
# vec_b = embeddings.embed_query("kittens inside a box")

# def cosine_similarity(vec_a , vec_b):
#     return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
# similarity = cosine_similarity(vec_a , vec_b)
# print(similarity)
    docs = [
        Document(
            page_content=scene['description'],
            metadata={
                "scene_num": scene['scene_num'],
                "scene_timespan": scene['scene_timespan'],
                "doc": scene['doc']
            }
        )
        for scene in processed_doc
    ]
    docsearch = FAISS.from_documents(docs , embeddings) 
    docsearch.save_local("agentic/Data/faiss/faiss_index")

def load_faiss():
    docsearch = FAISS.load_local("agentic/Data/faiss/faiss_index" , embeddings ,  allow_dangerous_deserialization=True)

    query = "kitten inside a box"
    results = docsearch.similarity_search(query , k=2)
    return docsearch