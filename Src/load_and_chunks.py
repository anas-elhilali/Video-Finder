from langchain_community.document_loaders import TextLoader 
import os
import re
import json


def chunking_docs(raw_docs  , file_names):
    processed_docs = []
    for idx , doc in enumerate(raw_docs):
        try:

            pattern = r"Scene (\d+) \((.*?)\):\s*(.*?)(?=\nScene \d+ \(|\Z)"
            matches = re.findall(pattern , doc , re.DOTALL)
            for scene_num , scene_timespan , description in matches:
                processed_docs.append({"scene_num" : f"Scene {scene_num}" , 
                                    "scene_timespan" : scene_timespan,
                                    "doc" : file_names[idx],
                                    "description" : description.strip()})
                
        except Exception as e:
                     print("Error" , e)
        with open('agentic/Data/processed/processed_docs.json' , 'w' , encoding = 'utf-8') as d:
            json.dump(processed_docs , d , ensure_ascii=False , indent=4)
    return processed_docs
def load_raw_docs(raw_folder_path):
    raw_documents = []
    file_names = []
    for raw_doc in os.listdir(raw_folder_path):
        raw_doc_path = os.path.join(raw_folder_path , raw_doc)
        file_names.append(raw_doc)
        loader = TextLoader(raw_doc_path , encoding="utf-8")
        raw_docs = loader.load()
        raw_documents.append(raw_docs[0].page_content)
    return raw_documents  ,file_names
if __name__ == "__main__":
    raw_folder_path = './agentic/Data/raw'
    raw_documents , file_names = load_raw_docs(raw_folder_path)
    processed_docs  = chunking_docs(raw_documents , file_names)
    print(processed_docs[:5])