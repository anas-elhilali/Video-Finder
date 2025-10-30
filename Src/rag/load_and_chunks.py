from langchain_community.document_loaders import TextLoader 
import os
import re
import json


def chunking_docs(raw_docs  , file_names ,project_name):
    processed_docs = []
    base_path = "Video"
    for idx , doc in enumerate(raw_docs):
        video_name = re.sub(r"_description\.txt$", ".webm", file_names[idx])
        video_path = os.path.join(base_path, project_name, video_name).replace("\\", "/")

        try:

            pattern = r"Scene (\d+) \((.*?)\):\s*(.*?)(?=\nScene \d+ \(|\Z)"
            matches = re.findall(pattern , doc , re.DOTALL)
            for scene_num , scene_timespan , description in matches:
                processed_docs.append({"scene_num" : f"Scene {scene_num}" , 
                                    "scene_timespan" : scene_timespan,
                                    "doc" : file_names[idx],
                                    "description" : description.strip(),
                                    "video_path" : video_path.strip()
                                    })
                
        except Exception as e:
                     print("Error" , e)
        processed_folder = f'Data/Projects/{project_name}/processed'
        os.makedirs(processed_folder, exist_ok=True)   
        with open(f'{processed_folder}/processed_docs.json', 'w', encoding='utf-8') as d:
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
    project_name ="kitty_milk"
    raw_folder_path = f'Data/Projects/{project_name}/raw'
    raw_documents , file_names = load_raw_docs(raw_folder_path)
    processed_docs  = chunking_docs(raw_documents , file_names , project_name)
