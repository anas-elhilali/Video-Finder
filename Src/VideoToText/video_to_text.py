import os
import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"

ANALYSIS_PROMPT = """
You are an expert at describe videos for a specific use case , you will have to describe videos about cats , your job is to give a descibtion that will help another model generating scripts from the descriptions you give , the scripts are related to cat stories and the descriptions should contain enough information that descibe the video .
You will have to add timespams for each scenes , if there's one scene descibe it with it timespam if there's more descibe each scene whit it's timespam ,
avoid describing text overlay focus only on scenes ,
if you could make the description short enough while keep the details do it , give a consisten format here is an example
"Scene 1 (00:00–00:01): Close-up of a golden British Shorthair cat lying on a pink mat with red cherry patterns.
"""



def clean_filename(text: str) -> str:
    """Cleans a string to be a valid filename."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'[^a-zA-Z0-9_ ]', '', text)
    text = text.strip().lower().replace(" ", "_")
    text = re.sub(r'__+', '_', text)
    return text[:200]


def analyze_and_create_description(video_path: str, model: GenerativeModel, description_folder: str):
    

    print(f"🧠 Analyzing video file with Gemini ({GEMINI_MODEL})...")
    
    try:
        # Read the video file into memory
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        # Create a Part object from the video file data
        video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")

        # Combine the prompt and the video into a single request
        contents = [ANALYSIS_PROMPT, video_part]
        
        # Generate content using the model
        response = model.generate_content(contents)
        

        description = response.text
        print(f"💡 Model's Description: {description}")

        # --- Create Text File ---
        # Ensure description folder exists
        os.makedirs(description_folder, exist_ok=True)

        original_filename, _ = os.path.splitext(os.path.basename(video_path))
        txt_filename = f"{original_filename}_description.txt"
        txt_path = os.path.join(description_folder, txt_filename)

        # Check if the text file already exists
        counter = 1
        while os.path.exists(txt_path):
            txt_filename = f"{original_filename}_description_{counter}.txt"
            txt_path = os.path.join(description_folder, txt_filename)
            counter += 1

        # Write the description to the text file
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(description)
        
        print(f"✅ Description saved to: {txt_path}\n")

    except Exception as e:
        print(f"❌ An API error occurred: {e}\n")
