import os
import re
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID") 
LOCATION = os.getenv("LOCATION")  
SERVICE_ACCOUNT_KEY_FILE = os.getenv("SERVICE_ACCOUNT_KEY_FILE")  
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


if __name__ == '__main__':
        try:
            Project_name = "cat"
            VIDEO_FOLDER =f"agentic/Video/{Project_name}"
            DESCRIPTION_FOLDER = f"./agentic/Data/Project/{Project_name}" 
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_KEY_FILE
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            gemini_model = GenerativeModel(GEMINI_MODEL)
            print(f"✅ Successfully initialized Gemini model: {GEMINI_MODEL}")
            if not os.path.isdir(VIDEO_FOLDER):
                print(f"❌ Error: Folder not found at '{VIDEO_FOLDER}'.")
            else:
                print(f"📂 Processing videos in folder: '{VIDEO_FOLDER}'")
                supported_extensions = ('.mp4', '.mov', '.avi', '.mkv' , '.webm')

                video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(supported_extensions)]
                total_videos = len(video_files)
                print(f"Found {total_videos} videos to process.")

                for i, filename in enumerate(video_files):
                    print(f"\n--- Processing video {i+1}/{total_videos}: {filename} ---")
                    video_to_analyze = os.path.join(VIDEO_FOLDER, filename)
                    analyze_and_create_description(video_to_analyze, gemini_model, DESCRIPTION_FOLDER)
                    
                    if i < total_videos - 1:
                        delay_seconds = 2  # Delay to respect API rate limits
                        print(f"⏳ Waiting for {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                
                print("\n✅ All videos processed.")

        except Exception as e:
            print(f"❌ A setup or authentication error occurred: {e}")
            print("Please ensure your Project ID, Location, and Key File are correct.")