import os , sys
import re 
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time
from dotenv import load_dotenv

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANALYSIS_PROMPT = """
You are an expert at describe videos for a specific use case , you will have to describe videos  , your job is to give a descibtion that will help another model generating scripts from the descriptions you give .
You will have to add timespams for each scenes , if there's one scene descibe it with it timespam if there's more descibe each scene whit it's timespam .
if you could make the description short enough while keep the details do it , give a consisten format here is an example
here's an example of the format you should return , always use the format ,it's mandatory : 
"Scene 1 (00:00–00:01): Close-up of a golden British Shorthair cat lying on a pink mat with red cherry patterns."
"""





def analyze_and_create_description(video_path: str, description_folder: str):
    from google import genai
    client = genai.Client(api_key = GEMINI_API_KEY )
    
    sys.stdout.reconfigure(encoding='utf-8')
    print(f"🧠 Analyzing video file with Gemini ({GEMINI_MODEL})...")
    
    try:
        video_file = client.files.upload(file=video_path)
       
        while video_file.state.name == "PROCESSING":
                time.sleep(2)  # Wait 5 seconds
                video_file = client.files.get(name=video_file.name)
        # Generate content using the model
        response    = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[ANALYSIS_PROMPT, video_file]
        )
        

        description = response.text
        print(f"💡 Model's Description: {description}")

        # --- Create Text File ---
        # Ensure description folder exists
        os.makedirs(description_folder, exist_ok=True)

        original_filename, extension = os.path.splitext(os.path.basename(video_path))
        extension = extension[1:] 
        txt_filename = f"{original_filename}_{extension}_description.txt"
        txt_path = os.path.join(description_folder, txt_filename)

        # Check if the text file already exists
        counter = 1
        while os.path.exists(txt_path):
            txt_filename = f"{original_filename}_{extension}_description.txt"
            txt_path = os.path.join(description_folder, txt_filename)
            counter += 1

        # Write the description to the text file
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write(description)
        
        print(f"✅ Description saved to: {txt_path}\n")

    except Exception as e:
        print(f"❌ An API error occurred: {e}\n")
