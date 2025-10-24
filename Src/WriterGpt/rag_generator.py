from google.ai.generativelanguage_v1beta.types import Tool as GenAITool
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
import os 
script_dir = os.path.dirname(os.path.abspath(__file__))

dotenv_path = os.path.join(script_dir, '..', '..', '.env')

load_dotenv(dotenv_path=dotenv_path)
os.environ['HTTP_PROXY'] = 'http://10.8.20.7:8089'
os.environ['HTTPS_PROXY'] = 'http://10.8.20.7:8089'
gemini_api_key = os.getenv("GEMINI_API_KEYS")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-latest", temperature=0)

resp = llm.invoke(
    "When is the next total solar eclipse in US?",
    tools=[GenAITool(google_search={})],
    api_key = gemini_api_key
)

print(resp.content)