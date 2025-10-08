# from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from langchain_community.llms.ollama import Ollama

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

load_dotenv()

# os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
# llm = ChatAnthropic(
#     model="claude-3-5-haiku-latest",
#     temperature=0,
#     max_tokens=1024,
#     timeout=None,
#     max_retries=2,
# )
# messages = [
#     (
#         "system",
#         "You are a helpful assistant that translates English to French. Translate the user sentence.",
#     ),
#     ("human", "I love programming."),
# ]
# ai_msg = llm.invoke(messages)

# print(ai_msg.content)

llm = Ollama(model="gemma3:4b")
response = llm.invoke("what's the capital o france")
print(response)
