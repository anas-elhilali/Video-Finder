# from langchain_anthropic import ChatAnthropic
import pprint
from dotenv import load_dotenv
import os
from langchain_community.llms.ollama import Ollama
from ..rag import create_store_embeddings 
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
import pandas as pd
from langchain.globals import set_debug
from langchain_google_genai import ChatGoogleGenerativeAI

set_debug(True)
script_dir = os.path.dirname(os.path.abspath(__file__))

dotenv_path = os.path.join(script_dir, '..', '..', '.env')

load_dotenv(dotenv_path=dotenv_path)

def load_and_clean_csv():
    csv_example = pd.read_csv("agentic/Data/Mrpeach-i4t_transcription.csv")
    csv_example = csv_example[csv_example["view_count"]>8000000].reset_index(drop=True)
    example = "\n ---another example :".join(csv_example["transcript"].astype(str))
    return example

example = load_and_clean_csv()

    
template = """
    You are an expert in generating scripts your job is to help a content creator is to generate an engaging scripts that are simialir to the provided examples . 
    from the provided docs you have to create a story from the question you are being asked 
    here's example to get inspired by  
    {example}
    create a story based on those docs chose create an engaging story 

    {context}
    question: {question}       
    
    """
PROMPT = PromptTemplate(
        template=template , 
        input_variables=["context", "question", "example"]
    ) 
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

doc_search = create_store_embeddings.load_faiss()
# llm = Ollama(model="gemma3:4b" , callbacks=callback_manager)
llm = Ollama(model="gemma3:4b" )
# llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , google_api_key=os.getenv("GEMINI_API_KEYS")  )

retriever = doc_search.as_retriever(search_kwargs = {"k" : 5})
prompt_with_examples = PROMPT.partial(example=example)
qa = RetrievalQA.from_chain_type(
        llm = llm ,
        retriever = retriever , 
        chain_type = "stuff",
        chain_type_kwargs={"prompt": prompt_with_examples},
        return_source_documents=True

    )

print(retriever)
def rag_tool(query: str ):
        result = qa.invoke({"query": query})
        pprint.pprint(result)
        answer = result["result"]
        return answer


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
query= "cat that being left out"
# rag_tool = tools[0].func
final_answer = rag_tool(query )

print(final_answer)
