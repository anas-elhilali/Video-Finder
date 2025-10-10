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

os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

load_dotenv()
template = """
    You are an expert in finding scenes that describe a script your job is to help a contetn creator find scenes that support their script(story )
    you have to follow this instructions : 
    If the input script has multiple parts:
    1. Break it into smaller sub-queries.
    2. Use the 'RAG Retriever' tool for each sub-query separately.
    3. Collect all the results.
    4. Synthesize into a final answer, citing sources.
    5- Always include the document name(s)(source)  for any fact mentioned.
    6- Include the source 
    7- Always try to answer using the retrieved documents, even if the match isn’t perfect.
    8- If exact events are missing, do your best to reconstruct the scene from related events.
    {context}
    question: look for scenes in this script , each part of this script give the source  "{question}"   
    Answer (include source):
    
    
    """
PROMPT = PromptTemplate(
        template=template , 
        input_variables= ["context" , "question"]
    ) 
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

doc_search = create_store_embeddings.load_faiss()
# llm = Ollama(model="gemma3:4b" , callbacks=callback_manager)
llm = Ollama(model="gemma3:4b" )

retriever = doc_search.as_retriever(search_kwargs = {"k" : 5})

qa = RetrievalQA.from_chain_type(
        llm = llm ,
        retriever = retriever , 
        chain_type = "stuff",
        chain_type_kwargs={"prompt": PROMPT},
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
query= "angry cat"
# rag_tool = tools[0].func
final_answer = rag_tool(query )

print(final_answer)
