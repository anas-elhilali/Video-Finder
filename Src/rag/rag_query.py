import sys
from langchain_core.prompts import PromptTemplate
import pprint
import os 
from functools import lru_cache
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import  ConversationBufferMemory
from langchain_classic.callbacks.manager import CallbackManager
from langchain_classic.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

os.environ['HTTP_PROXY'] = 'http://10.8.32.11:8089'
os.environ['HTTPS_PROXY'] = 'http://10.8.32.11:8089'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)
gemini_api_key = os.getenv("GEMINI_API_KEYS")



def get_prompt():
    template = """
    You are an expert in finding scenes that describe a script your job is to help a contetn creator find scenes that support their script(story )
    =====previous conversation=====
    Previous conversation:
    {chat_history}
    =====instruction=====
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
    =====context=====
    {context}
    =====question=====
    question: look for scenes in this script , each part of this script give the source  "{question}"   
    Answer (include source it should be clickable with the timespan ):
    
    
    """
    PROMPT = PromptTemplate(
        template=template , 
        input_variables=["context", "question", "chat_history"]
    )
    return PROMPT




def _load_docsearch(project_name):
    import rag.create_store_embeddings as create_store_embeddings
    docsearch =  create_store_embeddings.load_faiss(project_name)
    return docsearch
def build_retriever(PROMPT , project_name): 
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])   
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash-lite",
        api_key = gemini_api_key, 
        streaming = True
    )
    memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer" 
        )
    docsearch =_load_docsearch(project_name)
    retriever = docsearch.as_retriever(search_kwargs ={"k" : 50})
    
    qa =    ConversationalRetrievalChain.from_llm(
        llm = llm ,
        retriever = retriever , 
        memory = memory , 
        chain_type = "stuff",
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )
    return qa , llm




def run_rag(qa , query : str ,st_callbacks ):
        callbacks = [StreamingStdOutCallbackHandler()]  # Console output
        if st_callbacks:
            callbacks.append(st_callbacks)
        result = qa.invoke({"question": query},
                            config={"callbacks": callbacks})
        pprint.pprint(result)
        answer = result["answer"]
        return answer


