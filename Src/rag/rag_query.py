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
from dotenv import load_dotenv



script_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(script_dir,"..", '..', '.env')
load_dotenv(dotenv_path=dotenv_path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
gemini_api_key = os.getenv("GEMINI_API_KEY")



def get_prompt():
    template = """
    You are an expert in finding scenes that describe a script your job is to help a content creator find scenes that support their script(story )
    =====previous conversation=====
    Previous conversation:
    {chat_history}
    =====instruction=====
    you have to follow this instructions : 
    If the input script has multiple parts:

    - Synthesize into a final answer, citing sources.
    - Always include the document name(s)(source)  for any fact mentioned.
    - Include the source 
    
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
    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])   
    llm = ChatGoogleGenerativeAI(
        model = "gemini-2.5-flash",
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




def run_rag(qa , query : str  ):
        # callbacks = [StreamingStdOutCallbackHandler()]  # Console output
        # if st_callbacks:
        #     callbacks.append(st_callbacks)
        result = qa.invoke({"question": query})
        pprint.pprint(result)
        answer = result["answer"]
        return answer

if __name__ == "__main__":
    prompt = get_prompt()
    load_docsearch = _load_docsearch("kitty_milk")
    qa , llm = build_retriever(prompt , "kitty_milk")
    answer = run_rag(qa , "blue house")