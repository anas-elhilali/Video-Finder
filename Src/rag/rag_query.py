import sys
from langchain.prompts import PromptTemplate
import pprint
import os 
from functools import lru_cache
from langchain.memory import ConversationBufferMemory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "Src")
sys.path.insert(0, SRC_DIR)
_llm_cache = {}
_docsearch_cache = {}
_memory_cache = {}


@lru_cache(maxsize=1)
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
    Answer (include source):
    
    
    """
    PROMPT = PromptTemplate(
        template=template , 
        input_variables=["context", "question", "chat_history"]
    )
    return PROMPT

def get_memory(project_name):
    if project_name not in _memory_cache:
        _memory_cache[project_name] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer" 
        )
    return _memory_cache[project_name]

def _get_llm():
    if 'llm' not in _llm_cache:
        from langchain_community.llms.ollama import Ollama
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain.callbacks.manager import CallbackManager 
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        _llm_cache['llm'] = Ollama(model="gemma3:4b" , callbacks=callback_manager)
    return _llm_cache['llm']

def _load_docsearch(project_name):
    if project_name not in _docsearch_cache:
        import rag.create_store_embeddings as create_store_embeddings
        _docsearch_cache[project_name] = create_store_embeddings.load_faiss(project_name)
    return _docsearch_cache[project_name]
def build_retriever(PROMPT , project_name):
    from langchain.chains import ConversationalRetrievalChain
    
    llm = _get_llm()
    
    # llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , google_api_key="---" , streaming = True , callback_manager = callback_manager)
    docsearch =_load_docsearch(project_name)
    memory = get_memory(project_name)
    retriever = docsearch.as_retriever(search_kwargs ={"k" : 50})
    print(retriever)
    qa = ConversationalRetrievalChain.from_llm(
        llm = llm ,
        retriever = retriever , 
        memory=memory,
        chain_type = "stuff",
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=True
    )
    return qa , llm


# def build_tools(qa):
def rag_tool(qa , query: str , st_callbacks):
        result = qa.invoke({"question": query},
                            config={"callbacks": [st_callbacks]} if st_callbacks else {})
        pprint.pprint(result)
        answer = result["answer"]

        return answer

def run_rag(qa , query : str  , st_callbacks):
    # rag_tool = tools[0].func  
    final_answer = rag_tool(qa , query ,  st_callbacks=st_callbacks)
    return final_answer

def clear_cache():
    _llm_cache.clear()
    _docsearch_cache.clear()
    get_prompt.cache_clear()
    _memory_cache.clear()
