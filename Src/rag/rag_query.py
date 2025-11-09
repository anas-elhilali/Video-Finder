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
    template = """Please provide an answer based solely on the provided sources. 
    When referencing information from a source, 
    cite the appropriate source(s) using their corresponding numbers. 
    Every answer should include at least one source citation. 
    Only cite a source when you are explicitly referencing it. 
    If none of the sources are helpful, you should indicate that. 
    You shouldn't add multiple source_ids in one single brackets , each source_id should be sperated inside a bracket.
    Query: When is water wet?\n
    Answer: Water will be wet when the sky is red [2], 
    which occurs in the evening [1].\n
    Now it's your turn. Below are several numbered sources of information:
    \n------\n"
    {context}"
    \n------\n
    Query: {question}\n
    "Answer: """
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

        result = qa.invoke({"question": query})
        answer = result["answer"]
        return answer

# if __name__ == "__main__":
#     prompt = get_prompt()
#     load_docsearch = _load_docsearch("kitty_milk")
#     qa , llm = build_retriever(prompt , "kitty_milk")
#     answer = run_rag(qa , "blue house")
#     print(answer)