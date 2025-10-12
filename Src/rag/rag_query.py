from langchain.agents import initialize_agent , Tool
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
import rag.create_store_embeddings as create_store_embeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pprint

def get_prompt():
    template = """
    You are an expert in finding scenes that describe a script your job is to help a contetn creator find scenes that support their script(story )
    you have to follow this instructions : 
    If the input script has multiple parts:
    . Collect all the results.
    . Synthesize into a final answer, citing sources.
    - Always include the document name(s)(source)  for any fact mentioned.
    - Include the source 
    - Always try to answer using the retrieved documents, even if the match isn’t perfect.
    - If exact events are missing, do your best to reconstruct the scene from related events.
    {context}
    question: look for scenes in this script , each part of this script give the source  "{question}"   
    Answer (include source):
    
    
    """
    PROMPT = PromptTemplate(
        template=template , 
        input_variables= ["context" , "question"]
    )
    return PROMPT


  
def build_retriever(PROMPT):
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = Ollama(model="gemma3:4b" , callbacks=callback_manager)
    # llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash" , google_api_key="---" , streaming = True , callback_manager = callback_manager)
    docsearch = create_store_embeddings.load_faiss()

    retriever = docsearch.as_retriever(search_kwargs ={"k" : 50})
    print(retriever)
    qa = RetrievalQA.from_chain_type(
        llm = llm ,
        retriever = retriever , 
        chain_type = "stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True

    )
    return qa , llm


def build_tools(qa):
    def rag_tool(query: str , st_callbacks):
        result = qa.invoke({"query": query},
                            config={"callbacks": [st_callbacks]} if st_callbacks else {})
        pprint.pprint(result)
        answer = result["result"]

        # sources = []
        # for doc in result["source_documents"]:
        #     metadata = doc.metadata
        #     scene = metadata.get("scene_num", "N/A")
        #     timespan = metadata.get("scene_timespan", "N/A")
        #     doc_name = metadata.get("doc", metadata.get("source", "Unknown"))
        #     description = metadata.get("description", "")

        #     sources.append(
        #         f"{scene} | {timespan} | {doc_name}\n→ {description}"
        #     )

        # sources_text = "\n\n".join(sources)

        # combined_result = (
        #     f"{answer}\n\n"
        #     f"📖 Sources:\n{sources_text}"
        # )

        # pprint.pprint(combined_result)

        return answer

    # tools = [
    #     Tool(
    #         name="RAG Retriever",
    #         func=rag_tool,
    #         description="Use this tool to answer questions based on the local documents."
    #     )
    # ]
    return rag_tool
# def run_agent(llm ,tools , query:str , st_callbacks):
#     agent = initialize_agent(
#         tools , 
#         llm , 
#         agent = "zero-shot-react-description",
#         verbose=True ,
#         max_iteration =3 , 
#         return_intermediate_steps=True,
#         handle_parsing_errors=True 
#     )


#     response = agent.invoke({"input": query}, config={"callbacks": [st_callbacks]}, return_intermediate_steps=True)
#     final_answer = response["output"]
#     if response["intermediate_steps"]:
#         _, retriever_text = response["intermediate_steps"][0]
#     else:
#         retriever_text = "No intermediate steps were produced."

#     sources = retriever_text.split("📖 Sources:")[-1].strip()
#     pprint.pprint(response)
#     return final_answer , sources

def run_rag(query : str , rag_tool , st_callbacks):
    # rag_tool = tools[0].func  
    final_answer = rag_tool(query ,  st_callbacks=st_callbacks)
    return final_answer
# if __name__ == "__main__":
#     PROMPT = get_prompt()
#     qa, llm = build_retriever(PROMPT)
#     tools = build_tools(qa)
#     rag_tool = tools[0].func  
#     query = "angry cats"
#     response  = run_rag(query, rag_tool)
#     print(response)

