from langchain.agents import initialize_agent , Tool
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
import create_store_embeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager


def get_prompt():
    template = """
    Use the following documents to answer the question . 
    {context}
    Question: {question}
    Instructions:
    - Always include the document name(s)(source)  for any fact mentioned.
    - Include the source 
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

    retriever = docsearch.as_retriever(search_kwargs ={"k" : 10})
   
    print(retriever)
    qa = RetrievalQA.from_chain_type(
        llm = llm ,
        retriever = retriever , 
        chain_type = "stuff",
        chain_type_kwargs = {"prompt" : PROMPT},
        return_source_documents=True
    )
    return qa , llm


def build_tools(qa):
    def rag_tool(query: str):
        result = qa({"query": query})
        answer = result["result"]

        sources = []
        for doc in result["source_documents"]:
            metadata = doc.metadata
            scene = metadata.get("scene_num", "N/A")
            timespan = metadata.get("scene_timespan", "N/A")
            doc_name = metadata.get("doc", metadata.get("source", "Unknown"))
            description = metadata.get("description", "")
            sources.append(f"{scene} | {timespan} | {doc_name}\n→ {description}")

        sources_text = "\n\n".join(sources)
        # Combine answer and sources into the result
        combined_result = f"{answer}\n\n📖 Sources:\n{sources_text}"
        print(combined_result)  # For debugging
        return combined_result

    tools = [
        Tool(
            name="RAG Retriever",
            func=rag_tool,
            description="Use this tool to answer questions based on the local documents."
        )
    ]
    return tools
def run_agent(llm ,tools , query:str  , st_callbacks):
    agent = initialize_agent(
        tools , 
        llm , 
        agent = "zero-shot-react-description",
        max_iterations = 3,
        verbose=True ,
        return_intermediate_steps=True
    )


    response = agent.invoke({"input": query}, config={"callbacks": [st_callbacks]}, return_intermediate_steps=True)
    final_answer = response["output"]
    _ , retriever_text = response["intermediate_steps"][0]
    sources = retriever_text.split("📖 Sources:")[-1].strip()
    return final_answer , sources  
# if __name__ == "__main__":
#     PROMPT = get_prompt()
#     qa, llm = build_retriever(PROMPT)
#     tools = build_tools(qa)
#     query = "angry cats"
#     response , sources = run_agent(llm, tools, query)
#     print(response)


# {'input': 'light blue pet house', 
#  'output': 'The light blue pet house is shown in Scene 2, between 0:15-0:20. It’s a close-up view of the entrance to the large blue cat bed, revealing its shiny, insulated interior and the misty atmosphere. Two golden British Shorthair cats stand nearby, gazing intently at it.', 
#  'intermediate_steps': 
#      [(AgentAction(tool='RAG Retriever', 
#                    tool_input='query: "light blue pet house', 
#                    log='I need to find information about a "light blue pet house" using the RAG Retriever tool.\nAction: RAG Retriever\nAction Input: query: "light blue pet house"'), 
#        'The light blue pet house is shown in **Scene 2**. The scene takes place around **0:15-0:20** in the video. It’s a close-up view of the entrance to the large blue cat bed, revealing its shiny, insulated interior and the misty atmosphere. Two golden British Shorthair cats stand nearby, gazing intently at it.\n\n📖 Sources:\nScene 1 | 00:00–00:02 | 70_description.txt\n→ \n\nScene 7 | 00:14-00:16 | 223_description.txt\n→ \n\nScene 14 | 00:46–00:49 | 372_description.txt\n→ ')]}
