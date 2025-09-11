from langchain.agents import initialize_agent , Tool
from langchain_community.llms.ollama import Ollama
from langchain.chains import RetrievalQA
import create_store_embeddings
from langchain.prompts import PromptTemplate

template = """
Use the following documents to answer the question. Include the doc ,the scene number and timespan in your answer.

{context}

Question: {question}

Answer (include scene info):
"""
PROMPT = PromptTemplate(
    template=template , 
    input_variables= ["context" , "question"]
)

    

llm = Ollama(model="gemma3:4b")

docsearch = create_store_embeddings.load_faiss()

retriever = docsearch.as_retriever(search_kwargs ={"k" : 3})
qa = RetrievalQA.from_chain_type(
    llm = llm ,
    retriever = retriever , 
    chain_type = "stuff",
    chain_type_kwargs = {"prompt" : PROMPT}
)

def rag_tool(query : str):
    result = qa.run(query)
    return result 
tools = [
    Tool(
        name="RAG Retriever",
        func=rag_tool,
        description="Use this tool to answer questions based on the local documents."
    )
]
agent = initialize_agent(
    tools , 
    llm , 
    agent = "zero-shot-react-description",
    max_iterations = 3,
    verbose = True
)


query = "owner bring a blue light pet house"

response = agent.run(query)

print("answer" , response)