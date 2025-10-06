from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def create_chatbot(vector_store):
    """
    Creates a chatbot using the given vector store.
    """
    if not vector_store:
        raise ValueError("No vector store provided to create the chatbot.")

    # Get the OpenAI API key from the environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Initialize the ChatOpenAI model
    llm = ChatOpenAI(
        openai_api_key=api_key,
        model_name='gpt-4.1-2025-04-14',
        temperature=0
    )

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True
    )

    return qa_chain

def ask_question(qa_chain, query):
    """
    Asks a question to the chatbot and returns the answer.
    """
    if not qa_chain:
        raise ValueError("No qa_chain provided to ask a question.")
        
    if not query:
        raise ValueError("No query provided to ask a question.")

    response = qa_chain.invoke({"query": query})
    return response["result"]
