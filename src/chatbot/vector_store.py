from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def create_vector_store(documents):
    """
    Creates a FAISS vector store from the given documents.
    """
    if not documents:
        raise ValueError("No documents provided to create the vector store.")

    # Get the OpenAI API key from the environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables.")

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    
    # Split the documents into chunks
    texts = text_splitter.split_documents(documents)

    # Initialize the OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, chunk_size=1000)

    # Process documents in batches
    batch_size = 1000
    
    # Create the vector store with the first batch
    vector_store = FAISS.from_documents(texts[:batch_size], embeddings)

    # Add the remaining documents in batches
    for i in range(batch_size, len(texts), batch_size):
        vector_store.add_documents(texts[i:i + batch_size])

    return vector_store
