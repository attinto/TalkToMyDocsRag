
"""
Basic RAG Pipeline
------------------

This module implements a straightforward Retrieval-Augmented Generation (RAG) pipeline using LangChain.

**Key Components:**
1.  **DocumentLoader**: Loads the source text from a file.
2.  **TextSplitter**: Splits the document into smaller, manageable chunks.
3.  **Embeddings**: Converts text chunks into numerical vectors using an OpenAI model.
4.  **VectorStore**: Stores the vectors and allows for efficient similarity searches (FAISS).
5.  **Retriever**: Fetches the most relevant chunks for a given question.
6.  **LLM**: An OpenAI chat model (e.g., GPT-3.5-turbo) that generates the answer.
7.  **LCEL Chain**: The LangChain Expression Language chain that orchestrates the flow from question to answer.

**SOLID Principles:**
-   **Single Responsibility Principle**: The `BasicRAGPipeline` class is solely responsible for constructing and executing this specific RAG strategy. Methods within the class are broken down by function (loading, creating vector store, creating the chain).
-   **Open/Closed Principle**: The `execute_rag` function acts as a clean entry point. While the internal logic of `BasicRAGPipeline` can be modified, its public contract (`invoke`) is stable. The overall system is open to extension by adding new RAG files, without modifying this one.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .rag_config import GENERAL_CONFIG, BASIC_RAG_CONFIG


class BasicRAGPipeline:
    """
    Encapsulates the logic for a basic RAG pipeline.
    This class is designed to be self-contained, handling everything from
    document loading to answer generation.
    """

    def __init__(self, document_path: str = GENERAL_CONFIG["file_path"]):
        """
        Initializes the pipeline.

        Args:
            document_path (str): The path to the source document to be loaded.
        """
        # An API key is required to use OpenAI's models.
        # Ensure the OPENAI_API_KEY environment variable is set.
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.document_path = document_path
        self.vector_store = self._create_vector_store()
        self.chain = self._create_chain()

    def _load_and_split_documents(self):
        """
        Loads the document and splits it into chunks.
        This is the first step in preparing our data.
        """
        loader = TextLoader(self.document_path)
        documents = loader.load()

        # --- Text Splitting (Parameter Tuning) ---
        # `RecursiveCharacterTextSplitter` is robust for general text.
        # `chunk_size`: Defines the maximum size of each text chunk (in characters).
        #   - A common starting point is 1000. Larger chunks provide more context but can be less precise.
        #   - Smaller chunks are more precise but may miss broader context.
        # `chunk_overlap`: Defines how many characters overlap between adjacent chunks.
        #   - This is crucial to ensure that a sentence or idea isn't awkwardly split between two chunks.
        #   - A common value is 200, or 10-20% of the chunk_size.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=BASIC_RAG_CONFIG["chunk_size"],
            chunk_overlap=BASIC_RAG_CONFIG["chunk_overlap"]
        )
        return text_splitter.split_documents(documents)

    def _create_vector_store(self):
        """
        Creates the vector store from the split documents.
        This involves embedding the text chunks and storing them in FAISS.
        """
        docs = self._load_and_split_documents()
        # OpenAIEmbeddings is a popular choice for generating high-quality text embeddings.
        embeddings = OpenAIEmbeddings()
        # FAISS is a fast, local vector store. It's great for getting started without a database.
        # `from_documents` is a convenience method that handles embedding and indexing in one step.
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    def _create_chain(self):
        """
        Creates the LangChain Expression Language (LCEL) chain.
        This chain defines the logic of the RAG pipeline.
        """
        retriever = self.vector_store.as_retriever()

        # The prompt template is key to guiding the LLM.
        # It instructs the model to answer the question based *only* on the provided context.
        # This helps prevent the model from using its pre-trained knowledge (hallucinating).
        prompt = ChatPromptTemplate.from_template(BASIC_RAG_CONFIG["prompt_template"])

        llm = ChatOpenAI(model_name=GENERAL_CONFIG["model_name"])

        # This helper function formats the retrieved documents into a single string.
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # The LCEL chain. This is where the magic happens.
        # 1. `{"context": retriever | format_docs, "question": RunnablePassthrough()}`:
        #    - This runs in parallel.
        #    - The `retriever` is called with the input question. Its output (a list of Documents)
        #      is then passed to `format_docs`.
        #    - `RunnablePassthrough()` simply passes the original question through unchanged.
        #    - The result is a dictionary with "context" and "question" keys.
        # 2. `| prompt`: The dictionary is passed to the prompt template.
        # 3. `| llm`: The formatted prompt is passed to the language model.
        # 4. `| StrOutputParser()`: The model's output (a ChatMessage) is converted to a simple string.
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        """
        Invokes the RAG chain with a specific question.

        Args:
            question (str): The user's question.

        Returns:
            str: The generated answer.
        """
        return self.chain.invoke(question)

    def invoke_with_context(self, question: str) -> dict:
        """
        Invokes the RAG chain and returns both answer and retrieved contexts.
        This is useful for evaluation purposes.

        Args:
            question (str): The user's question.

        Returns:
            dict: {"answer": str, "contexts": List[str]}
        """
        # Get the retriever and retrieve documents
        retriever = self.vector_store.as_retriever()
        retrieved_docs = retriever.invoke(question)
        
        # Extract the text content from each document
        contexts = [doc.page_content for doc in retrieved_docs]
        
        # Get the answer
        answer = self.chain.invoke(question)
        
        return {
            "answer": answer,
            "contexts": contexts
        }


# --- Entry point for the CLI ---
# This function is called by `main.py`. It acts as an adapter between the
# generic CLI and our specific RAG implementation.
def execute_rag(question: str) -> str:
    """
    Initializes and runs the Basic RAG pipeline.

    Args:
        question (str): The question to be answered.

    Returns:
        str: The answer from the RAG pipeline.
    """
    try:
        pipeline = BasicRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        # Return a helpful error message if something goes wrong.
        return f"An error occurred in the Basic RAG pipeline: {e}"


def execute_rag_with_context(question: str) -> dict:
    """
    Initializes and runs the Basic RAG pipeline with context retrieval.
    Used for evaluation purposes.

    Args:
        question (str): The question to be answered.

    Returns:
        dict: {"answer": str, "contexts": List[str]}
    """
    try:
        pipeline = BasicRAGPipeline()
        return pipeline.invoke_with_context(question)
    except Exception as e:
        # Return error information
        return {
            "answer": f"An error occurred in the Basic RAG pipeline: {e}",
            "contexts": []
        }

