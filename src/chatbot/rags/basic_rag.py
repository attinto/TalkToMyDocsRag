
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
import logging
import time
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .rag_config import GENERAL_CONFIG, BASIC_RAG_CONFIG

# Logger for this module
logger = logging.getLogger(__name__)


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
        logger.info("Initializing BasicRAGPipeline with document_path=%s", document_path)
        start = time.time()
        self.vector_store = self._create_vector_store()
        vs_time = time.time() - start
        logger.info("Vector store created in %.2fs", vs_time)

        start = time.time()
        self.chain = self._create_chain()
        chain_time = time.time() - start
        logger.info("Chain created in %.2fs", chain_time)

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
            chunk_overlap=BASIC_RAG_CONFIG["chunk_overlap"],
        )
        return text_splitter.split_documents(documents)

    def _create_vector_store(self):
        """
        Creates the vector store from the split documents.
        This involves embedding the text chunks and storing them in FAISS.
        """
        docs = self._load_and_split_documents()
        logger.info("Loaded and split documents: %d chunks", len(docs))

        # OpenAIEmbeddings is a popular choice for generating high-quality text embeddings.
        embeddings = OpenAIEmbeddings()
        # FAISS is a fast, local vector store. It's great for getting started without a database.
        # `from_documents` is a convenience method that handles embedding and indexing in one step.
        t0 = time.time()
        vector_store = FAISS.from_documents(docs, embeddings)
        logger.info("FAISS.from_documents finished in %.2fs", time.time() - t0)
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
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> dict:
        """
        Invokes the RAG chain with a specific question.

        Args:
            question (str): The user's question.

        Returns:
            dict: A dictionary containing the answer and the retrieved context.
        """
        logger.info("Invoking BasicRAGPipeline for question: %s", question if len(question) < 200 else question[:200] + "...")

        t0 = time.time()
        docs = self.vector_store.similarity_search(question)
        retrieval_time = time.time() - t0
        logger.info("Retrieved %d documents in %.3fs", len(docs), retrieval_time)

        # Format documents into a single context string (same logic as format_docs)
        context_str = "\n\n".join(doc.page_content for doc in docs)

        # Pass the precomputed context into the chain to avoid retriever running again
        t1 = time.time()
        result = self.chain.invoke({"question": question, "context": context_str})
        llm_time = time.time() - t1
        logger.info("LLM chain invoked in %.3fs", llm_time)

        # Log a short preview of the answer
        preview = str(result)
        if len(preview) > 200:
            preview = preview[:200] + "..."
        logger.debug("Answer preview: %s", preview)

        return {"answer": result, "context": docs}


# --- Entry point for the CLI ---
# This function is called by `main.py`. It acts as an adapter between the
# generic CLI and our specific RAG implementation.
def execute_rag(question: str) -> dict:
    """
    Initializes and runs the Basic RAG pipeline.

    Args:
        question (str): The question to be answered.

    Returns:
        dict: A dictionary containing the answer and the retrieved context.
    """
    # Use a module-level cached pipeline to avoid rebuilding embeddings/vectorstore on every call
    global _BASIC_PIPELINE
    try:
        _BASIC_PIPELINE
    except NameError:
        _BASIC_PIPELINE = None

    try:
        if _BASIC_PIPELINE is None:
            _BASIC_PIPELINE = BasicRAGPipeline()
        return _BASIC_PIPELINE.invoke(question)
    except Exception as e:
        return {"answer": f"An error occurred in the Basic RAG pipeline: {e}", "context": []}

