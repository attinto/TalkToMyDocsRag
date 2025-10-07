
"""
Hybrid Search RAG Pipeline
--------------------------

This module implements a Retrieval-Augmented Generation (RAG) pipeline that uses hybrid search.

**Key Components:**
1.  **DocumentLoader**: Loads the source text from a file.
2.  **TextSplitter**: Splits the document into smaller, manageable chunks.
3.  **Embeddings**: Converts text chunks into numerical vectors using an OpenAI model.
4.  **VectorStore (FAISS)**: Stores the vectors for semantic search.
5.  **BM25Retriever**: A keyword-based retriever.
6.  **EnsembleRetriever**: Combines the results of the FAISS retriever and the BM25 retriever.
7.  **LLM**: An OpenAI chat model that generates the answer.
8.  **LCEL Chain**: The LangChain Expression Language chain that orchestrates the flow.

**SOLID Principles:**
-   **Single Responsibility Principle**: The `HybridSearchRAGPipeline` class is responsible for constructing and executing the hybrid search RAG strategy.
-   **Open/Closed Principle**: The `execute_rag` function provides a stable entry point, while the internal logic can be modified.
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
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

from .rag_config import GENERAL_CONFIG, HYBRID_SEARCH_RAG_CONFIG


class HybridSearchRAGPipeline:
    """
    Encapsulates the logic for a hybrid search RAG pipeline.
    """

    def __init__(self, document_path: str = GENERAL_CONFIG["file_path"]):
        """
        Initializes the pipeline.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.document_path = document_path
        self.docs = self._load_and_split_documents()
        self.chain = self._create_chain()

    def _load_and_split_documents(self):
        """
        Loads the document and splits it into chunks.
        """
        loader = TextLoader(self.document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=HYBRID_SEARCH_RAG_CONFIG["chunk_size"],
            chunk_overlap=HYBRID_SEARCH_RAG_CONFIG["chunk_overlap"]
        )
        return text_splitter.split_documents(documents)

    def _create_chain(self):
        """
        Creates the LangChain Expression Language (LCEL) chain.
        """
        # Create the FAISS vector store and retriever
        embeddings = OpenAIEmbeddings()
        faiss_vectorstore = FAISS.from_documents(self.docs, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

        # Create the BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = 5

        # Create the ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=HYBRID_SEARCH_RAG_CONFIG["ensemble_weights"],
            search_type="mmr"
        )

        prompt = ChatPromptTemplate.from_template(HYBRID_SEARCH_RAG_CONFIG["prompt_template"])

        llm = ChatOpenAI(model_name=GENERAL_CONFIG["model_name"])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": ensemble_retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> dict:
        """
        Invokes the RAG chain with a specific question.

        Returns:
            dict: A dictionary containing the answer and the retrieved context.
        """
        result = self.chain.invoke(question)
        # The context is already part of the chain's execution, but we need to retrieve it separately for ragas
        embeddings = OpenAIEmbeddings()
        faiss_vectorstore = FAISS.from_documents(self.docs, embeddings)
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = 5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever],
            weights=HYBRID_SEARCH_RAG_CONFIG["ensemble_weights"],
            search_type="mmr"
        )
        context = ensemble_retriever.get_relevant_documents(question)
        return {"answer": result, "context": context}


# --- Entry point for the CLI ---
def execute_rag(question: str) -> dict:
    """
    Initializes and runs the Hybrid Search RAG pipeline.

    Returns:
        dict: A dictionary containing the answer and the retrieved context.
    """
    global _HYBRID_PIPELINE
    try:
        _HYBRID_PIPELINE
    except NameError:
        _HYBRID_PIPELINE = None

    try:
        if _HYBRID_PIPELINE is None:
            _HYBRID_PIPELINE = HybridSearchRAGPipeline()
        return _HYBRID_PIPELINE.invoke(question)
    except Exception as e:
        return {"answer": f"An error occurred in the Hybrid Search RAG pipeline: {e}", "context": []}

