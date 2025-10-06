
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

# --- Constants and Configuration ---
FILE_PATH = "src/data/peter_pan_book.txt"
LLM_MODEL = "gpt-4.1-2025-04-14"


class HybridSearchRAGPipeline:
    """
    Encapsulates the logic for a hybrid search RAG pipeline.
    """

    def __init__(self, document_path: str = FILE_PATH):
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
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
            weights=[0.5, 0.5],
            search_type="mmr"
        )

        prompt_template = """
        You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.

        Context: {context}
        Question: {question}
        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)

        llm = ChatOpenAI(model_name=LLM_MODEL)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        """
        Invokes the RAG chain with a specific question.
        """
        return self.chain.invoke(question)


# --- Entry point for the CLI ---
def execute_rag(question: str) -> str:
    """
    Initializes and runs the Hybrid Search RAG pipeline.
    """
    try:
        pipeline = HybridSearchRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        return f"An error occurred in the Hybrid Search RAG pipeline: {e}"

