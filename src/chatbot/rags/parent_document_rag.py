
"""
Parent Document RAG Pipeline
--------------------------

This module implements the Parent Document Retriever strategy.

**The Problem:**
-   **Small Chunks:** Good for precise searching, but lack surrounding context for the LLM.
-   **Large Chunks:** Good context for the LLM, but the search can be imprecise.

**How Parent Document RAG Works (The Best of Both Worlds):**
1.  **Splitting**: The source document is split twice.
    -   First, into larger "Parent Documents" (e.g., a whole page or large paragraph).
    -   Second, each Parent Document is split into smaller "Child Documents" (e.g., sentences).
2.  **Indexing**: Only the small Child Documents are embedded and stored in a vector store.
    This makes the search step highly accurate.
3.  **Retrieval**: When a question is asked, the retriever finds the most relevant Child Documents.
4.  **Lookup**: Instead of returning the small child chunks, the retriever looks up their corresponding
    Parent Documents from a separate document store.
5.  **Generation**: The full content of the Parent Documents is passed to the LLM, providing it
    with rich, complete context to generate a high-quality answer.

**SOLID Principles:**
-   This implementation follows the same SOLID structure as the other RAG modules, encapsulating
    the specific strategy within its own class and using a clean `execute_rag` entry point.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .rag_config import GENERAL_CONFIG, PARENT_DOCUMENT_RAG_CONFIG


class ParentDocumentRAGPipeline:
    """
    Encapsulates the logic for a Parent Document RAG pipeline.
    """

    def __init__(self, document_path: str = GENERAL_CONFIG["file_path"]):
        """
        Initializes the pipeline by setting up the retriever and the LCEL chain.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.retriever = self._create_retriever(document_path)
        self.chain = self._create_chain()

    def _create_retriever(self, document_path: str) -> ParentDocumentRetriever:
        """
        Sets up the ParentDocumentRetriever.
        This involves defining parent/child splitters and configuring the stores.
        """
        loader = TextLoader(document_path)
        docs = loader.load()

        # --- Splitter Configuration (Parameter Tuning) ---
        # This splitter creates the Parent Documents. These should be large enough
        # to provide complete context for any child chunk within them.
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_DOCUMENT_RAG_CONFIG["parent_chunk_size"], chunk_overlap=200)

        # This splitter creates the Child Documents. These are the small, precise chunks
        # that will be embedded and searched over.
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=PARENT_DOCUMENT_RAG_CONFIG["child_chunk_size"], chunk_overlap=50)

        # The vector store to use for indexing the child documents.
        # FAISS cannot be initialized empty, so we create it with a dummy text.
        vectorstore = FAISS.from_texts(["dummy"], OpenAIEmbeddings())

        # The in-memory store for the parent documents.
        docstore = InMemoryStore()

        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
        )

        # This crucial step adds the documents to the retriever. The retriever handles
        # the splitting, embedding, and storage of both parent and child documents.
        retriever.add_documents(docs)
        return retriever

    def _create_chain(self):
        """
        Creates the LCEL chain. This is very similar to the basic RAG chain,
        as the ParentDocumentRetriever handles the complex logic.
        """
        prompt = ChatPromptTemplate.from_template(PARENT_DOCUMENT_RAG_CONFIG["prompt_template"])

        llm = ChatOpenAI(model_name=GENERAL_CONFIG["model_name"])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # The LCEL chain.
        # The `retriever` here is the ParentDocumentRetriever. When invoked, it finds
        # the relevant child docs and returns their corresponding parent docs.
        chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
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
    Initializes and runs the Parent Document RAG pipeline.
    """
    try:
        pipeline = ParentDocumentRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        return f"An error occurred in the Parent Document RAG pipeline: {e}"
