
"""
Query Expansion RAG Pipeline
----------------------------

This module implements a Retrieval-Augmented Generation (RAG) pipeline that uses query expansion to improve retrieval.

**Key Components:**
1.  **Query Expansion**: An LLM is used to generate multiple variations of the user's question.
2.  **DocumentLoader**: Loads the source text from a file.
3.  **TextSplitter**: Splits the document into smaller, manageable chunks.
4.  **Embeddings**: Converts text chunks into numerical vectors.
5.  **VectorStore (FAISS)**: Stores the vectors for semantic search.
6.  **Retriever**: Fetches documents for each of the generated queries.
7.  **LLM**: An OpenAI chat model that generates the final answer.
8.  **LCEL Chain**: The LangChain Expression Language chain that orchestrates the flow.

**SOLID Principles:**
-   **Single Responsibility Principle**: The `QueryExpansionRAGPipeline` class is responsible for this specific RAG strategy.
-   **Open/Closed Principle**: The `execute_rag` function provides a stable entry point.
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
from langchain.load import dumps, loads

from .rag_config import GENERAL_CONFIG, QUERY_EXPANSION_RAG_CONFIG


class QueryExpansionRAGPipeline:
    """
    Encapsulates the logic for a Query Expansion RAG pipeline.
    """

    def __init__(self, document_path: str = GENERAL_CONFIG["file_path"]):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.document_path = document_path
        self.vector_store = self._create_vector_store()
        self.chain = self._create_chain()

    def _load_and_split_documents(self):
        loader = TextLoader(self.document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=QUERY_EXPANSION_RAG_CONFIG["chunk_size"],
            chunk_overlap=QUERY_EXPANSION_RAG_CONFIG["chunk_overlap"]
        )
        return text_splitter.split_documents(documents)

    def _create_vector_store(self):
        docs = self._load_and_split_documents()
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    def _create_chain(self):
        retriever = self.vector_store.as_retriever()

        # --- Query Expansion (Parameter Tuning) ---
        # This prompt is crucial. It instructs the LLM on how to expand the query.
        # You can tune this to change the style and number of generated questions.
        # For example, you could ask for more creative or more technical variations.
        expansion_prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_RAG_CONFIG["expansion_prompt_template"])

        llm = ChatOpenAI(model_name=GENERAL_CONFIG["model_name"])

        # Chain to generate the expanded queries
        expansion_chain = expansion_prompt | llm | StrOutputParser()

        def get_retrieved_docs(question):
            # Generate expanded queries
            expanded_queries_str = expansion_chain.invoke({"question": question})
            queries = [question] + expanded_queries_str.strip().split('\n')
            
            # Retrieve documents for all queries
            docs = []
            for q in queries:
                docs.extend(retriever.invoke(q))
            
            # Deduplicate documents based on page_content
            unique_docs = {doc.page_content: doc for doc in docs}
            return list(unique_docs.values())

        # Main RAG prompt
        prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_RAG_CONFIG["main_rag_prompt_template"])

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": get_retrieved_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> dict:
        result = self.chain.invoke(question)
        # The context is already part of the chain's execution, but we need to retrieve it separately for ragas
        retriever = self.vector_store.as_retriever()
        expansion_prompt = ChatPromptTemplate.from_template(QUERY_EXPANSION_RAG_CONFIG["expansion_prompt_template"])
        llm = ChatOpenAI(model_name=GENERAL_CONFIG["model_name"])
        expansion_chain = expansion_prompt | llm | StrOutputParser()
        expanded_queries_str = expansion_chain.invoke({"question": question})
        queries = [question] + expanded_queries_str.strip().split('\n')
        docs = []
        for q in queries:
            docs.extend(retriever.invoke(q))
        unique_docs = {doc.page_content: doc for doc in docs}
        context = list(unique_docs.values())

        return {"answer": result, "context": context}


# --- Entry point for the CLI ---
def execute_rag(question: str) -> dict:
    """
    Initializes and runs the Query Expansion RAG pipeline.
    """
    try:
        pipeline = QueryExpansionRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        return {"answer": f"An error occurred in the Query Expansion RAG pipeline: {e}", "context": []}

