
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

# --- Constants and Configuration ---
FILE_PATH = "src/data/peter_pan_book.txt"
LLM_MODEL = "gpt-4.1-2025-04-14"


class QueryExpansionRAGPipeline:
    """
    Encapsulates the logic for a Query Expansion RAG pipeline.
    """

    def __init__(self, document_path: str = FILE_PATH):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.document_path = document_path
        self.vector_store = self._create_vector_store()
        self.chain = self._create_chain()

    def _load_and_split_documents(self):
        loader = TextLoader(self.document_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
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
        expansion_prompt_template = """
        You are a helpful assistant that generates multiple search queries based on a single input query.
        Generate 3 additional questions related to the original question.
        The questions should be diverse and cover different aspects of the original question.
        Output ONLY the queries, each on a new line. Do not number them.

        Original question: {question}
        """
        expansion_prompt = ChatPromptTemplate.from_template(expansion_prompt_template)

        llm = ChatOpenAI(model_name=LLM_MODEL)

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

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": get_retrieved_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        return self.chain.invoke(question)


# --- Entry point for the CLI ---
def execute_rag(question: str) -> str:
    """
    Initializes and runs the Query Expansion RAG pipeline.
    """
    try:
        pipeline = QueryExpansionRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        return f"An error occurred in the Query Expansion RAG pipeline: {e}"

