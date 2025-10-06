
"""
Auto-merging RAG Pipeline
-------------------------

This module implements the Auto-merging Retriever strategy, a sophisticated technique
designed to answer complex questions that require synthesizing information from multiple
parts of a document.

**The Problem:**
Many questions (e.g., "Compare character A at the beginning and end of the story")
cannot be answered from a single contiguous chunk of text. Basic retrievers fetch
multiple small chunks, but the LLM then has to piece together a dozen fragmented
snippets, which can be difficult and lead to poor answers.

**How Auto-merging Works:**
1.  **Hierarchy**: The document is split into a hierarchy: large parent chunks and smaller
    child chunks derived from them.
2.  **Indexing**: Only the small child chunks are indexed, making retrieval precise.
    Each child chunk keeps a reference to its parent's ID.
3.  **Retrieval**: A large number of child chunks are retrieved for a given question.
4.  **Merging Logic**: This is the key step. The system checks the retrieved child chunks.
    If multiple chunks (e.g., more than 3) all come from the *same parent document*,
    it assumes that a broader context is needed. It then discards those small chunks
    and fetches the single, larger parent document instead.
5.  **Generation**: The final context sent to the LLM is a mix of large, merged parent
    documents and any other relevant small chunks. This gives the LLM both the
    "big picture" and the specific details it needs.
"""

import os
import uuid
from collections import defaultdict
from operator import itemgetter

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Constants and Configuration ---
FILE_PATH = "src/data/peter_pan_book.txt"
LLM_MODEL = "gpt-3.5-turbo"


class AutoMergingRAGPipeline:
    """
    Encapsulates the logic for an Auto-merging RAG pipeline.
    """

    def __init__(self, document_path: str = FILE_PATH, merge_threshold: int = 3):
        """
        Initializes the pipeline.

        Args:
            document_path (str): Path to the source document.
            merge_threshold (int): The number of retrieved child documents from the same
                                   parent required to trigger a merge. This is a key
                                   tuning parameter.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.merge_threshold = merge_threshold
        self.vector_store, self.doc_store = self._create_stores_and_index(document_path)
        self.chain = self._create_chain()

    def _create_stores_and_index(self, document_path: str):
        """
        Creates the vector and document stores and indexes the documents.
        """
        docs = TextLoader(document_path).load()

        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=200)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        parent_docs = parent_splitter.split_documents(docs)
        parent_ids = [str(uuid.uuid4()) for _ in parent_docs]

        child_docs = []
        for i, p_doc in enumerate(parent_docs):
            _child_docs = child_splitter.split_documents([p_doc])
            for _child_doc in _child_docs:
                _child_doc.metadata["parent_id"] = parent_ids[i]
            child_docs.extend(_child_docs)

        doc_store = InMemoryStore()
        doc_store.mset(list(zip(parent_ids, parent_docs)))

        vector_store = FAISS.from_documents(child_docs, OpenAIEmbeddings())

        return vector_store, doc_store

    def _auto_merge_retrieved_documents(self, docs: list[Document]) -> list[Document]:
        """
        The core merging logic.
        It checks for multiple retrieved chunks from the same parent and merges them.
        """
        parent_id_to_docs = defaultdict(list)
        for doc in docs:
            if "parent_id" in doc.metadata:
                parent_id_to_docs[doc.metadata["parent_id"]].append(doc)

        final_docs = []
        for parent_id, child_docs in parent_id_to_docs.items():
            if len(child_docs) >= self.merge_threshold:
                # If the threshold is met, retrieve and add the parent document.
                parent_doc = self.doc_store.mget([parent_id])[0]
                if parent_doc:
                    final_docs.append(parent_doc)
            else:
                # Otherwise, just add the individual child documents.
                final_docs.extend(child_docs)
        return final_docs

    def _create_chain(self):
        """
        Creates the LCEL chain for the Auto-merging pipeline.
        """
        # Retrieve a larger number of docs to give the merging logic more to work with.
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 20})

        prompt_template = """... (same as before) ..."""
        prompt = ChatPromptTemplate.from_template(prompt_template.replace("... (same as before) ...",
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}\nQuestion: {question}\nAnswer:"))

        llm = ChatOpenAI(model_name=LLM_MODEL)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        retrieval_and_merging_chain = (
            retriever
            | RunnableLambda(self._auto_merge_retrieved_documents)
        )

        chain = (
            RunnablePassthrough.assign(
                context=itemgetter("question") | retrieval_and_merging_chain | format_docs
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        return self.chain.invoke({"question": question})


# --- Entry point for the CLI ---
def execute_rag(question: str) -> str:
    try:
        pipeline = AutoMergingRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        return f"An error occurred in the Auto-merging RAG pipeline: {e}"
