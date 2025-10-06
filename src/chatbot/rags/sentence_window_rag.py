
"""
Sentence Window RAG Pipeline
--------------------------

This module implements a more advanced RAG technique called Sentence Window Retrieval.

**The Problem with Basic RAG:**
Basic RAG retrieves entire chunks of text. If a chunk is large (e.g., 1000 characters),
the specific sentence containing the answer might be buried within a lot of irrelevant text,
diluting the LLM's focus.

**How Sentence Window Works:**
1.  **Indexing**: The document is split into individual sentences. Each sentence is embedded
    and stored as a separate document in the vector store. This makes retrieval highly precise.
2.  **Retrieval**: When a question is asked, the retriever finds the single sentence that is most
    semantically similar to the question.
3.  **Windowing**: Before sending the retrieved sentence to the LLM, we expand the context by
    including `k` sentences before and `k` sentences after it. This "window" provides the
    LLM with the necessary context to understand the retrieved sentence.
4.  **Generation**: The LLM answers the question using this focused, high-quality context.

**SOLID Principles:**
-   **Single Responsibility**: The `SentenceWindowRAGPipeline` class is responsible for this specific
    strategy. The logic is further broken down into methods for loading, indexing, and chain creation.
-   **Dependency Inversion**: The `execute_rag` function depends on the `SentenceWindowRAGPipeline`
    abstraction, not its concrete implementation details.
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from .rag_config import GENERAL_CONFIG, SENTENCE_WINDOW_RAG_CONFIG


class SentenceWindowRAGPipeline:
    """
    Encapsulates the logic for a Sentence Window RAG pipeline.
    """

    def __init__(self, document_path: str = GENERAL_CONFIG["file_path"], window_size: int = SENTENCE_WINDOW_RAG_CONFIG["window_size"]):
        """
        Initializes the pipeline.

        Args:
            document_path (str): The path to the source document.
            window_size (int): The number of sentences to include before and after the
                               retrieved sentence. This is a key tuning parameter.
                               - A larger window provides more context but may introduce noise.
                               - A smaller window is more focused but may miss relevant info.
                               A value of 1 or 2 is a good starting point.
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.window_size = window_size
        self.all_sentences = self._load_and_split_into_sentences(document_path)
        self.vector_store = self._create_vector_store()
        self.chain = self._create_chain()

    def _load_and_split_into_sentences(self, document_path: str) -> list[str]:
        """
        Loads a document and splits it into a list of sentences.
        """
        loader = TextLoader(document_path)
        documents = loader.load()

        # Use a text splitter configured to split by sentence markers.
        # We first join all document content and then split.
        full_text = " ".join(doc.page_content for doc in documents)
        # A simple split by '.' can be effective, but a more robust sentence splitter
        # from a library like NLTK would be better for production.
        # For this example, we'll keep it simple.
        sentences = full_text.split('. ')
        return [s.strip() for s in sentences if s.strip()] # Remove empty strings

    def _create_vector_store(self):
        """
        Creates a vector store where each entry is a single sentence.
        """
        # Create LangChain Document objects for each sentence.
        sentence_docs = [Document(page_content=s) for s in self.all_sentences]

        embeddings = OpenAIEmbeddings()
        # Index the sentence documents in FAISS.
        vector_store = FAISS.from_documents(sentence_docs, embeddings)
        return vector_store

    def _get_window_context(self, docs: list[Document]) -> str:
        """
        A custom function to create the window of context around retrieved sentences.
        """
        # Get the content of the retrieved sentences
        retrieved_sentences = [doc.page_content for doc in docs]
        all_retrieved_indices = []

        # Find the index of each retrieved sentence in the original document
        for sentence in retrieved_sentences:
            try:
                index = self.all_sentences.index(sentence)
                all_retrieved_indices.append(index)
            except ValueError:
                pass # Sentence not found, skip

        # Expand the indices to include the window
        window_indices = set()
        for index in all_retrieved_indices:
            start = max(0, index - self.window_size)
            end = min(len(self.all_sentences), index + self.window_size + 1)
            window_indices.update(range(start, end))

        # Sort the unique indices and build the final context string
        final_indices = sorted(list(window_indices))
        context = ". ".join(self.all_sentences[i] for i in final_indices)
        return context

    def _create_chain(self):
        """
        Creates the LCEL chain for the Sentence Window pipeline.
        """
        from operator import itemgetter

        # The retriever will fetch the top_k most relevant sentences.
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(SENTENCE_WINDOW_RAG_CONFIG["prompt_template"])

        llm = ChatOpenAI(model_name=GENERAL_CONFIG["model_name"])

        # This sub-chain is responsible for fetching and formatting the context.
        # 1. `itemgetter("question")`: Extracts the question string from the input dict.
        # 2. `retriever`: The question string is passed to the retriever.
        # 3. `RunnableLambda(self._get_window_context)`: The retrieved docs are passed to our
        #    custom function to build the final context string.
        context_chain = (
            itemgetter("question")
            | retriever
            | RunnableLambda(self._get_window_context)
        )

        # The full chain.
        # `RunnablePassthrough.assign(context=...)` runs the context_chain and assigns
        # its output to the "context" key, while keeping the original "question" key.
        chain = (
            RunnablePassthrough.assign(
                context=context_chain
            )
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def invoke(self, question: str) -> str:
        """
        Invokes the RAG chain with a specific question.
        """
        return self.chain.invoke({"question": question})


# --- Entry point for the CLI ---
def execute_rag(question: str) -> str:
    """
    Initializes and runs the Sentence Window RAG pipeline.
    """
    try:
        pipeline = SentenceWindowRAGPipeline()
        return pipeline.invoke(question)
    except Exception as e:
        return f"An error occurred in the Sentence Window RAG pipeline: {e}"
