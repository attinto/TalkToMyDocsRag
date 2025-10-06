
"""
Centralized Configuration for RAG Pipelines
-------------------------------------------

This file contains all the tunable parameters for the different RAG strategies.
By modifying the values in this file, you can experiment with different models,
chunking strategies, and prompts without altering the core logic of the pipelines.

Each RAG strategy has its own configuration dictionary. A `GENERAL_CONFIG` dictionary
holds parameters that are shared across all pipelines.

How to Tune Parameters:
1.  **LLM Model**: Change `model_name` to use different OpenAI models (e.g., "gpt-4-turbo", "gpt-3.5-turbo").
2.  **Chunking**: Adjust `chunk_size` and `chunk_overlap` to control how documents are split.
3.  **Prompts**: Modify the `prompt_template` for each RAG to change how the LLM is instructed to generate answers.
4.  **Retrieval**: For RAGs like Hybrid Search, you can tune the `weights` to favor one retriever over another.
"""

# --- General Configuration (Shared Across RAGs) ---
GENERAL_CONFIG = {
    "file_path": "src/data/peter_pan_book.txt",
    "model_name": "gpt-4.1-2025-04-14",
}

# --- Configuration for Basic RAG ---
BASIC_RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "prompt_template": """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.

    Context: {context}
    Question: {question}
    Answer:
    """
}

# --- Configuration for Sentence Window RAG ---
SENTENCE_WINDOW_RAG_CONFIG = {
    # The number of sentences to include before and after the retrieved sentence.
    "window_size": 3,
    "prompt_template": BASIC_RAG_CONFIG["prompt_template"] # Reusing the basic prompt
}

# --- Configuration for Parent Document RAG ---
PARENT_DOCUMENT_RAG_CONFIG = {
    "parent_chunk_size": 2000,
    "child_chunk_size": 400,
    "prompt_template": BASIC_RAG_CONFIG["prompt_template"] # Reusing the basic prompt
}

# --- Configuration for Auto-merging RAG ---
AUTO_MERGING_RAG_CONFIG = {
    "parent_chunk_size": 2000,
    "child_chunk_size": 400,
    "merge_threshold": 3, # Number of retrieved child documents from the same parent to trigger a merge
    "prompt_template": BASIC_RAG_CONFIG["prompt_template"] # Reusing the basic prompt
}

# --- Configuration for Hybrid Search RAG ---
HYBRID_SEARCH_RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    # Weights for the ensemble retriever [BM25, FAISS]. Must sum to 1.0
    "ensemble_weights": [0.5, 0.5],
    "prompt_template": BASIC_RAG_CONFIG["prompt_template"] # Reusing the basic prompt
}

# --- Configuration for Query Expansion RAG ---
QUERY_EXPANSION_RAG_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "expansion_prompt_template": """
    You are a helpful assistant that generates multiple search queries based on a single input query.
    Generate 3 additional questions related to the original question.
    The questions should be diverse and cover different aspects of the original question.
    Output ONLY the queries, each on a new line. Do not number them.

    Original question: {question}
    """,
    "main_rag_prompt_template": BASIC_RAG_CONFIG["prompt_template"] # Reusing the basic prompt
}
