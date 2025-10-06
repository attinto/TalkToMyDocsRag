# TalkToMyDocsRag

This project is a chatbot that can answer questions about a book. It uses a Retrieval-Augmented Generation (RAG) model built with LangChain.

## Prerequisites

This project uses `uv` to manage the Python environment and dependencies. If you don't have `uv` installed, you can install it with:

```bash
pip install uv
```

## Setup

1.  **Create a virtual environment:**

    Use `uv` to create a virtual environment.

    ```bash
    uv venv
    ```

2.  **Activate the virtual environment:**

    On macOS and Linux:

    ```bash
    source .venv/bin/activate
    ```

    On Windows:

    ```bash
    .venv\Scripts\activate
    ```

3.  **Install dependencies:**

    Install the project dependencies using `uv`.

    ```bash
    uv pip install langchain openai langchain_community faiss-cpu pypdf python-dotenv ipykernel
    ```

4.  **Set up your OpenAI API Key:**

    Create a `.env` file in the root of the project and add your OpenAI API key to it:

    ```
    OPENAI_API_KEY=your_openai_api_key
    ```

5.  **Add the book:**

    Place the book you want to chat with in the `src/book` directory. The chatbot is currently configured to read `historia_venezuela.txt`.

## Usage

To run the chatbot, execute the `main.py` script from the root of the project:

```bash
python src/main.py
```

## Key Concepts

### Embeddings

Embeddings are numerical representations of text (or other data types like images and audio). They are created by deep learning models and capture the semantic meaning of the content. In the context of RAG, words, phrases, and entire document chunks are converted into vectors. This allows for mathematical comparisons between different pieces of text. For example, the embeddings for "king" and "queen" would be closer to each other than to the embedding for "apple".

### Vector Store

A vector store is a specialized database designed to store and search for embeddings. It takes the high-dimensional vectors created by the embedding model and indexes them for efficient similarity search. When a user asks a question, the question is also converted into an embedding, and the vector store is used to find the document chunks with the most similar embeddings. This is the "retrieval" part of RAG.

### Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models (LLMs) with the vastness of external knowledge bases. Instead of relying solely on the information it was trained on, a RAG model can access and use up-to-date information to answer questions and generate text.

The process works in two main stages:

1.  **Retrieval:** When a user provides a prompt or asks a question, the RAG model first searches a knowledge base (like a collection of documents or a database) for relevant information. This is done by converting the user's query into an embedding and using a vector store to find the most similar content.
2.  **Generation:** The retrieved information is then combined with the original prompt and fed to the LLM. The LLM uses this augmented context to generate a more accurate and informative response.

This approach helps to reduce hallucinations (the model making up facts) and allows the model to answer questions about topics it wasn't explicitly trained on.

## How it Works: The RAG Pipeline

The chatbot uses a Retrieval-Augmented Generation (RAG) architecture. This approach enhances the power of Large Language Models (LLMs) by connecting them to external knowledge sources. Here's a breakdown of the two main phases of the RAG pipeline:

### 1. Indexing Phase: Preparing the Knowledge Base

This phase is about preparing the external data for the LLM to use.

*   **Document Loading:** The process starts by loading the book's text file into memory using LangChain's `TextLoader`.
*   **Text Splitting:** The loaded document is then split into smaller, more manageable chunks using the `RecursiveCharacterTextSplitter`. This is crucial for fitting the text within the LLM's context window.
*   **Embedding:** Each text chunk is converted into a numerical vector using an embedding model (like the one from OpenAI). These vectors capture the semantic meaning of the text.
*   **Storage:** The embeddings are stored in a vector database (in this case, FAISS - Facebook AI Similarity Search). This allows for efficient searching of the most relevant text chunks based on their semantic similarity.

### 2. Retrieval and Generation Phase: Answering Questions

This phase is triggered when a user asks a question.

*   **Query Embedding:** The user's question is also converted into a vector using the same embedding model.
*   **Similarity Search:** The vector of the question is used to search the vector store for the most similar document chunks.
*   **Context Assembly:** The retrieved document chunks are then combined with the original question to form an "augmented prompt".
*   **Generation:** This augmented prompt is then fed to the LLM (e.g., GPT-4), which generates a comprehensive answer based on the provided context.

## Chain Types in LangChain

LangChain provides different ways to combine the retrieved documents and the user's query. These are called "chain types". This project uses the `map_reduce` chain type.

Here are some of the most common chain types:

### `stuff`

This is the simplest method. It "stuffs" all the retrieved document chunks into a single prompt along with the user's question.

*   **Example:** Imagine you have 3 retrieved document chunks. The `stuff` method would combine them into one large piece of text and send it to the LLM.
*   **Pros:** Simple, fast, and gives the LLM all the context at once.
*   **Cons:** Fails if the combined text exceeds the LLM's context window limit.

### `map_reduce`

This method is designed to handle a large number of documents that won't fit into the context window.

*   **Map Step:** It processes each document chunk individually, sending each one to the LLM with a prompt (e.g., "summarize this document chunk"). This creates a set of summaries.
*   **Reduce Step:** It then takes the summaries, combines them, and sends them to the LLM with a final prompt (e.g., "based on these summaries, answer the user's question").
*   **Example:** If you have 100 retrieved documents, the `map_reduce` chain would first generate 100 summaries, and then combine those summaries to generate the final answer.
*   **Pros:** Can handle a very large number of documents.
*   **Cons:** Requires more calls to the LLM, making it slower and more expensive. Can lose some detail as it relies on summaries.

### `refine`

The `refine` chain type processes documents one by one, refining the answer at each step.

*   **Process:** It sends the first document to the LLM to get an initial answer. Then, it sends the second document along with the previous answer to the LLM, asking it to refine the answer based on the new document. This process is repeated for all documents.
*   **Example:** For a set of documents, it would generate an answer from the first, then improve that answer with the second, then improve it again with the third, and so on.
*   **Pros:** Can handle a large number of documents and can build up a more detailed answer.
*   **Cons:** Requires many calls to the LLM and is order-dependent (the order of the documents matters).

### `map_rerank`

The `map_rerank` chain type is designed for question-answering tasks where you want to find the best answer from a set of documents.

*   **Process:** It processes each document individually, asking the LLM to answer the question based on that document and to provide a confidence score.
*   **Rerank Step:** It then selects the answer with the highest confidence score.
*   **Example:** If you have multiple documents that might contain the answer, `map_rerank` will evaluate each one and pick the best one.
*   **Pros:** Good for finding the most relevant answer from multiple sources.
*   **Cons:** Only returns the answer from a single document.

## Tuning Parameters

You can tune the following parameters to adjust the chatbot's behavior:

### In `src/chatbot/vector_store.py`:

*   `chunk_size`: The number of characters in each text chunk.
*   `chunk_overlap`: The number of characters to overlap between adjacent chunks.

### In `src/chatbot/chatbot.py`:

*   `model_name`: The OpenAI model to use for generating answers.
*   `temperature`: Controls the randomness of the generated text.
*   `chain_type`: The type of chain to use. This project uses `map_reduce`. You can experiment with other chain types like `stuff`, `refine`, or `map_rerank`.