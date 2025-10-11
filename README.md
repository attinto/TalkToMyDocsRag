
# Talk To My Docs (RAG Comparison Tool)

This project performs Retrieval-Augmented Generation (RAG) on the classic novel *Peter Pan*, allowing for a comparative analysis of various RAG strategies against a single, rich long text source. Setup instructions are provided below.

python -m src.main "QUESTION HERE" --all

python -m src.main "How does Captain Hook describe Peter Pan?" --all

python -m src.main "How does the tone of Neverland change between Wendy’s first visit and her last?" --all


## Answering question:  
### **How does Captain Hook describe Peter Pan?**

---

### RAG Comparison Results

| **RAG Strategy** | **Answer** |
|------------------|------------|
| **Basic RAG** | The provided context does not contain a direct description of Peter Pan by Captain Hook. It details Hook's interactions and actions regarding Peter, but does not include Hook's explicit words describing him. Therefore, I don't know how Captain Hook describes Peter Pan based on this context. |
| **Sentence Window RAG** | Captain Hook describes Peter Pan as a "wonderful boy" during the guessing game, confirming that Peter is not ordinary but extraordinary. This description comes out when Hook asks if he is an "ordinary boy" and Peter answers "No!" and then affirms being a "wonderful boy." Hook is puzzled and impressed by Peter's uniqueness. |
| **Parent Document RAG** | Captain Hook describes Peter Pan as "proud and insolent youth" and refers to him as his enemy, blaming him for his troubles. He also expresses his desire for revenge, saying, "Most of all...I want their captain, Peter Pan." Hook sees Peter as a formidable and personal adversary. |
| **Auto-merging RAG** | Captain Hook describes Peter Pan as the captain he most wants to capture, calling him “proud and insolent youth” and blaming him for giving his hook a taste for him. Hook passionately declares Peter Pan is his arch enemy, saying, “’Twas he cut off my arm.” He sees Peter as his primary foe and refers to him in dark, threatening terms. |
| **Hybrid Search RAG** | Captain Hook describes Peter Pan as the captain of the Lost Boys and his arch enemy, saying, "’Twas he cut off my arm." He refers to Peter as a "wonderful boy" rather than just an ordinary boy. Hook passionately expresses his desire for revenge against Peter, making it clear he views Peter as a formidable and infuriating adversary. |
| **Query Expansion RAG** | Captain Hook is deeply tormented by Peter Pan's cockiness, which goads him to frenzy and makes his iron claw twitch. It is Peter’s cockiness—not his courage or appearance—that most disturbs and infuriates Hook. Hook views Peter as a "wonderful boy," but is primarily obsessed with his irritating self-confidence. |

---

*Each RAG strategy retrieves and interprets slightly different nuances from the text, reflecting how retrieval context shapes model understanding.*



## Features

- **Multiple RAG Strategies**: Implements six distinct RAG pipelines out-of-the-box.
- **Flexible CLI**: A command-line interface built with Typer to easily run and compare one, many, or all strategies at once.
- **Modular Architecture**: Designed with SOLID principles, making it easy to add new RAG strategies.
- **Side-by-Side Comparison**: Displays results in a clean, formatted table for easy analysis.

## Implemented RAG Strategies

1.  **Basic RAG**: The simplest approach. It retrieves large, fixed-size chunks of text that are relevant to the question.
2.  **Sentence Window RAG**: A more precise method. It retrieves the single most relevant sentence and expands the context by including a few sentences before and after it (the "window").
3.  **Parent Document RAG**: A hybrid approach. It searches over small, precise text chunks but retrieves their larger parent documents to provide the LLM with rich, complete context.
4.  **Auto-merging RAG**: The most advanced strategy here. It retrieves many small chunks and, if several come from the same parent document, it "merges" them by retrieving the larger parent document instead. This is excellent for questions that require synthesizing information from multiple places.
5.  **Hybrid Search RAG**: Combines two different search techniques: traditional keyword-based search (BM25) and modern semantic search (FAISS). This approach leverages the precision of keywords for specific terms and the contextual understanding of vectors for broader concepts, often leading to more relevant and robust retrieval results.
6.  **Query Expansion RAG**: Uses an LLM to rewrite the user's initial question into several different variations. By searching for documents that match these expanded queries, this strategy increases the chances of finding relevant information, even if it's phrased differently from the original question.

## Centralized Configuration

To make the RAG pipelines more flexible and easier to experiment with, all tunable parameters have been centralized in `src/chatbot/rags/rag_config.py`.

This file allows you to modify various aspects of each RAG strategy without touching the core logic. You can adjust:

-   **`GENERAL_CONFIG`**: Parameters shared across all RAGs, such as the `file_path` of the source document and the `model_name` for the LLM.
-   **Individual RAG Configurations**: Each RAG (e.g., `BASIC_RAG_CONFIG`, `QUERY_EXPANSION_RAG_CONFIG`) has its own dictionary containing specific parameters like `chunk_size`, `chunk_overlap`, `prompt_template`, `window_size`, `ensemble_weights`, and `expansion_prompt_template`.

**How to Modify:**

Simply open `src/chatbot/rags/rag_config.py` and change the values of the desired parameters. Detailed comments within the file explain the purpose of each parameter and how it affects the RAG pipeline.

Experimenting with these configurations is highly encouraged to find the optimal settings for your specific use case and data.


## Setup and Installation

1.  **Prerequisites**:
    -   Python 3.10+
    -   `uv` (or `pip`)

2.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd TalkToMyDocsRag
    ```

3.  **Install dependencies**:
    ```bash
    uv pip install -r requirements.txt
    ```

4.  **Set Environment Variable**:
    This project uses OpenAI models. You must set your API key as an environment variable.
    ```bash
    export OPENAI_API_KEY='your_key_here'
    ```

    Activate venv
    source .venv/bin/activate

## How to Use

The main entry point is `src/main.py`. You can run it as a module.

### Answering Questions

**Base Command:**
`python -m src.main run "<Your Question>" [OPTIONS]`

**Options:**

- `-r, --rag-type TEXT`: Specify one or more RAG strategies to use (can be repeated)
- `--all`: Run all enabled RAG strategies

**Examples:**

```bash
# Ask a question using a single RAG strategy
python -m src.main run "How does Captain Hook describe Peter Pan?" -r basic

# Compare multiple specific strategies
python -m src.main run "How does Captain Hook describe Peter Pan?" -r basic -r sentence -r parent

# Run all enabled strategies
python -m src.main run "How does Captain Hook describe Peter Pan?" --all
```

### Evaluating RAG Strategies

The project includes a comprehensive evaluation system using the **RAGAS** (Retrieval Augmented Generation Assessment) framework. This allows you to objectively measure and compare the performance of different RAG strategies.

**Evaluation Command:**
`python -m src.main evaluate [OPTIONS]`

**Options:**

- `-s, --strategy TEXT`: Specify one or more RAG strategies to evaluate (can be repeated)
- `--all`: Evaluate all enabled RAG strategies
- `-o, --output TEXT`: Specify output file for the evaluation report (default: `evaluation_report.md`)

**Examples:**

```bash
# Evaluate a single strategy
python -m src.main evaluate -s basic

# Evaluate multiple specific strategies
python -m src.main evaluate -s basic -s hybrid -s parent

# Evaluate all enabled strategies
python -m src.main evaluate --all

# Evaluate with custom output file
python -m src.main evaluate --all -o my_results.md
```

### Understanding RAGAS Evaluation

RAGAS provides four key metrics to evaluate RAG pipeline quality:

#### 1. **Faithfulness (0-1, higher is better)**
Measures if the answer is factually consistent with the retrieved context. This metric detects hallucinations and ensures the LLM doesn't make claims unsupported by the source material.

- **High score**: Answer is well-grounded in the retrieved context
- **Low score**: Answer contains information not present in the context (hallucination)

#### 2. **Answer Relevancy (0-1, higher is better)**
Measures how well the answer addresses the original question. A factually correct answer can still score low if it doesn't actually answer what was asked.

- **High score**: Answer directly and completely addresses the question
- **Low score**: Answer is off-topic or doesn't fully address the question

#### 3. **Context Precision (0-1, higher is better)**
Measures whether the retrieved contexts are relevant to answering the question. This evaluates the quality of your retrieval mechanism.

- **High score**: Retrieved documents are highly relevant to the question
- **Low score**: Many irrelevant documents were retrieved (noisy retrieval)

#### 4. **Context Recall (0-1, higher is better)**
Measures if all necessary information from the ground truth was successfully retrieved. This checks if your retrieval mechanism missed important context.

- **High score**: All relevant information was retrieved
- **Low score**: Important information was missed during retrieval

### The Evaluation Dataset

The evaluation uses a curated dataset in `src/data/eval_dataset.json` with 6 test questions covering:

- **Factual questions** (easy/medium difficulty): Testing basic retrieval and comprehension
- **Analytical questions** (medium difficulty): Testing understanding of relationships and reasoning
- **Multi-hop questions** (hard difficulty): Testing ability to synthesize information from multiple sources

Each test case includes:
- The question
- Ground truth answer
- Ground truth context (exact passages from the source)
- Question type and difficulty level
- Expected retrieval information

### Evaluation Output

After running the evaluation, you'll get:

1. **Console output** showing real-time progress and a summary table
2. **Markdown report** with detailed metrics comparison and interpretation

Example output:

```
RAG Strategies Evaluation Report

## Overall Metrics Comparison

| Strategy              | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|-----------------------|--------------|------------------|-------------------|----------------|
| Basic RAG             | 0.850        | 0.780            | 0.720             | 0.810          |
| Sentence Window RAG   | 0.880        | 0.820            | 0.790             | 0.760          |
| Parent Document RAG   | 0.910        | 0.850            | 0.830             | 0.880          |
```

### How Evaluation Works

1. **Load Test Dataset**: The evaluator loads questions from `src/data/eval_dataset.json`
2. **Run Each Strategy**: For each test question, each RAG strategy generates an answer and retrieves contexts
3. **RAGAS Scoring**: Using LLM-based evaluation, RAGAS compares:
   - Generated answers vs. ground truth answers
   - Retrieved contexts vs. ground truth contexts
4. **Aggregate Metrics**: Scores are averaged across all questions to produce final metrics
5. **Generate Report**: Results are formatted into an easy-to-read markdown report

### Tips for Using Evaluation

- **Start small**: Evaluate one or two strategies first to understand the metrics
- **Iterate**: Use the metrics to guide parameter tuning in `src/chatbot/rags/rag_config.py`
- **Compare**: Focus on relative performance between strategies rather than absolute scores
- **Cost awareness**: RAGAS uses OpenAI API calls for evaluation, so testing all strategies on large datasets will incur costs

---

## More Notes on Advanced RAG Techniques

## More Notes on Advanced RAG Techniques

Beyond the implemented strategies, the field of Retrieval-Augmented Generation is rich with advanced techniques to improve performance, accuracy, and efficiency. Below are some key concepts that can be explored to build even more sophisticated RAG systems.

[Advanced RAG techniques](https://www.youtube.com/watch?v=sGvXO7CVwc0)

### 1. The Importance of Context, Queries, and Metadata

The quality of a RAG system's output is critically dependent on the quality of its input—both the user's query and the context it retrieves.

-   **Chunking and Embedding**: The foundational step of any RAG system is breaking down large documents into smaller, manageable chunks. These chunks are then converted into vector embeddings and stored in a database. The effectiveness of this process is crucial; if the right information isn't in a chunk, it can never be retrieved.
-   **Metadata is Key**: To enhance retrieval, each chunk should be enriched with metadata. This can include labels, categories, document IDs, or even the original question that a chunk might answer. Rich metadata allows for more powerful, filtered queries. For example, you could limit a search to chunks with a specific `document_id` or those tagged with a certain `category`, dramatically improving the chances of a precise semantic match. Specially if it is for an specific product or you can have an id that limit the chunks.
-   **Storing Questions**: A powerful technique is to store user questions alongside the data chunks that answered them. This creates a direct link between a query and its relevant context, which can be used to fine-tune future retrievals.

### 2. Database Choices for RAG

While specialized vector databases (like FAISS, Pinecone, or Weaviate) are the most common choice for storing and querying embeddings, they are not the only option. Traditional relational databases are increasingly equipped with vector search capabilities.

-   **PostgreSQL with `pgvector`**: You can use a robust, open-source database like PostgreSQL with the `pgvector` extension to handle both structured metadata and vector embeddings in the same place. This simplifies the tech stack and allows you to leverage the power of SQL for complex, metadata-based filtering alongside semantic search.

### 3. Advanced Retrieval and Ranking Techniques

Retrieval is not just about finding documents; it's about finding the *best* documents.

-   **Re-ranking**: A common pattern is to use a fast, initial retriever (like BM25 or a basic vector search) to gather a large set of potentially relevant documents. Then, a second, more computationally intensive model (a "re-ranker" or a cross-encoder) is used to score and re-order this smaller set, pushing the most relevant documents to the top.
-   **Scoring and Thresholds**: For even greater control, you can implement a scoring algorithm that assigns a relevance score to each retrieved chunk. The LLM would then only receive chunks that surpass a certain score threshold. This is a more advanced technique that can prevent irrelevant or low-quality context from ever reaching the final answer-generation step.

### 4. Evaluation and Orchestration

-   **Data-Driven Evaluation**: To objectively measure the quality of your RAG system, you can use data science techniques. This involves creating evaluation datasets with question-answer pairs and using metrics like context relevance, answer faithfulness, and overall accuracy to benchmark different strategies.
-   **Multi-Step LLM Calls**: A single call to the LLM is not always enough. More complex workflows can involve multiple interactions. For instance, you could first ask an LLM to summarize each retrieved chunk. These summaries, being more concise, can then be fed into a final LLM call to synthesize the answer. This can improve the signal-to-noise ratio and help the model focus on the most critical information.
