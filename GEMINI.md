## Project Overview

This project is a Python-based command-line tool for comparing various Retrieval-Augmented Generation (RAG) strategies. It uses the classic novel *Peter Pan* as a sample long-text source to answer questions. The tool is built with `Typer` for the CLI, `langchain` for the RAG pipelines, and `faiss-cpu` for vector storage. The architecture is modular, allowing for easy addition and configuration of new RAG strategies.

## Building and Running

**1. Install Dependencies:**

```bash
uv pip install -r requirements.txt
```

**2. Set Environment Variable:**

This project uses OpenAI models. You must set your API key as an environment variable.

```bash
export OPENAI_API_KEY='your_key_here'
```

**3. Run the tool:**

The main entry point is `src/main.py`. You can run it as a module.

**Base Command:**

```bash
python -m src.main "<Your Question>" [OPTIONS]
```

**Options:**

*   `-r`, `--rag-type TEXT`: Specify a RAG strategy to run (e.g., `basic`, `sentence`, `parent`, `merging`). Can be used multiple times.
*   `--all`: Run all enabled RAG strategies defined in `config.json`.

**Examples:**

*   **Run a single RAG strategy:**
    ```bash
    python -m src.main "Who is Tinker Bell?" -r basic
    ```
*   **Compare two specific strategies:**
    ```bash
    python -m src.main "What did Wendy do with Peter's shadow?" -r basic -r sentence
    ```
*   **Run all strategies for a complex question:**
    ```bash
    python -m src.main "How does Wendy's perception of Peter Pan change from their first meeting to the end of the story?" --all
    ```

## Development Conventions

*   **Modular Architecture:** Each RAG strategy is implemented in its own module within the `src/chatbot/rags/` directory.
*   **Configuration:** The `config.json` file at the root of the project is used to configure the available RAG strategies and their settings. Each RAG strategy also has its own configuration file in `src/chatbot/rags/rag_config.py`.
*   **Dynamic Loading:** The `src/main.py` script dynamically loads the RAG modules based on the `config.json` file.
*   **SOLID Principles:** The code aims to follow SOLID principles, with classes and functions having single responsibilities.
*   **LangChain Expression Language (LCEL):** The RAG pipelines are built using LCEL to orchestrate the flow from question to answer.
*   **CLI:** The command-line interface is built using `Typer`.
*   **Dependencies:** Project dependencies are managed in the `requirements.txt` file.

## Testing and Evaluation

To systematically evaluate the performance of the different RAG strategies, a testing framework can be implemented. This involves creating a dataset of questions and ground-truth answers, and then using a set of metrics to score the generated answers.

### Dataset Organization

*   **Location:** Create a new directory `src/tests/` to store all testing-related files.
*   **File Format:** The dataset should be stored in a `JSONL` file (e.g., `src/tests/test_dataset.jsonl`). Each line in the file would be a JSON object representing a single test case.

### Dataset Format

Each JSON object in the `test_dataset.jsonl` file should contain the following fields:

*   `question`: The question to be asked to the RAG pipeline.
*   `ground_truth_answer`: The ideal, human-verified answer to the question. This answer should be based on the content of the source document (*Peter Pan*).
*   `ground_truth_context`: The specific excerpt(s) from the source document that contain the information needed to answer the question. This is useful for evaluating the retrieval part of the RAG pipeline.

**Example:**

```json
{"question": "Who is Tinker Bell?", "ground_truth_answer": "Tinker Bell is a fairy who is Peter Pan's companion.", "ground_truth_context": ["Tinker Bell is a common fairy..."]}
{"question": "What did Wendy do with Peter's shadow?", "ground_truth_answer": "Wendy sewed Peter's shadow back on for him.", "ground_truth_context": ["...Wendy, who was a good sewer, sewed the shadow back on."]}
```

### Evaluation Metrics

To evaluate the performance of the RAG pipelines, you can use the following metrics. These can be implemented using libraries like `ragas` or by creating custom evaluation scripts.

*   **Answer Relevancy:** This metric measures how relevant the generated answer is to the question. It is scored on a scale of 0 to 1, where 1 is the most relevant. This can be evaluated by an LLM that checks if the answer directly addresses the question.

*   **Faithfulness:** This metric measures how factually accurate the generated answer is, based on the retrieved context. It is scored on a scale of 0 to 1, where 1 means the answer is completely supported by the context. This helps to identify if the RAG pipeline is hallucinating.

*   **Context Precision:** This metric evaluates the quality of the retrieved context. It measures the signal-to-noise ratio in the retrieved context. It is scored on a scale of 0 to 1, where 1 means all of the retrieved context is relevant to answering the question.

*   **Context Recall:** This metric measures the ability of the retriever to retrieve all the necessary information needed to answer the question. It is scored on a scale of 0 to 1, where 1 means all the relevant context was retrieved.

*   **Answer Correctness:** This metric measures the accuracy of the generated answer when compared to the ground-truth answer. It can be evaluated by an LLM that compares the generated answer with the ground-truth answer for semantic similarity.

By implementing this testing framework, you can systematically evaluate and compare the performance of different RAG strategies, and make data-driven decisions on which strategy is best suited for your use case.