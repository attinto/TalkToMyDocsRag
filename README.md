
# Talk To My Docs (RAG Comparison Tool)

This project is a flexible and extensible tool for comparing different Retrieval-Augmented Generation (RAG) strategies. It allows you to ask a question against a source document and see how different RAG pipelines generate answers, making it easy to understand the trade-offs of each approach.

## Features

- **Multiple RAG Strategies**: Implements four distinct RAG pipelines out-of-the-box.
- **Flexible CLI**: A command-line interface built with Typer to easily run and compare one, many, or all strategies at once.
- **Modular Architecture**: Designed with SOLID principles, making it easy to add new RAG strategies.
- **Side-by-Side Comparison**: Displays results in a clean, formatted table for easy analysis.

## Implemented RAG Strategies

1.  **Basic RAG**: The simplest approach. It retrieves large, fixed-size chunks of text that are relevant to the question.
2.  **Sentence Window RAG**: A more precise method. It retrieves the single most relevant sentence and expands the context by including a few sentences before and after it (the "window").
3.  **Parent Document RAG**: A hybrid approach. It searches over small, precise text chunks but retrieves their larger parent documents to provide the LLM with rich, complete context.
4.  **Auto-merging RAG**: The most advanced strategy here. It retrieves many small chunks and, if several come from the same parent document, it "merges" them by retrieving the larger parent document instead. This is excellent for questions that require synthesizing information from multiple places.

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

## How to Use

The main entry point is `src/main.py`. You can run it as a module.

**Base Command:**
`python -m src.main "<Your Question>" [OPTIONS]`

**Options:**
-   `-r`, `--rag-type TEXT`: Specify a RAG strategy to run (e.g., `basic`, `sentence`, `parent`, `merging`). Can be used multiple times.
-   `--all`: Run all enabled RAG strategies defined in `config.json`.

### Examples

**Run a single RAG strategy:**
```bash
python -m src.main "Who is Tinker Bell?" -r basic
```

**Compare two specific strategies:**
```bash
python -m src.main "What did Wendy do with Peter's shadow?" -r basic -r sentence
```

**Run all strategies for a complex question:**
```bash
python -m src.main "How does Wendy's perception of Peter Pan change from their first meeting to the end of the story?" --all
```

## Example Output

Below is an example of the output when running the tool with all strategies.

> <!-- PASTE YOUR TERMINAL OUTPUT HERE -->

