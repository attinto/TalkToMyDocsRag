
# Talk To My Docs (RAG Comparison Tool)

This project is a flexible and extensible tool for comparing different Retrieval-Augmented Generation (RAG) strategies. It allows you to ask a question against a source document and see how different RAG pipelines generate answers, making it easy to understand the trade-offs of each approach.

python -m src.main "How does Wendy's perception of Peter Pan change from their first meeting to the end of the story?" --all

Example with Peter Pan Book: How does Wendy's perception of Peter Pan change from their first meeting to the end of the story?

| **RAG Strategy**       | **Answer** |
|-------------------------|------------|
| **Basic RAG** | At first, Wendy sees Peter Pan as an exciting, magical, and heroic figure, eager for thrilling talks about their adventures. Over time, she notices Peter’s forgetfulness and lack of emotional growth—he cannot remember past adventures or people, like Captain Hook or Tinker Bell, showing that he never changes. By the end, Wendy realizes she must grow up while Peter remains unchanged, and she takes on a more maternal, wistful role, ultimately letting her daughter (and later granddaughter) go to Neverland in her place, accepting Peter’s unchanging nature and her own passage into adulthood. |
| **Sentence Window RAG** | Wendy's perception of Peter Pan changes from seeing him as a playful and magical friend to realizing the dangers and limitations of his world. She goes from enjoying the adventures he offers to experiencing fear and uncertainty about his intentions, ultimately leading to a different understanding of him by the end of the story. Wendy's perception shifts from a carefree friendship to a more cautious and realistic view of Peter Pan and the world he inhabits. |
| **Parent Document RAG** | Wendy's perception of Peter Pan changes from viewing him as a hero and a figure of fascination to realizing his forgetfulness and self-centeredness. She goes from admiring him for his adventurous spirit to being disappointed by his lack of memory and consideration for others. Ultimately, Wendy's view of Peter shifts from admiration to a more critical understanding of his character. |
| **Auto-merging RAG** | Wendy's perception of Peter Pan changes from seeing him as a lovely boy clad in skeleton leaves and entrancing to feeling the fierce hatred and doomed flight due to Tinker Bell's jealousy. She initially saw him as a grown-up and gnashed his teeth at her, symbolizing a shift from innocence to realization of darker aspects of his character. Wendy's perception of Peter Pan evolves from a fantastical character to someone with complexities and tensions, ultimately leading to their parting ways due to Tinker Bell's malicious actions. |



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
