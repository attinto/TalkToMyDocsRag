# RAG Strategies Evaluation Report

*Generated using RAGAS evaluation framework*

## Overall Metrics Comparison

|           |   faithfulness |   answer_relevancy |   context_precision |   context_recall |
|:----------|---------------:|-------------------:|--------------------:|-----------------:|
| Basic RAG |          0.892 |               0.96 |                   1 |            0.583 |

## Metrics Interpretation

Each metric is scored from 0 to 1, where higher scores are better:

- **Faithfulness** (0-1): Measures if the answer is factually consistent with the retrieved context.
  - Detects hallucinations and unsupported claims
  - High score = answer is well-grounded in the context

- **Answer Relevancy** (0-1): Measures how well the answer addresses the question.
  - Checks if the response is on-topic and complete
  - High score = answer directly and fully addresses the question

- **Context Precision** (0-1): Measures if the retrieved contexts are relevant to answering the question.
  - Lower precision means more irrelevant documents were retrieved
  - High score = retrieval returned relevant, useful context

- **Context Recall** (0-1): Measures if all necessary information from the ground truth was retrieved.
  - Lower recall means important information was missed
  - High score = retrieval captured all relevant information

## Evaluation Dataset

- **Number of test questions**: 6
- **Question types**: factual: 2, analytical: 2, multi-hop: 2
- **Difficulty levels**: easy: 1, medium: 3, hard: 2

## How RAGAS Works

RAGAS uses LLM-based evaluation to assess RAG pipeline quality. Unlike traditional metrics,
it can understand semantic similarity and factual consistency, providing more nuanced
evaluation than simple string matching.

The evaluation process:
1. For each test question, run the RAG pipeline to get an answer and contexts
2. Compare the generated answer with the ground truth using LLM-based evaluation
3. Assess the quality of retrieved contexts against ground truth contexts
4. Aggregate scores across all test questions to get overall metrics
