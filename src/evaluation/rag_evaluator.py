"""
RAG Evaluation Module
--------------------

This module provides functionality to evaluate different RAG strategies
using the RAGAS framework and a custom evaluation dataset.

RAGAS (Retrieval Augmented Generation Assessment) is a framework specifically 
designed for evaluating RAG pipelines. It provides several key metrics:

1. **Faithfulness**: Measures if the answer is factually consistent with the 
   retrieved context. This ensures the LLM isn't hallucinating facts.
   
2. **Answer Relevancy**: Measures how well the answer addresses the question.
   Even a factually correct answer can be irrelevant if it doesn't answer what was asked.
   
3. **Context Precision**: Measures whether the retrieved contexts are relevant 
   to the question. High precision means less noise in retrieval.
   
4. **Context Recall**: Measures if all the necessary information from the ground 
   truth was retrieved. High recall means we're not missing important context.

Each metric returns a score between 0 and 1, where higher is better.
"""

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from typing import Dict, List, Any, Callable
import os


class RAGEvaluator:
    """
    Evaluates RAG strategies using RAGAS metrics.
    
    This class takes your evaluation dataset and runs each RAG strategy
    against it, measuring performance across multiple dimensions.
    """

    def __init__(self, eval_dataset_path: str = "src/data/eval_dataset.json"):
        """
        Initialize the evaluator with an evaluation dataset.

        Args:
            eval_dataset_path: Path to the evaluation dataset JSON file
        """
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.eval_data = self._load_eval_dataset(eval_dataset_path)
        
        # Define which metrics to use
        # Note: RAGAS uses an LLM (GPT-3.5 or GPT-4) to evaluate the outputs
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]

    def _load_eval_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load the evaluation dataset from JSON."""
        with open(path, "r") as f:
            return json.load(f)

    def evaluate_rag_strategy(
        self, 
        rag_executor_func: Callable[[str], str],
        rag_executor_with_context_func: Callable[[str], Dict[str, Any]],
        strategy_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a single RAG strategy.

        Args:
            rag_executor_func: Function that takes a question and returns an answer (str)
            rag_executor_with_context_func: Function that takes a question and returns 
                                           {"answer": str, "contexts": List[str]}
            strategy_name: Name of the RAG strategy being evaluated

        Returns:
            Dictionary of metric scores
        """
        print(f"\n{'='*60}")
        print(f"Evaluating: {strategy_name}")
        print(f"{'='*60}\n")

        # Prepare data for RAGAS
        questions = []
        answers = []
        ground_truths = []
        contexts = []

        for i, item in enumerate(self.eval_data, 1):
            question = item["question"]
            print(f"Processing question {i}/{len(self.eval_data)}: {question[:50]}...")
            
            questions.append(question)
            ground_truths.append(item["ground_truth_answer"])

            # Get answer and contexts from RAG
            try:
                # Try to get answer with contexts
                if rag_executor_with_context_func:
                    result = rag_executor_with_context_func(question)
                    answers.append(result.get("answer", ""))
                    contexts.append(result.get("contexts", item["ground_truth_context"]))
                else:
                    # Fallback to just getting answer
                    answer = rag_executor_func(question)
                    answers.append(answer)
                    # Use ground truth context as fallback
                    contexts.append(item["ground_truth_context"])
                
            except Exception as e:
                print(f"  ⚠️  Error processing question: {e}")
                answers.append("Error generating answer")
                contexts.append(item["ground_truth_context"])

        # Create dataset for RAGAS
        # RAGAS expects a very specific format
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,  # List of lists of context strings
            "ground_truth": ground_truths,
        })

        print(f"\n📊 Running RAGAS evaluation...")
        
        # Run evaluation
        result = evaluate(
            eval_dataset,
            metrics=self.metrics,
        )

        print(f"✅ Evaluation complete for {strategy_name}\n")
        
        # Convert result to dictionary with average scores
        # RAGAS returns lists of scores (one per question), so we need to average them
        result_dict = {}
        for metric in self.metrics:
            scores = result[metric.name]
            # Filter out None values or zeros that might indicate evaluation errors
            valid_scores = [score for score in scores if score is not None and score > 0]
            if valid_scores:
                # Calculate the average score
                avg_score = sum(valid_scores) / len(valid_scores)
                result_dict[metric.name] = avg_score
            else:
                # If no valid scores, return 0
                result_dict[metric.name] = 0.0
        
        return result_dict

    def compare_strategies(
        self, 
        rag_strategies: Dict[str, Dict[str, Callable]]
    ) -> pd.DataFrame:
        """
        Compare multiple RAG strategies side by side.

        Args:
            rag_strategies: Dictionary mapping strategy names to a dict containing:
                           - "executor": function that returns answer string
                           - "executor_with_context": function that returns {answer, contexts}

        Returns:
            DataFrame with comparison results
        """
        results = {}

        for strategy_name, executors in rag_strategies.items():
            executor = executors.get("executor")
            executor_with_context = executors.get("executor_with_context")
            
            result = self.evaluate_rag_strategy(
                executor, 
                executor_with_context,
                strategy_name
            )
            results[strategy_name] = result

        # Convert to DataFrame for easy comparison
        df = pd.DataFrame(results).T
        
        # Round to 3 decimal places for readability
        df = df.round(3)
        
        return df

    def generate_report(
        self, 
        results_df: pd.DataFrame, 
        output_path: str = "evaluation_report.md"
    ):
        """
        Generate a markdown report of the evaluation results.

        Args:
            results_df: DataFrame containing evaluation results
            output_path: Path to save the markdown report
        """
        # Print the dataframe for debugging
        print("\nEvaluation Results DataFrame:")
        print(results_df)
        print("\n")
        
        with open(output_path, "w") as f:
            f.write("# RAG Strategies Evaluation Report\n\n")
            f.write(f"*Generated using RAGAS evaluation framework*\n\n")
            
            f.write("## Overall Metrics Comparison\n\n")
            f.write(results_df.to_markdown())
            f.write("\n\n")
            
            # Add best performers if we have multiple strategies
            if len(results_df) > 1:
                f.write("## Best Performers by Metric\n\n")
                for metric in results_df.columns:
                    if results_df[metric].max() > 0:  # Only if we have valid scores
                        best_strategy = results_df[metric].idxmax()
                        best_score = results_df[metric].max()
                        f.write(f"- **{metric}**: {best_strategy} ({best_score:.3f})\n")
                f.write("\n")
            
            # Add interpretation
            f.write("## Metrics Interpretation\n\n")
            f.write("Each metric is scored from 0 to 1, where higher scores are better:\n\n")
            f.write("- **Faithfulness** (0-1): Measures if the answer is factually consistent with the retrieved context.\n")
            f.write("  - Detects hallucinations and unsupported claims\n")
            f.write("  - High score = answer is well-grounded in the context\n\n")
            
            f.write("- **Answer Relevancy** (0-1): Measures how well the answer addresses the question.\n")
            f.write("  - Checks if the response is on-topic and complete\n")
            f.write("  - High score = answer directly and fully addresses the question\n\n")
            
            f.write("- **Context Precision** (0-1): Measures if the retrieved contexts are relevant to answering the question.\n")
            f.write("  - Lower precision means more irrelevant documents were retrieved\n")
            f.write("  - High score = retrieval returned relevant, useful context\n\n")
            
            f.write("- **Context Recall** (0-1): Measures if all necessary information from the ground truth was retrieved.\n")
            f.write("  - Lower recall means important information was missed\n")
            f.write("  - High score = retrieval captured all relevant information\n\n")
            
            # Add dataset info
            f.write("## Evaluation Dataset\n\n")
            f.write(f"- **Number of test questions**: {len(self.eval_data)}\n")
            
            # Count by type and difficulty
            question_types = {}
            difficulty_levels = {}
            
            for item in self.eval_data:
                q_type = item.get("question_type", "unknown")
                difficulty = item.get("difficulty", "unknown")
                question_types[q_type] = question_types.get(q_type, 0) + 1
                difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1
            
            f.write(f"- **Question types**: {', '.join([f'{k}: {v}' for k, v in question_types.items()])}\n")
            f.write(f"- **Difficulty levels**: {', '.join([f'{k}: {v}' for k, v in difficulty_levels.items()])}\n")
            f.write("\n")
            
            f.write("## How RAGAS Works\n\n")
            f.write("RAGAS uses LLM-based evaluation to assess RAG pipeline quality. Unlike traditional metrics,\n")
            f.write("it can understand semantic similarity and factual consistency, providing more nuanced\n")
            f.write("evaluation than simple string matching.\n\n")
            
            f.write("The evaluation process:\n")
            f.write("1. For each test question, run the RAG pipeline to get an answer and contexts\n")
            f.write("2. Compare the generated answer with the ground truth using LLM-based evaluation\n")
            f.write("3. Assess the quality of retrieved contexts against ground truth contexts\n")
            f.write("4. Aggregate scores across all test questions to get overall metrics\n")
            
        print(f"\n✅ Report saved to: {output_path}")
