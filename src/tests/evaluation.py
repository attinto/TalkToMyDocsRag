
import typer
from typing_extensions import Annotated
import json
from rich.console import Console
from rich.table import Table
import importlib
import sys
import logging
sys.path.append("src")
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Setup logger (set to DEBUG for diagnostic output)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()
console = Console()

def get_rag_config():
    """Loads the RAG configuration from config.json."""
    with open("config.json", "r") as f:
        return json.load(f)

def load_test_dataset(file_path: str):
    """Loads the test dataset from a JSONL file."""
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

@app.command()
def evaluate_rag(
    rag_type: Annotated[list[str], typer.Option("--rag-type", "-r", help="The short name of the RAG type to run (e.g., 'basic'). Can be used multiple times.")] = None,
    run_all: Annotated[bool, typer.Option("--all", help="Run all enabled RAGs defined in config.json.")] = False,
    dataset_path: str = "src/tests/test_dataset.jsonl",
):
    """
    Evaluates one or more RAG strategies against a test dataset.
    """
    config = get_rag_config()
    rags_to_run = []

    if run_all:
        rags_to_run = [key for key, val in config["rags"].items() if val.get("enabled", False)]
    elif rag_type:
        for r_type in rag_type:
            if r_type in config["rags"]:
                rags_to_run.append(r_type)
            else:
                console.print(f"[bold red]Error:[/RAG type '{r_type}' not found in config.json.")
                raise typer.Exit(1)

    if not rags_to_run:
        console.print("[bold yellow]Warning:[/No RAGs selected to run.")
        return

    dataset_rows = load_test_dataset(dataset_path)

    for rag_key in rags_to_run:
        rag_info = config["rags"][rag_key]
        rag_name = rag_info.get("name", rag_key)
        module_name = rag_info.get("module")

        logger.info(f"Evaluating RAG Strategy: {rag_name}")
        console.print(f"\n[bold]Evaluating RAG Strategy:[/]{rag_name}\n")

        if not module_name:
            logger.error(f"Module not configured for {rag_name}...")
            console.print(f"[italic red]Module not configured for {rag_name}...[/]")
            continue

        try:
            rag_module = importlib.import_module(module_name)
            logger.info(f"Successfully imported module: {module_name}")
        except ImportError:
            logger.error(f"Could not import module: {module_name}")
            console.print(f"[italic red]Could not import module: {module_name}[/]")
            continue

        results = []
        for item in dataset_rows:
            # Accept several common key names from test datasets to be more robust.
            # Priority: explicit 'question' then 'user_input' or other common variants.
            question = (
                item.get("question")
                or item.get("user_input")
                or item.get("query")
                or item.get("prompt")
            )

            # Ground truth answer can be stored under several keys depending on the dataset.
            ground_truth_answer = (
                item.get("ground_truth_answer")
                or item.get("ground_truth")
                or item.get("reference")
                or item.get("answer")
            )

            if question is None:
                logger.warning(f"Skipping dataset item without a question-like field: {item}")
                continue
            try:
                rag_output = rag_module.execute_rag(question)
                answer = rag_output["answer"]
                contexts = rag_output.get("context")
                logger.debug(f"Raw rag_output['context'] type: {type(contexts)}, repr: {repr(contexts)[:200]}")
                # Ensure contexts is a list of strings
                if contexts and len(contexts) > 0:
                    logger.debug(f"Pre-normalization contexts length: {len(contexts)}; first-type: {type(contexts[0])}")
                    if hasattr(contexts[0], "page_content"):
                        contexts = [c.page_content for c in contexts]
                    elif isinstance(contexts[0], dict) and "page_content" in contexts[0]:
                        contexts = [c["page_content"] for c in contexts]
                    else:
                        contexts = [str(c) for c in contexts]
                    logger.debug(f"Post-normalization contexts length: {len(contexts)}; sample: {contexts[0][:200] if len(contexts)>0 else None}")
                logger.debug(f"Question: {question}\nAnswer: {answer}\nContexts: {contexts}\nGround Truth: {ground_truth_answer}")
            except Exception as e:
                logger.error(f"Error during RAG execution: {e}")
                answer = f"[italic red]An error occurred: {e}[/]"
                contexts = []

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth_answer
            })

        # Convert results to ragas SingleTurnSample shape and validate
        logger.info(f"Converting results to ragas format (raw samples: {len(results)})")
        rg_samples = []
        try:
            from ragas.dataset_schema import SingleTurnSample
        except Exception:
            SingleTurnSample = None

        for i, r in enumerate(results):
            # Ensure contexts are strings
            contexts = r.get("contexts") or []
            contexts = [c if isinstance(c, str) else (c.get("page_content") if isinstance(c, dict) else str(c)) for c in contexts]

            sample = {
                "user_input": r.get("question"),
                "retrieved_contexts": contexts,
                "response": r.get("answer"),
                "reference": r.get("ground_truth"),
            }

            # Validate with ragas SingleTurnSample if available
            if SingleTurnSample is not None:
                try:
                    SingleTurnSample(**sample)
                except Exception as e:
                    logger.warning(f"Sample {i} failed validation and will be skipped: {e}")
                    continue

            rg_samples.append(sample)
        # --- Debugging: compare system retrieved_contexts against original dataset's retrieved_contexts (if provided)
        try:
            for idx, sample in enumerate(rg_samples[:5]):
                original = dataset_rows[idx] if idx < len(dataset_rows) else {}
                gold_contexts = original.get("retrieved_contexts") or original.get("gold_contexts") or []
                sys_contexts = sample.get("retrieved_contexts") or []

                # Simple substring-based overlap check
                match_counts = 0
                for sc in sys_contexts:
                    for gc in gold_contexts:
                        try:
                            if not sc or not gc:
                                continue
                            if gc in sc or sc in gc:
                                match_counts += 1
                                break
                        except Exception:
                            continue

                logger.info(f"Sample[{idx}] - system retrieved {len(sys_contexts)} vs gold {len(gold_contexts)} - substring matches: {match_counts}")
                if len(gold_contexts) and match_counts == 0:
                    logger.debug(f"Sample[{idx}] GOLD contexts: {gold_contexts}")
                    logger.debug(f"Sample[{idx}] SYS contexts: {sys_contexts}")
        except Exception as e:
            logger.debug(f"Debug comparison failed: {e}")

        logger.info(f"Validated ragas samples: {len(rg_samples)}")
        if not rg_samples:
            logger.error("No valid ragas samples to evaluate. Skipping.")
            console.print(f"[bold red]Error:[/] No valid ragas samples to evaluate for {rag_name}.")
            continue

        # Create a HuggingFace dataset from validated ragas samples
        logger.info(f"Creating HuggingFace Dataset from ragas samples (num samples: {len(rg_samples)})")
        dataset = Dataset.from_list(rg_samples)

        # Evaluate the dataset
        logger.info(f"Running evaluation for {rag_name}")
        result = evaluate(
            dataset=dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy,
            ],
        )

        logger.info(f"Evaluation result type: {type(result)}")
        logger.info(f"Evaluation result dir: {dir(result)}")
        logger.info(f"Evaluation result repr: {repr(result)}")

        table = Table(title=f"{rag_name} Evaluation Results")
        table.add_column("Metric", justify="left", style="cyan")
        table.add_column("Score", justify="left", style="magenta")

        # Robust metrics extraction: try several known attributes and fallbacks
        metrics_dict = None
        tried_attrs = []
        for attr in ("metrics", "scores", "_scores_dict"):
            tried_attrs.append(attr)
            val = getattr(result, attr, None)
            if val:
                metrics_dict = val
                logger.info(f"Found metrics under attribute '{attr}'")
                break

        # If result itself is a plain dict-like object
        if metrics_dict is None and isinstance(result, dict):
            metrics_dict = result
            logger.info("Result is a dict; using it as metrics_dict")

        # If there's a to_pandas method, try converting
        if metrics_dict is None and hasattr(result, "to_pandas") and callable(getattr(result, "to_pandas")):
            try:
                df = result.to_pandas()
                # Expect df to have one row with metric columns
                if not df.empty:
                    # Convert first row to dict
                    metrics_dict = dict(df.iloc[0].to_dict())
                    logger.info("Extracted metrics from to_pandas() output")
            except Exception as e:
                logger.debug(f"to_pandas() exists but failed: {e}")

        # If metrics_dict is still not a plain dict, try to coerce
        if metrics_dict is not None and not isinstance(metrics_dict, dict):
            try:
                metrics_dict = dict(metrics_dict)
            except Exception:
                # last resort: check for internal repr dict
                metrics_dict = getattr(result, "_repr_dict", None) or getattr(result, "_scores_dict", None)

        # Final handling: display metrics if available
        if isinstance(metrics_dict, dict) and metrics_dict:
            for metric, score in metrics_dict.items():
                # Safely format score as float if possible
                try:
                    table.add_row(metric, f"{float(score):.4f}")
                except Exception:
                    table.add_row(metric, str(score))
        else:
            logger.warning(f"No metrics found in result after checking attrs {tried_attrs}.")
            table.add_row("Error", "No metrics found in result")
            logger.info(f"Result object: {result}")

        console.print(table)

if __name__ == "__main__":
    app()