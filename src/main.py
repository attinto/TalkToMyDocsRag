import typer
from typing_extensions import Annotated
import json
import importlib
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()

def get_rag_config():
    """Loads the RAG configuration from config.json."""
    with open("config.json", "r") as f:
        return json.load(f)

@app.command()
def run(
    question: Annotated[str, typer.Argument(help="The question you want to ask.")],
    rag_type: Annotated[list[str], typer.Option("--rag-type", "-r", help="The short name of the RAG type to run (e.g., 'basic'). Can be used multiple times.")] = None,
    run_all: Annotated[bool, typer.Option("--all", help="Run all enabled RAGs defined in config.json.")] = False
):
    """
    Answers a question using one or more RAG strategies.
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
                console.print(f"[bold red]Error:[/] RAG type '{r_type}' not found in config.json.")
                raise typer.Exit(1)
    else:
        console.print("[bold red]Error:[/] Please specify at least one RAG type with '--rag-type' or use '--all'.")
        raise typer.Exit(1)

    if not rags_to_run:
        console.print("[bold yellow]Warning:[/] No RAGs selected to run.")
        return

    table = Table(title="RAG Comparison Results")
    table.add_column("RAG Strategy", justify="left", style="cyan", no_wrap=True)
    table.add_column("Answer", justify="left", style="magenta")

    console.print(f"\n[bold]Answering question:[/] {question}\n")

    num_rags_to_run = len(rags_to_run)
    for i, rag_key in enumerate(rags_to_run):
        rag_info = config["rags"][rag_key]
        rag_name = rag_info.get("name", rag_key)
        module_name = rag_info.get("module")
        is_last_row = (i == num_rags_to_run - 1)
        answer = ""

        if not module_name:
            answer = "[italic red]Module not configured...[/]"
        else:
            try:
                # Dynamically import the module and execute the RAG
                rag_module = importlib.import_module(module_name)
                answer = rag_module.execute_rag(question)
            except ImportError:
                answer = f"[italic red]Could not import module: {module_name}[/]"
            except Exception as e:
                answer = f"[italic red]An error occurred: {e}[/]"

        # Add a separator line after every row except the last one.
        table.add_row(rag_name, answer, end_section=not is_last_row)

    console.print(table)


@app.command()
def evaluate(
    strategies: Annotated[list[str], typer.Option("--strategy", "-s", help="RAG strategies to evaluate. Can be used multiple times.")] = None,
    all_strategies: Annotated[bool, typer.Option("--all", help="Evaluate all enabled strategies.")] = False,
    output: Annotated[str, typer.Option("--output", "-o", help="Output file for evaluation report.")] = "evaluation_report.md"
):
    """
    Evaluate RAG strategies using the evaluation dataset.
    
    This command runs each RAG strategy against the test questions in the evaluation
    dataset and measures performance using RAGAS metrics (faithfulness, answer relevancy,
    context precision, and context recall).
    """
    from src.evaluation.rag_evaluator import RAGEvaluator
    
    config = get_rag_config()
    
    # Determine which strategies to evaluate
    strategies_to_eval = []
    if all_strategies:
        strategies_to_eval = [key for key, val in config["rags"].items() if val.get("enabled", False)]
    elif strategies:
        for s_type in strategies:
            if s_type in config["rags"]:
                strategies_to_eval.append(s_type)
            else:
                console.print(f"[bold red]Error:[/] Strategy '{s_type}' not found in config.json.")
                raise typer.Exit(1)
    else:
        console.print("[bold red]Error:[/] Please specify strategies with '--strategy' or use '--all'.")
        raise typer.Exit(1)
    
    if not strategies_to_eval:
        console.print("[bold yellow]Warning:[/] No strategies selected to evaluate.")
        raise typer.Exit(1)
    
    # Build executor functions for each strategy
    rag_executors = {}
    
    for strategy_key in strategies_to_eval:
        rag_info = config["rags"][strategy_key]
        rag_name = rag_info.get("name", strategy_key)
        module_name = rag_info.get("module")
        
        if not module_name:
            console.print(f"[bold yellow]Warning:[/] No module configured for '{strategy_key}'. Skipping.")
            continue
        
        try:
            rag_module = importlib.import_module(module_name)
            
            # Each RAG module should have an execute_rag function
            executor = getattr(rag_module, "execute_rag", None)
            
            # Try to get the function that returns contexts too (optional)
            executor_with_context = getattr(rag_module, "execute_rag_with_context", None)
            
            if executor:
                rag_executors[rag_name] = {
                    "executor": executor,
                    "executor_with_context": executor_with_context
                }
            else:
                console.print(f"[bold yellow]Warning:[/] Module '{module_name}' has no execute_rag function. Skipping.")
                
        except ImportError as e:
            console.print(f"[bold red]Error:[/] Could not import module '{module_name}': {e}")
        except Exception as e:
            console.print(f"[bold red]Error:[/] Unexpected error with '{strategy_key}': {e}")
    
    if not rag_executors:
        console.print("[bold red]Error:[/] No valid strategies to evaluate.")
        raise typer.Exit(1)
    
    # Run evaluation
    console.print(f"\n[bold cyan]Starting evaluation of {len(rag_executors)} strategies...[/]")
    console.print(f"[dim]This may take several minutes as each strategy processes multiple questions.[/]\n")
    
    try:
        evaluator = RAGEvaluator()
        results_df = evaluator.compare_strategies(rag_executors)
        
        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]📊 Evaluation Complete![/]")
        console.print("="*60 + "\n")
        
        # Check if results_df is empty
        if results_df.empty:
            console.print("[bold red]No valid evaluation results were produced.[/]")
        else:
            try:
                # Format the DataFrame for display
                console.print(results_df.round(3).to_string())
            except Exception as e:
                console.print(f"[bold yellow]Warning: Could not display results as table: {e}[/]")
                console.print("Raw results:")
                console.print(str(results_df))
        
        console.print("\n")
        
        # Generate report
        try:
            evaluator.generate_report(results_df, output)
            console.print(f"[bold green]📄 Full report saved to:[/] {output}\n")
        except Exception as e:
            console.print(f"[bold red]Error generating report: {e}[/]")
        
    except Exception as e:
        console.print(f"\n[bold red]Error during evaluation:[/] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()