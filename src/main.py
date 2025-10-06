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


if __name__ == "__main__":
    app()