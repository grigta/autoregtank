from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from dotenv import load_dotenv

from .client import MissingAPIKeyError
from .engine import SelfTalkEngine, EngineConfig, write_transcript_jsonl
from .prompts import resolve_prompt_input

app = typer.Typer(add_completion=False, help="Self-dialogue generator using Mistral API")
console = Console()


@app.command()
def run(
    system_prompt: str = typer.Option(..., "--system-prompt", help="System prompt text or a file path"),
    goal: Optional[str] = typer.Option(None, "--goal", help="Optional user goal or input"),
    iterations: int = typer.Option(3, "--iterations", min=1, help="Number of self-dialogue iterations"),
    mode: str = typer.Option("critic", "--mode", help="Mode: critic or debate", case_sensitive=False),
    model: str = typer.Option("mistral-large-latest", "--model", help="Mistral model name"),
    temperature: float = typer.Option(0.3, "--temperature", min=0.0, max=2.0, help="Sampling temperature"),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Max tokens for the response"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Nucleus sampling top_p"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Optional random seed for deterministic responses"),
    out: Path = typer.Option(Path("transcript.jsonl"), "--out", help="Path to save transcript JSONL"),
    result: Path = typer.Option(Path("result.txt"), "--result", help="Path to save final result"),
):
    """Run the self-dialogue engine and save transcript/result."""
    load_dotenv()

    if iterations < 1:
        raise typer.BadParameter("iterations must be >= 1")

    mode = mode.lower()
    if mode not in {"critic", "debate"}:
        raise typer.BadParameter("mode must be 'critic' or 'debate'")

    system_prompt_text = resolve_prompt_input(system_prompt)

    cfg = EngineConfig(
        system_prompt=system_prompt_text,
        user_goal=goal,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        random_seed=seed,
        iterations=iterations,
        mode=mode,
    )

    engine = SelfTalkEngine()

    try:
        final, transcript = engine.run(cfg)
    except MissingAPIKeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Unexpected error:[/red] {e}")
        raise typer.Exit(code=1)

    # Ensure parent directories exist
    out.parent.mkdir(parents=True, exist_ok=True)
    result.parent.mkdir(parents=True, exist_ok=True)

    write_transcript_jsonl(transcript, str(out))

    with open(result, "w", encoding="utf-8") as f:
        f.write(final)

    console.print(f"Saved transcript to [bold]{out}[/bold] and result to [bold]{result}[/bold]")


if __name__ == "__main__":
    app()
