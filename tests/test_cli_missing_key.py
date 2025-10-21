import os
from typer.testing import CliRunner

from selftalk.cli import app


def test_cli_missing_api_key(monkeypatch):
    runner = CliRunner()
    # Ensure env var is not set
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    result = runner.invoke(
        app,
        [
            "run",
            "--system-prompt",
            "You are helpful",
            "--goal",
            "Test goal",
            "--iterations",
            "1",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "MISTRAL_API_KEY is not set" in result.stdout
