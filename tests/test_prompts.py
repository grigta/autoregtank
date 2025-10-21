from pathlib import Path

from selftalk.prompts import resolve_prompt_input, build_initial_messages


def test_resolve_prompt_input_reads_file(tmp_path: Path):
    p = tmp_path / "prompt.txt"
    p.write_text("You are a helpful assistant.", encoding="utf-8")
    content = resolve_prompt_input(str(p))
    assert content == "You are a helpful assistant."


def test_resolve_prompt_input_inline_text():
    content = resolve_prompt_input("Act professionally.")
    assert content == "Act professionally."


def test_build_initial_messages():
    msgs = build_initial_messages("System prompt", "Help me with X")
    assert msgs[0].role == "system"
    assert msgs[0].content == "System prompt"
    assert msgs[1].role == "user"
    assert msgs[1].content == "Help me with X"
