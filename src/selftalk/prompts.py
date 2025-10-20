from __future__ import annotations

import os
from typing import Optional

from .models import Message


CRITIC_INSTRUCTION = (
    "You are an expert critic. Review the assistant's last answer. "
    "Provide clear, actionable feedback focused on correctness, completeness, clarity, and structure. "
    "Return a concise bullet list of issues and suggestions."
)

REVISE_INSTRUCTION = (
    "Revise your earlier answer using the critic feedback above. "
    "Keep the best parts, fix errors, fill gaps, and improve clarity. "
    "Only output the revised answer, no commentary."
)

DEBATE_PRO_INSTRUCTION = (
    "You are Agent Pro. Argue for the best possible answer to the user's goal. "
    "Provide reasoning and a concrete candidate answer."
)

DEBATE_CON_INSTRUCTION = (
    "You are Agent Con. Critique the current proposal rigorously. "
    "Find flaws, edge cases, and risks, and propose fixes if possible."
)

DEBATE_FINAL_INSTRUCTION = (
    "Act as a fair judge. Synthesize the debate into the best possible final answer for the user. "
    "Be concise and directly provide the answer."
)


def resolve_prompt_input(system_prompt_input: str) -> str:
    # If it's a file path that exists, read it; otherwise treat as literal content
    if os.path.isfile(system_prompt_input):
        with open(system_prompt_input, "r", encoding="utf-8") as f:
            return f.read().strip()
    return system_prompt_input


def build_initial_messages(system_prompt: str, user_goal: Optional[str]) -> list[Message]:
    messages: list[Message] = [Message(role="system", content=system_prompt)]
    if user_goal:
        messages.append(Message(role="user", content=user_goal))
    return messages
