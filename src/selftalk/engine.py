from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

from rich.console import Console
from rich.progress import track

from .client import MistralClient, MissingAPIKeyError
from .models import Message, ChatRequest
from .prompts import (
    build_initial_messages,
    CRITIC_INSTRUCTION,
    REVISE_INSTRUCTION,
    DEBATE_CON_INSTRUCTION,
    DEBATE_FINAL_INSTRUCTION,
    DEBATE_PRO_INSTRUCTION,
)

console = Console()


@dataclass
class EngineConfig:
    system_prompt: str
    user_goal: Optional[str]
    model: str = "mistral-large-latest"
    temperature: float = 0.3
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = None
    random_seed: Optional[int] = None
    iterations: int = 3
    mode: str = "critic"  # or "debate"


TranscriptEntry = Dict[str, Any]


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_transcript(transcript: List[TranscriptEntry], role: str, content: str, model: str, iteration: Optional[int]) -> None:
    transcript.append({
        "ts": _ts(),
        "role": role,
        "content": content,
        "model": model,
        "iteration": iteration,
    })


def write_transcript_jsonl(transcript: List[TranscriptEntry], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in transcript:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class SelfTalkEngine:
    def __init__(self, client: Optional[MistralClient] = None):
        self.client = client or MistralClient()

    def run(self, cfg: EngineConfig) -> Tuple[str, List[TranscriptEntry]]:
        mode = cfg.mode.lower()
        if mode == "critic":
            return self._run_critic(cfg)
        elif mode == "debate":
            return self._run_debate(cfg)
        else:
            raise ValueError(f"Unknown mode: {cfg.mode}")

    def _chat(self, messages: List[Message], cfg: EngineConfig) -> str:
        req = ChatRequest(
            model=cfg.model,
            messages=messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            top_p=cfg.top_p,
            random_seed=cfg.random_seed,
        )
        resp = self.client.chat(req)
        return resp.first_message_content()

    def _run_critic(self, cfg: EngineConfig) -> Tuple[str, List[TranscriptEntry]]:
        messages = build_initial_messages(cfg.system_prompt, cfg.user_goal)
        transcript: List[TranscriptEntry] = []
        for m in messages:
            _append_transcript(transcript, m.role, m.content, cfg.model, iteration=None)

        console.rule("Solver: initial draft")
        draft = self._chat(messages, cfg)
        messages.append(Message(role="assistant", content=draft))
        _append_transcript(transcript, "assistant", draft, cfg.model, iteration=0)

        for i in track(range(1, cfg.iterations + 1), description="Critique and revise"):
            console.rule(f"Critic: feedback round {i}")
            critic_prompt = (
                "Critic: " + CRITIC_INSTRUCTION
            )
            messages.append(Message(role="user", content=critic_prompt))
            _append_transcript(transcript, "user", critic_prompt, cfg.model, iteration=i)
            critic_feedback = self._chat(messages, cfg)
            messages.append(Message(role="assistant", content=critic_feedback))
            _append_transcript(transcript, "assistant", critic_feedback, cfg.model, iteration=i)

            console.rule(f"Solver: revision round {i}")
            revise_prompt = (
                "Reviser: " + REVISE_INSTRUCTION
            )
            messages.append(Message(role="user", content=revise_prompt))
            _append_transcript(transcript, "user", revise_prompt, cfg.model, iteration=i)
            revised = self._chat(messages, cfg)
            messages.append(Message(role="assistant", content=revised))
            _append_transcript(transcript, "assistant", revised, cfg.model, iteration=i)

        final_answer = messages[-1].content if messages else ""
        return final_answer, transcript

    def _run_debate(self, cfg: EngineConfig) -> Tuple[str, List[TranscriptEntry]]:
        messages = build_initial_messages(cfg.system_prompt, cfg.user_goal)
        transcript: List[TranscriptEntry] = []
        for m in messages:
            _append_transcript(transcript, m.role, m.content, cfg.model, iteration=None)

        for i in track(range(1, cfg.iterations + 1), description="Pro/Con debate"):
            # Pro argues
            pro_prompt = "Agent(Pro): " + DEBATE_PRO_INSTRUCTION
            messages.append(Message(role="user", content=pro_prompt))
            _append_transcript(transcript, "user", pro_prompt, cfg.model, iteration=i)
            pro_msg = self._chat(messages, cfg)
            messages.append(Message(role="assistant", content=pro_msg))
            _append_transcript(transcript, "assistant", pro_msg, cfg.model, iteration=i)

            # Con responds
            con_prompt = "Agent(Con): " + DEBATE_CON_INSTRUCTION
            messages.append(Message(role="user", content=con_prompt))
            _append_transcript(transcript, "user", con_prompt, cfg.model, iteration=i)
            con_msg = self._chat(messages, cfg)
            messages.append(Message(role="assistant", content=con_msg))
            _append_transcript(transcript, "assistant", con_msg, cfg.model, iteration=i)

        # Final synthesis
        final_prompt = "Judge: " + DEBATE_FINAL_INSTRUCTION
        messages.append(Message(role="user", content=final_prompt))
        _append_transcript(transcript, "user", final_prompt, cfg.model, iteration=cfg.iterations)
        final_answer = self._chat(messages, cfg)
        messages.append(Message(role="assistant", content=final_answer))
        _append_transcript(transcript, "assistant", final_answer, cfg.model, iteration=cfg.iterations)

        return final_answer, transcript
