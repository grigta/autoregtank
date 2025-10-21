from typing import List

from selftalk.engine import SelfTalkEngine, EngineConfig, write_transcript_jsonl
from selftalk.models import ChatResponse, ChatChoice, Message


class FakeClient:
    def __init__(self, outputs: List[str], model: str = "mistral-large-latest"):
        self._outputs = outputs
        self.model = model
        self.calls = 0

    def chat(self, request):  # request is ChatRequest
        # Ignore request contents; return next output deterministically
        if self.calls >= len(self._outputs):
            raise AssertionError("No more fake outputs left")
        content = self._outputs[self.calls]
        self.calls += 1
        resp = ChatResponse(
            id="chatcmpl_fake",
            object="chat.completion",
            created=0,
            model=self.model,
            choices=[
                ChatChoice(index=0, message=Message(role="assistant", content=content))
            ],
        )
        return resp


def test_engine_critic_flow(tmp_path):
    outputs = [
        "Draft answer 1",
        "Critic feedback 1",
        "Revised answer 1",
        "Critic feedback 2",
        "Revised answer 2",
    ]
    fake = FakeClient(outputs)
    engine = SelfTalkEngine(client=fake)

    cfg = EngineConfig(
        system_prompt="You are helpful.",
        user_goal="Explain gravity.",
        iterations=2,
        model=fake.model,
    )

    final, transcript = engine.run(cfg)

    assert final == "Revised answer 2"

    # Initial system+user (2) + initial draft (1) + (critic+assist+revise+assist)*2 => 3 + 4*2 = 11
    assert len(transcript) == 11

    # Verify transcript JSONL writing
    out = tmp_path / "t.jsonl"
    write_transcript_jsonl(transcript, str(out))
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(transcript)

    # Check required keys in a sample entry
    import json

    entry = json.loads(lines[-1])
    for key in ("ts", "role", "content", "model", "iteration"):
        assert key in entry
