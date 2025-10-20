from selftalk.engine import SelfTalkEngine, EngineConfig
from selftalk.models import ChatResponse, ChatChoice, Message


class FakeClient:
    def __init__(self, outputs, model: str = "mistral-large-latest"):
        self._outputs = outputs
        self.model = model
        self.calls = 0

    def chat(self, request):
        if self.calls >= len(self._outputs):
            raise AssertionError("No more fake outputs left")
        content = self._outputs[self.calls]
        self.calls += 1
        return ChatResponse(
            id="chatcmpl_fake",
            object="chat.completion",
            created=0,
            model=self.model,
            choices=[ChatChoice(index=0, message=Message(role="assistant", content=content))],
        )


def test_engine_debate_flow():
    outputs = [
        # round 1
        "Pro 1", "Con 1",
        # round 2
        "Pro 2", "Con 2",
        # final synthesis
        "Final answer",
    ]
    fake = FakeClient(outputs)
    engine = SelfTalkEngine(client=fake)
    cfg = EngineConfig(
        system_prompt="You are helpful.",
        user_goal="Sum up topic.",
        iterations=2,
        model=fake.model,
        mode="debate",
    )
    final, transcript = engine.run(cfg)

    assert final == "Final answer"
    # Entries: system + user = 2, per round: 4, final judge: 2 => 2 + 4*2 + 2 = 12
    assert len(transcript) == 12
