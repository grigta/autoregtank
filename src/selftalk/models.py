from __future__ import annotations

from typing import Literal, Optional, List
from pydantic import BaseModel, Field


Role = Literal["system", "user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.3
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    stream: bool = False
    random_seed: Optional[int] = Field(default=None, alias="random_seed")

    class Config:
        populate_by_name = True


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[dict] = None

    def first_message_content(self) -> str:
        if not self.choices:
            return ""
        return self.choices[0].message.content
