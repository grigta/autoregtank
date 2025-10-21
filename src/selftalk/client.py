from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

import httpx
from pydantic import ValidationError

from .models import ChatRequest, ChatResponse


class MistralAPIError(Exception):
    pass


class MissingAPIKeyError(MistralAPIError):
    pass


class MistralClient:
    BASE_URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 30.0):
        # Support .env for local use
        if api_key is None:
            api_key = os.getenv("MISTRAL_API_KEY")
        self.api_key = api_key
        self.timeout = timeout
        self._http = httpx.Client(timeout=self.timeout)

    def ensure_api_key(self) -> None:
        if not self.api_key:
            raise MissingAPIKeyError(
                "MISTRAL_API_KEY is not set. Please set it in your environment or a .env file."
            )

    def _headers(self) -> dict:
        self.ensure_api_key()
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(self, request: ChatRequest) -> ChatResponse:
        payload = request.model_dump(by_alias=True)
        return self._post_with_backoff(payload)

    def _post_with_backoff(self, json_body: dict, max_retries: int = 5) -> ChatResponse:
        backoff = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._http.post(self.BASE_URL, headers=self._headers(), json=json_body)
                if resp.status_code == 429 or 500 <= resp.status_code < 600:
                    # Backoff on rate limit / server error
                    delay = backoff * (1 + random.random() * 0.25)
                    time.sleep(delay)
                    backoff = min(backoff * 2, 16)
                    continue
                resp.raise_for_status()
                data = resp.json()
                try:
                    return ChatResponse.model_validate(data)
                except ValidationError as ve:
                    raise MistralAPIError(f"Invalid response schema from Mistral: {ve}")
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exc = e
                # Only backoff on transient server errors; otherwise bail out
                if isinstance(e, httpx.HTTPStatusError):
                    status = e.response.status_code if e.response is not None else None
                    if status is not None and status < 500 and status != 429:
                        raise MistralAPIError(f"HTTP error from Mistral: {status} {e}")
                if attempt >= max_retries:
                    break
                delay = backoff * (1 + random.random() * 0.25)
                time.sleep(delay)
                backoff = min(backoff * 2, 16)
        if last_exc:
            raise MistralAPIError(f"Failed to call Mistral after retries: {last_exc}")
        raise MistralAPIError("Failed to call Mistral: unknown error")
