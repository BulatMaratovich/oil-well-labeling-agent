from __future__ import annotations

import json
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import settings


class MistralChatClient:
    """Thin adapter that exposes the .chat() interface expected by pipeline agents."""

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        timeout: Optional[float] = None,
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        payload.update(extra)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            url=f"{settings.mistral_api_base.rstrip('/')}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {settings.mistral_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        resolved_timeout = timeout if timeout is not None else settings.mistral_timeout_sec
        try:
            with urlopen(request, timeout=resolved_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Mistral API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Mistral API network error: {exc.reason}") from exc
