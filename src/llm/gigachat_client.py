from __future__ import annotations

import os


class GigaChatClient:
    def __init__(self) -> None:
        from langchain_gigachat import GigaChat

        credentials = os.getenv("GIGACHAT_CREDENTIALS", "")
        if not credentials:
            raise RuntimeError("Missing GIGACHAT_CREDENTIALS in .env")

        self.client = GigaChat(
            credentials=credentials,
            scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP"),
            model=os.getenv("GIGACHAT_MODEL", "GigaChat-2-Max"),
            verify_ssl_certs=False,
            profanity_check=False,
            max_tokens=int(os.getenv("GIGACHAT_MAX_TOKENS", "5000")),
            temperature=float(os.getenv("GIGACHAT_TEMPERATURE", "0.2")),
        )

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = self.client.invoke(messages)
        return response.content or ""


def build_llm_client_from_env() -> GigaChatClient:
    print("[llm] Using GigaChat-2-Max")
    return GigaChatClient()
