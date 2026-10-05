"""Groq LLM wrapper with per-request token accounting."""
import contextvars
import json
import re

from groq import Groq

from core.config import settings

usage_var: contextvars.ContextVar = contextvars.ContextVar("usage", default=None)
_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def chat(messages, temperature=0.1, json_mode=False, max_tokens=1024, model=None) -> str:
    kw = {"response_format": {"type": "json_object"}} if json_mode else {}
    r = _get_client().chat.completions.create(
        model=model or settings.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kw,
    )
    u = usage_var.get()
    if u is not None and r.usage:
        u["tokens"] += r.usage.total_tokens
        u["calls"] += 1
    return r.choices[0].message.content


def chat_json(messages, **kw) -> dict:
    """Ask for JSON (prompts must mention 'JSON'); tolerate stray text."""
    raw = chat(messages, json_mode=True, **kw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else {}
