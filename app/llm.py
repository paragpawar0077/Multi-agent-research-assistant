"""
Thin wrapper around the GROQ chat completion API so every agent calls the
same function instead of re-instantiating a client everywhere.
"""
import os
from groq import Groq

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. export GROQ_API_KEY=... before running."
            )
        _client = Groq(api_key=api_key)
    return _client


def chat(system: str, user: str, model: str | None = None, temperature: float = 0.3) -> str:
    """Single-turn system+user call, returns plain text."""
    # llama-3.3-70b-versatile is deprecated on Groq as of 2026; openai/gpt-oss-120b
    # is Groq's recommended replacement for general-purpose/reasoning workloads.
    model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    resp = _get_client().chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()