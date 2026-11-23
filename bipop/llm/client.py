"""Chat-completion backends.

* OpenAI models (``gpt-4.1``, ``o3``) use the standard OpenAI client, configured
  with ``OPENAI_API_KEY`` (and optionally ``OPENAI_BASE_URL``).
* ``deepseek-r1`` is served by OpenRouter (``OPENROUTER_API_KEY``) as
  DeepSeek-R1-0528, routed to Novita and then AtlasCloud.

API keys are only read from the environment.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI

from .log import debug_print

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# R1-0528 is the updated R1 release, more directly comparable to o3. Routing is
# pinned to Novita then AtlasCloud; both leave ample output budget for R1's
# reasoning plus the JSON answer.
OPENROUTER_R1_MODEL = "deepseek/deepseek-r1-0528"
OPENROUTER_PROVIDER_ORDER = ["novita", "atlas-cloud"]

DEEPSEEK_MODELS = {"deepseek-r1"}
REASONING_MODELS = {"o3"}  # no sampling parameters (temperature, top_p, ...)

# R1 spends a large, variable share of its budget on hidden reasoning before
# it answers; the default 4k cap truncates the answer (finish_reason "length").
# The cap must stay below the smallest routed provider's output limit (Novita:
# 32,768), otherwise OpenRouter answers "no endpoints found" (404).
DEEPSEEK_MAX_TOKENS = 32000
DEFAULT_TIMEOUT = 200
DEFAULT_RETRIES = 3


def _deepseek_backends() -> List[Dict[str, Any]]:
    """DeepSeek backends in priority order (currently OpenRouter only)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No DeepSeek backend is configured; set OPENROUTER_API_KEY "
            "before requesting a DeepSeek model."
        )
    return [
        {
            "name": "openrouter",
            "api_key": api_key,
            "base_url": OPENROUTER_BASE_URL,
            "model": OPENROUTER_R1_MODEL,
            "provider_routing": {"order": OPENROUTER_PROVIDER_ORDER, "allow_fallbacks": False},
        }
    ]


def build_client(model_name: str) -> OpenAI:
    """OpenAI-compatible client for ``model_name``."""
    if model_name in DEEPSEEK_MODELS:
        backend = _deepseek_backends()[0]
        return OpenAI(
            api_key=backend["api_key"],
            base_url=backend["base_url"],
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_RETRIES,
        )
    return OpenAI(timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_RETRIES)


def _sampling_parameters(temperature: float) -> Dict[str, float]:
    return {
        "temperature": temperature,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }


def _read_chat_content(response: Any, backend_name: str) -> str:
    choice = response.choices[0]
    content = choice.message.content or ""
    if getattr(choice, "finish_reason", None) == "length":
        debug_print(
            f"[LLM] WARNING: response truncated (finish_reason=length) on backend "
            f"'{backend_name}'; the answer is likely incomplete. content_chars={len(content)}",
            flush=True,
        )
    return content


def _chat_deepseek(messages: List[Dict[str, str]], temperature: float | None) -> str:
    last_exc: Exception | None = None
    for backend in _deepseek_backends():
        client = OpenAI(
            api_key=backend["api_key"],
            base_url=backend["base_url"],
            timeout=DEFAULT_TIMEOUT,
            max_retries=DEFAULT_RETRIES,
        )
        payload: Dict[str, Any] = {
            "model": backend["model"],
            "messages": messages,
            "max_tokens": DEEPSEEK_MAX_TOKENS,
        }
        if temperature is not None:
            payload.update(_sampling_parameters(temperature))
        if backend["provider_routing"] is not None:
            payload["extra_body"] = {"provider": backend["provider_routing"]}
        try:
            response = client.chat.completions.create(**payload)
        except Exception as exc:
            last_exc = exc
            debug_print(
                f"[LLM] DeepSeek backend '{backend['name']}' failed: "
                f"{type(exc).__name__}: {exc}; trying next backend.",
                flush=True,
            )
            continue
        debug_print(f"[LLM] DeepSeek served by backend '{backend['name']}'.", flush=True)
        return _read_chat_content(response, backend["name"])
    raise RuntimeError("All DeepSeek backends failed.") from last_exc


def chat(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float | None,
) -> str:
    """Send ``messages`` to ``model_name`` and return the answer text.

    ``temperature=None`` (and reasoning models) leave the sampling parameters
    at the API defaults.
    """
    if model_name in DEEPSEEK_MODELS:
        return _chat_deepseek(messages, temperature)

    payload: Dict[str, Any] = {"model": model_name, "messages": messages}
    if model_name not in REASONING_MODELS and temperature is not None:
        payload.update(_sampling_parameters(temperature))
    response = client.chat.completions.create(**payload)
    return _read_chat_content(response, model_name)
