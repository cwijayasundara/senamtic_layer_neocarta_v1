"""LLM + OpenAI client factories (model ids from config).

Ensures OPENAI_API_KEY is present in the process environment (the OpenAI SDK
and LangChain read it from os.environ; pydantic Settings only loads .env).
"""

import os

import openai
from langchain.chat_models import init_chat_model

from semantic_layer.config import settings


def _ensure_key() -> None:
    if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key


def get_chat_model(model: str | None = None):
    _ensure_key()
    return init_chat_model(model or settings.llm_model)


def get_openai_client() -> openai.OpenAI:
    _ensure_key()
    return openai.OpenAI()
