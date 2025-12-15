# src/agent/llm_utils.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.core.config import settings

OPENAI_API_KEY = settings.openai_api_key
ANTHROPIC_API_KEY = settings.anthropic_api_key
DEEPSEEK_API_KEY = settings.deepseek_api_key
MISTRAL_API_KEY = settings.mistral_api_key


def load_model(model_name: str, temperature: float = 0.1, max_tokens: int = 1024):
    """
    Load any supported chat model. This function is intentionally generic
    and reused by both the agent (intent classification, parameter extraction,
    answer generation) and the RAG module.

    All returned objects must support either:
        - .invoke(prompt: str)
        - being callable directly (llm(prompt))
    """

    # ---------------- OPENAI ----------------
    if model_name in {"gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"}:
        from langchain_openai import ChatOpenAI
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY is not configured.")
        return ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=OPENAI_API_KEY,
        )

    # ---------------- ANTHROPIC ----------------
    if model_name in {"claude-3-7-sonnet-20250219", "claude-3-5-haiku-20241022"}:
        from langchain_anthropic import ChatAnthropic
        if not ANTHROPIC_API_KEY:
            raise EnvironmentError("ANTHROPIC_API_KEY is not configured.")
        return ChatAnthropic(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            anthropic_api_key=ANTHROPIC_API_KEY,
        )

    # ---------------- DEEPSEEK ----------------
    if model_name in {"DeepSeek-V3", "deepseek-chat", "deepseek-reasoner"}:
        from langchain_openai import ChatOpenAI
        if not DEEPSEEK_API_KEY:
            raise EnvironmentError("DEEPSEEK_API_KEY is not configured.")
        return ChatOpenAI(
            model_name="deepseek-chat",  # DeepSeek API uses this fixed internal name
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base="https://api.deepseek.com/v1",
        )

    # ---------------- MISTRAL ----------------
    if model_name in {"Mixtral 8x22B", "open-mixtral-8x22b", "open-mixtral-8x7b"}:
        from langchain_mistralai import ChatMistralAI
        if not MISTRAL_API_KEY:
            raise EnvironmentError("MISTRAL_API_KEY is not configured.")

        resolved = "open-mixtral-8x22b" if "22" in model_name else "open-mixtral-8x7b"

        return ChatMistralAI(
            model_name=resolved,
            temperature=temperature,
            max_tokens=max_tokens,
            mistral_api_key=MISTRAL_API_KEY,
        )

    # ---------------- LOCAL MODELS ----------------
    if model_name.startswith("local:"):
        from transformers import pipeline as hf_pipeline
        from langchain.llms import HuggingFacePipeline

        subpath = model_name.split("local:", 1)[-1]
        model_dir = Path(os.getenv("MODEL_PATH", "")) / subpath

        if not model_dir.exists():
            raise FileNotFoundError(f"Local model not found at: {model_dir}")

        pipe = hf_pipeline(
            "text-generation",
            model=str(model_dir),
            max_new_tokens=max_tokens,
        )
        return HuggingFacePipeline(pipeline=pipe)

    raise ValueError(f"Unsupported model: {model_name}")


def get_chat_model(
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 512,
):
    """
    Convenience helper used by the agent workflow to obtain a chat model.

    This simply delegates to `load_model` but keeps a clearer semantic name
    in the workflow code.
    """
    return load_model(model_name=model_name, temperature=temperature, max_tokens=max_tokens)


def normalize_llm_output(result: Any) -> str:
    """
    Normalize any LLM output (AIMessage, list, plain string, HF pipeline output)
    into a clean string.
    """
    if hasattr(result, "content"):
        return str(result.content).strip()

    if isinstance(result, str):
        return result.strip()

    return str(result).strip()


def call_llm(llm: Any, prompt: str) -> str:
    """
    Calls a model in a unified way. Supports:
        - llm.invoke(prompt)    (LangChain chat models)
        - llm(prompt)           (callable pipelines or functions)

    Always returns a normalized string output.
    """
    # ChatOpenAI, ChatAnthropic, ChatMistralAI, DeepSeek via LC
    if hasattr(llm, "invoke") and callable(llm.invoke):
        result = llm.invoke(prompt)
        return normalize_llm_output(result)

    # HuggingFacePipeline or any callable object
    if callable(llm):
        result = llm(prompt)
        return normalize_llm_output(result)

    raise TypeError(
        f"Invalid LLM object. Expected .invoke() or __call__. Received: {type(llm)!r}"
    )


def call_llm_text(llm: Any, prompt: str) -> str:
    """
    Thin alias around `call_llm`, kept for semantic clarity in the agent code.

    The agent workflow uses `call_llm_text(model, prompt)` to emphasize that
    we expect a plain text string as the final result.
    """
    return call_llm(llm, prompt)


def build_single_turn_prompt(system_prompt: str, user_input: str) -> str:
    """
    Build a single-turn prompt for intent classification, parameter extraction,
    or agent answer generation.

    The scaffold is in PT-BR so the model is nudged to stay in Portuguese.
    """
    return (
        f"{system_prompt.strip()}\n\n"
        f"Mensagem do usuário:\n\"\"\"{user_input.strip()}\"\"\"\n\n"
        "Resposta:"
    )