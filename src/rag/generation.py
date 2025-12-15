# src/rag/generation.py

from __future__ import annotations

from typing import List

from src.rag.retrieval import RetrievedChunk
from src.agent.llm_utils import load_model, call_llm


def build_prompt(query: str, chunks: List[RetrievedChunk], max_chunks: int = 1) -> str:
    """
    Build a retrieval-augmented prompt in PT-BR for the restaurant assistant.

    The answer should be strictly grounded in the provided documentation.
    """
    selected = chunks[:max_chunks]
    context = (
        "\n\n".join(c.text.strip() for c in selected)
        if selected
        else "Nenhum contexto disponível."
    )

    return (
        "Você é um assistente virtual de um restaurante.\n"
        "Responda APENAS com base na documentação abaixo.\n"
        "Se a informação não estiver presente, diga que não está disponível.\n"
        "Responda em português do Brasil, de forma direta e curta.\n\n"
        f"Documentação:\n{context}\n\n"
        f"Pergunta:\n{query}\n\n"
        "Resposta:"
    )


def generate_answer_from_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    model_name: str = "Mixtral 8x22B",
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> str:
    """
    Generate a short answer in PT-BR using retrieved chunks and a chat model.

    Parameters
    ----------
    query : str
        User question in natural language.
    chunks : List[RetrievedChunk]
        Retrieved context chunks from the vector store.
    model_name : str
        Name of the model to load via `load_model`.
    temperature : float
        Sampling temperature for the model.
    max_tokens : int
        Maximum number of new tokens to generate.

    Returns
    -------
    str
        A short, grounded answer in PT-BR.
    """
    if not chunks:
        return (
            "Não encontrei informações suficientes na documentação para responder. "
            "Recomendo consultar um atendente."
        )

    # Reuse the shared model loader from the agent layer
    llm = load_model(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    prompt = build_prompt(query, chunks)

    # Use unified call helper (handles .invoke() or callable)
    answer = call_llm(llm, prompt)
    return answer.strip()