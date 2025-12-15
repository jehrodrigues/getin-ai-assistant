# src/agent/intents/intent_classifier.py

from __future__ import annotations

from typing import Literal

from src.agent.llm_utils import (
    call_llm,
    build_single_turn_prompt,
    get_chat_model,
)


Intent = Literal[
    "check_availability",
    "create_reservation",
    "view_next_reservation",
    "cancel_reservation",
    "restaurant_faq",
    "other",
]

ALLOWED_INTENTS = {
    "check_availability",
    "create_reservation",
    "view_next_reservation",
    "cancel_reservation",
    "restaurant_faq",
    "other",
}


SYSTEM_PROMPT = """
Você é um classificador de intenções para um assistente de restaurante integrado à plataforma GET IN.

Seu objetivo é ler a mensagem do usuário (em português do Brasil) e decidir QUAL é a intenção principal,
pensando em qual tipo de operação o sistema deve executar.

Você deve escolher EXATAMENTE UMA das seguintes intenções, e responder apenas com o rótulo:

- check_availability  -> quando o usuário quer saber se há mesa/disponibilidade para um dia/horário/
                         número de pessoas.
                         Exemplos:
                           - "tem mesa pra 4 hoje às 20h?"
                           - "vocês têm disponibilidade amanhã no jantar?"
                           - "qual horário tem vaga pra 2 pessoas?"

- create_reservation  -> quando o usuário quer criar/efetuar uma reserva nova.
                         Exemplos:
                           - "quero fazer uma reserva para sábado às 21h para 2 pessoas"
                           - "pode reservar uma mesa pra 6 amanhã no almoço?"
                           - "quero agendar uma reserva em meu nome"

- view_next_reservation -> quando o usuário quer consultar a próxima reserva futura dele.
                           Exemplos:
                             - "qual é minha próxima reserva?"
                             - "tenho alguma reserva para hoje?"
                             - "minha próxima reserva está confirmada?"

- cancel_reservation  -> quando o usuário quer cancelar uma reserva existente.
                         Exemplos:
                           - "quero cancelar minha reserva de hoje às 20h"
                           - "pode cancelar a reserva em nome de João amanhã?"
                           - "não vou mais, cancela minha reserva"

- restaurant_faq      -> quando o usuário faz perguntas gerais sobre o restaurante, não diretamente
                         sobre criar/cancelar/consultar reserva ou checar disponibilidade.
                         Exemplos:
                           - "vocês têm opções veganas?"
                           - "qual o dress code?"
                           - "aceitam crianças/pets?"
                           - "qual o telefone/endereço da casa?"

- other               -> quando a mensagem não se encaixa claramente em nenhuma das intenções anteriores
                         ou está muito ambígua/incompleta.

Regras importantes:
- Responda SOMENTE com um destes rótulos:
  check_availability, create_reservation, view_next_reservation,
  cancel_reservation, restaurant_faq, other.
- Não explique sua resposta.
- Não inclua mais nada além do rótulo.
""".strip()


def _normalize_output(raw_output: str) -> str:
    """
    Normalize the raw LLM output to one of the allowed intent labels.

    Handles cases like:
    - "check_availability"
    - "Intent: create_reservation"
    - "intenção: view_next_reservation\n"
    - "A melhor intenção é: restaurant_faq"
    """
    text = (raw_output or "").strip().lower()

    # Strip obvious prefixes the model might add
    for prefix in ("intent:", "intenção:", "intencao:", "label:", "rótulo:", "rotulo:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    # Exact match
    if text in ALLOWED_INTENTS:
        return text

    # Heuristic: check if any allowed label appears inside the text
    for intent in ALLOWED_INTENTS:
        if intent in text:
            return intent

    # Fallback
    return "other"


def classify_intent(user_input: str) -> Intent:
    """
    Classify the user's high-level intent using a generic LLM.

    This function internally loads a small chat model via `get_chat_model`
    to keep the call site (workflow) simple.

    Parameters
    ----------
    user_input : str
        Raw message written by the user in PT-BR.

    Returns
    -------
    Intent
        One of the allowed intent labels defined in `Intent`.
    """
    prompt = build_single_turn_prompt(SYSTEM_PROMPT, user_input)

    llm = get_chat_model(
        model_name="deepseek-chat",
        temperature=0.0,
        max_tokens=128,
    )

    raw = call_llm(llm, prompt)
    intent_str = _normalize_output(raw)

    if intent_str not in ALLOWED_INTENTS:
        intent_str = "other"

    return intent_str