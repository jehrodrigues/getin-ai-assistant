# src/agent/graph/workflow.py

from __future__ import annotations

from typing import Any, Dict

from langgraph.graph import StateGraph, END

from src.agent.graph.state import AgentState
from src.agent.intents.intent_classifier import classify_intent
from src.agent.extractors.params_extractor import extract_params
from src.agent.actions import api_availability
from src.agent.actions import api_reservations
from src.agent.actions import rag_action
from src.agent.llm_utils import get_chat_model, call_llm_text


def classify_intent_node(state: AgentState) -> AgentState:
    user_input = state.get("user_input") or ""

    model = get_chat_model(
        model_name="deepseek-chat",
        temperature=0.0,
        max_tokens=64,
    )

    previous_intent = state.get("intent")
    new_intent = classify_intent(user_input)

    if previous_intent and previous_intent != "other" and new_intent == "other":
        intent = previous_intent
    else:
        intent = new_intent

    state["intent"] = intent
    return state


def extract_params_node(state: AgentState) -> AgentState:
    user_input = state.get("user_input") or ""
    intent = state.get("intent")

    existing_params: Dict[str, Any] = state.get("params") or {}

    model = get_chat_model(
        model_name="deepseek-chat",  # or your default
        temperature=0.0,
        max_tokens=512,
    )

    new_params: Dict[str, Any] = extract_params(model, user_input, intent=intent)

    merged: Dict[str, Any] = dict(existing_params)

    for key, value in new_params.items():
        if key == "raw":
            raw_list = []
            if "raw" in merged and isinstance(merged["raw"], list):
                raw_list = merged["raw"]
            elif "raw" in merged and isinstance(merged["raw"], dict):
                raw_list = [merged["raw"]]

            raw_list.append(value)
            merged["raw"] = raw_list
            continue

        if value is not None:
            merged[key] = value
        else:
            merged.setdefault(key, None)

    state["params"] = merged
    return state


def run_action_node(state: AgentState) -> AgentState:
    """
    Node: dispatch to the correct action based on the classified intent.

    Expects:
        state["intent"]
        state["params"]

    Produces:
        state["action_result"]  (standardized dict with ok/error/etc.)

    For availability checks, if the action returns a list of available
    sectors (available_sectors), we also propagate that information into
    state["params"]["available_sectors"] so that later steps (e.g. reservation
    creation) and the answer generator can use user-friendly sector names.
    """
    intent = (state.get("intent") or "").strip()
    params: Dict[str, Any] = state.get("params") or {}

    # Default action result if something goes wrong before dispatch.
    action_result: Dict[str, Any] = {
        "type": "unknown",
        "ok": False,
        "error": {
            "code": "UNKNOWN_INTENT",
            "message": (
                "Não consegui entender exatamente o que você deseja fazer. "
                "Você pode reformular sua pergunta?"
            ),
        },
        "request": {
            "params": params,
        },
    }

    if intent == "check_availability":
        action_result = api_availability.check_availability(params)

        # If availability returned sectors, propagate them to params
        if action_result.get("ok") and action_result.get("type") == "availability":
            sectors = action_result.get("available_sectors") or []
            if sectors:
                merged_params = dict(params)
                merged_params["available_sectors"] = sectors
                state["params"] = merged_params

    elif intent == "create_reservation":
        action_result = api_reservations.create_reservation(params)

    elif intent == "view_next_reservation":
        action_result = api_reservations.view_next_reservation(params)

    elif intent == "list_reservations":
        action_result = api_reservations.list_reservations(params)

    elif intent == "cancel_reservation":
        action_result = api_reservations.cancel_reservation(params)

    elif intent == "restaurant_faq":
        # RAG-based FAQ: we pass both user_input and params
        user_input = state.get("user_input") or ""
        action_result = rag_action.answer_with_rag(user_input=user_input, params=params)

    # Fallback for completely unknown intents is the default initialized above.
    state["action_result"] = action_result
    return state


def generate_answer_node(state: AgentState) -> AgentState:
    """
    Node: turn the action_result into a natural PT-BR answer using an LLM.

    Expects:
        state["user_input"]
        state["intent"]
        state["params"]
        state["action_result"]

    Produces:
        state["answer"]  (final assistant reply in PT-BR)
    """
    user_input = state.get("user_input") or ""
    intent = state.get("intent") or ""
    params = state.get("params") or {}
    action_result = state.get("action_result") or {}

    # If the RAG action already produced a final answer, we can just reuse it.
    if action_result.get("type") == "restaurant_faq" and action_result.get("ok") is True:
        answer = action_result.get("answer") or ""
        state["answer"] = answer or (
            "Consegui consultar a documentação interna, mas não obtive uma resposta clara."
        )
        return state

    # If there is an explicit error, ask the LLM to explain it nicely in PT-BR.
    error = action_result.get("error")
    if error:
        prompt = (
            "Você é um assistente virtual de um restaurante, respondendo em português do Brasil.\n"
            "Você recebeu o resultado de uma ação interna (API GET IN ou RAG) que contém um erro.\n"
            "Sua tarefa é explicar educadamente ao usuário o que aconteceu, em poucas frases, "
            "e orientar o próximo passo quando possível.\n\n"

            "Interpretação importante de erros da API GET IN:\n"
            "- Quando o erro mencionar mensagens como:\n"
            "  'Não é possível realizar 2 reservas para o mesmo dia/horário utilizando este celular ou e-mail',\n"
            "  isso indica uma REGRA DE NEGÓCIO de duplicidade de contato.\n"
            "- Esse erro NÃO significa falta de disponibilidade nem que o horário esteja ocupado.\n"
            "- Significa apenas que o mesmo telefone ou e-mail não pode ser usado para mais de uma reserva no mesmo horário.\n\n"

            "- Nunca afirme que já existe uma reserva naquele horário a menos que isso tenha sido retornado explicitamente "
            "por uma consulta de reservas.\n\n"

            f"Pergunta do usuário:\n{user_input}\n\n"
            f"Intent reconhecida: {intent}\n\n"
            "Resultado da ação (JSON simplificado):\n"
            f"{action_result}\n\n"

            "Agora responda apenas a mensagem final em PT-BR para o usuário, "
            "sem mencionar JSON, APIs ou detalhes técnicos internos."
        )
    else:
        # Success case: ask the LLM to transform the payload into a friendly answer.
        # IMPORTANT: when available_sectors is present, the model should list the
        # sector_name values and ask the user to choose one of them if relevant.
        prompt = (
            "Você é um assistente virtual de um restaurante, respondendo em português do Brasil.\n"
            "Você recebeu o resultado de uma ação interna em formato JSON.\n\n"
            "Regras específicas importantes:\n"
            "- NUNCA diga que a reserva está 'confirmada' se o JSON indicar status 'pending'.\n"
            "- Se status='pending', diga que a reserva foi criada e está pendente de confirmação.\n"
            "- Se status='confirmed', diga que está confirmada.\n"
            "- Se existir 'id' em response.data.id, inclua como 'código da reserva'.\n"
            "- Se existir 'sector.name', use o nome do setor.\n"
            "- Se confirmation_sent=false, não diga que o e-mail já foi enviado; diga que poderá receber a confirmação.\n\n"
            f"Pergunta do usuário:\n{user_input}\n\n"
            f"Intent reconhecida: {intent}\n\n"
            f"Parâmetros extraídos: {params}\n\n"
            "Resultado da ação (JSON simplificado):\n"
            f"{action_result}\n\n"
            "Agora responda apenas a mensagem final em PT-BR."
        )

    model = get_chat_model(
        model_name="deepseek-chat",
        temperature=0.2,
        max_tokens=512,
    )
    answer = call_llm_text(model, prompt)

    state["answer"] = answer.strip() if answer else (
        "Não consegui gerar uma resposta adequada no momento. "
        "Por favor, tente novamente em instantes."
    )
    return state


def build_workflow() -> Any:
    """
    Build and compile the LangGraph workflow for the restaurant agent.

    The pipeline is:

        user_input
          → classify_intent_node
          → extract_params_node
          → run_action_node
          → generate_answer_node
          → END
    """
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("extract_params", extract_params_node)
    graph.add_node("run_action", run_action_node)
    graph.add_node("generate_answer", generate_answer_node)

    # Entry point
    graph.set_entry_point("classify_intent")

    graph.add_edge("classify_intent", "extract_params")
    graph.add_edge("extract_params", "run_action")
    graph.add_edge("run_action", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def run_agent(user_input: str) -> AgentState:
    """
    Convenience wrapper to run the whole workflow for a single turn.

    This is what your PoC 3 `run_poc3.py` can call.
    """
    workflow = build_workflow()
    initial_state: AgentState = {
        "user_input": user_input,
        "intent": None,
        "params": {},
        "action_result": None,
        "answer": None,
    }
    final_state: AgentState = workflow.invoke(initial_state)
    return final_state