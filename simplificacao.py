from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, AIMessage, AnyMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from typing import Any, cast, Annotated
from operator import add
from pathlib import Path

import os
import json

class WorkflowState(TypedDict):
    text: str

    # State of the messages graph
    llm_calls: Annotated[int, add]
    input_tokens: Annotated[int, add]
    output_tokens: Annotated[int, add]

    # Errors states
    simple_attempts: Annotated[int, add]
    moderate_attempts: Annotated[int, add]
    aggressive_attempts: Annotated[int, add]

    # Text analysis result
    analysis: dict[str, Any]

    # Simplification results
    simple_simplification: str
    moderate_simplification: str
    aggressive_simplification: str

    # Simplification feedbacks
    simple_simplification_feedback: dict[str, Any]
    moderate_simplification_feedback: dict[str, Any]
    aggressive_simplification_feedback: dict[str, Any]

def _router(feedback: dict[str, Any] | str) -> str:
    try:
        if feedback is not None and isinstance(feedback, dict):
            return str(feedback["status"]).lower()
        else:
            return "rejected"
    except KeyError:
        return "rejected"

def get_token_usage(message: AnyMessage) -> dict[str, int]:
    usage = getattr(message, "usage_metadata", None)

    if usage:
        return {
            "input_tokens": int(usage.get("input_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
        }

    # Fallback para providers que colocam uso em response_metadata
    response_metadata: dict[str, Any] = getattr(message, "response_metadata", {}) or {}

    token_usage = (
        response_metadata.get("token_usage")
        or response_metadata.get("usage")
        or {}
    )

    input_tokens = int(
        token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )

    output_tokens = int(
        token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )

    total_tokens = int(
        token_usage.get("total_tokens")
        or input_tokens + output_tokens
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }



class LinguagemSimplesGraph:
    @staticmethod
    def _load_prompt(filename: str) -> str:
        with open(filename, "r") as f:
            return f.read()

    @staticmethod
    def run(input_text: str, model_name: str):
        try:
            # if model name is defined but empty, throw
            if not model_name:
                raise ValueError("Model name cannot be empty")
        except ValueError:
            raise ValueError("Model name is required")
        model = init_chat_model(
            model=model_name,
            temperature=0.3,
            model_provider="openrouter",
            max_retries=3,
        )

        def llm_analisador_node(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/analisador.txt")
            message = model.invoke([SystemMessage(content=prompt), HumanMessage(content=state["text"])])

            usage = get_token_usage(message)
            try:
                content = json.loads(f"{message.content}")
            except json.JSONDecodeError:
                content = message.content

            return {"analysis": content, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def llm_simple_simplificator(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/simplificador.txt")
            message_list = []
            if "simple_simplification_feedback" in state and isinstance(state["simple_simplification_feedback"], dict) and state["simple_simplification_feedback"]["status"] == "rejected":
                message_list = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Estudantes, acadêmicos e profissionais da área.
Texto original: {state['text']}
"""),
                    AIMessage(content=f"Simplificação realizada: {state['simple_simplification']}"),
                    HumanMessage(content=f"Feedback para sua simplificação: {json.dumps(state['simple_simplification_feedback'])}. Melhore sua simplificação com base nele."),
                ]
            else:
                message_list = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Estudantes, acadêmicos e profissionais da área.
Texto original: {state['text']}
"""),
                ]

            message = model.invoke(message_list)

            usage = get_token_usage(message)

            return {"simple_simplification": message.content, "simple_attempts": 1, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def llm_moderate_simplificator(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/simplificador.txt")
            message_list = []
            if "moderate_simplification_feedback" in state and isinstance(state["moderate_simplification_feedback"], dict) and state["moderate_simplification_feedback"]["status"] == "rejected":
                message_list = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Jornalistas e profissionais de comunicação.
Texto original: {state['text']}
"""),
                    AIMessage(content=f"Simplificação realizada: {state['moderate_simplification']}"),
                    HumanMessage(content=f"Feedback para sua simplificação: {json.dumps(state['moderate_simplification_feedback'])}. Melhore sua simplificação com base nele."),
                ]
            else:
                message_list = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Jornalistas e profissionais de comunicação.
Texto original: {state['text']}
"""),
                    HumanMessage(content=state["text"]),
                ]
            message = model.invoke(message_list)

            usage = get_token_usage(message)

            return {"moderate_simplification": message.content, "moderate_attempts": 1, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def llm_aggressive_simplificator(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/simplificador.txt")
            message_list = []
            if "aggressive_simplification_feedback" in state and isinstance(state["aggressive_simplification_feedback"], dict) and state["aggressive_simplification_feedback"]["status"] == "rejected":
                message_list = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Público geral
Texto original: {state['text']}
"""),
                    AIMessage(content=f"Simplificação realizada: {state['aggressive_simplification']}"),
                    HumanMessage(content=f"Feedback para sua simplificação: {json.dumps(state['aggressive_simplification_feedback'])}. Melhore sua simplificação com base nele."),
                ]
            else:
                message_list = [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Público geral
Texto original: {state['text']}
"""),
                    HumanMessage(content=state["text"]),
                ]
            message = model.invoke(message_list)

            usage = get_token_usage(message)

            return {"aggressive_simplification": message.content, "aggressive_attempts": 1, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def llm_simple_avaliator(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/avaliador.txt")
            message = model.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Estudantes, acadêmicos e profissionais da área.
Texto original: {state['text']}
Texto simplificado: {state['simple_simplification']}
"""),
                ]
            )

            usage = get_token_usage(message)

            try:
                content = json.loads(f"{message.content}")
            except json.JSONDecodeError:
                content = message.content
                print(f"Failed to parse simple feedback: {content}")

            return {"simple_simplification_feedback": content, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def llm_moderate_avaliator(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/avaliador.txt")
            message = model.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Jornalistas e profissionais de comunicação.
Texto original: {state['text']}
Texto simplificado: {state['moderate_simplification']}
"""),
                ]
            )

            usage = get_token_usage(message)

            try:
                content = json.loads(f"{message.content}")
            except json.JSONDecodeError:
                content = message.content
                print(f"Failed to parse moderate feedback: {content}")

            return {"moderate_simplification_feedback": content, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def llm_aggressive_avaliator(state: WorkflowState) -> dict[str, Any]:
            prompt = LinguagemSimplesGraph._load_prompt("prompts/avaliador.txt")
            message = model.invoke(
                [
                    SystemMessage(content=prompt),
                    HumanMessage(content=f"""
Análise do texto original: {json.dumps(state['analysis'])}
Nível de simplificação: Público geral
Texto original: {state['text']}
Texto simplificado: {state['aggressive_simplification']}
"""),
                ]
            )

            usage = get_token_usage(message)

            try:
                content = json.loads(f"{message.content}")
            except json.JSONDecodeError:
                content = message.content
                print(f"Failed to parse aggressive feedback: {content}")

            return {"aggressive_simplification_feedback": content, "llm_calls": 1, "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}

        def branch_finished(state: WorkflowState) -> dict[str, Any]:
            if state["aggressive_attempts"] >= 3:
                print(f"Max attempts for aggressive simplification reached: {state['aggressive_attempts']}")
            elif state["moderate_attempts"] >= 3:
                print(f"Max attempts for moderate simplification reached: {state['moderate_attempts']}")
            elif state["simple_attempts"] >= 3:
                print(f"Max attempts for simple simplification reached: {state['simple_attempts']}")

            return {}

        def aggregator(state: WorkflowState) -> dict:
            output_path = Path(f"output/{model_name}/data.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with output_path.open("r", encoding="utf-8") as f_read:
                    existing_data = json.load(f_read)

                    if not isinstance(existing_data, list):
                        existing_data = []

            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            existing_data.append(dict(state))

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=4, ensure_ascii=False)

            return {}

        def route_decision_simple_feedback(state: WorkflowState) -> str:
            status = _router(state["simple_simplification_feedback"])
            if status == "approved":
                return "approved"
            if state.get("simple_attempts", 0) < 3:
                print("Simple simplification rejected, attempting again")
                return "rejected"
            return "max_attempts"

        def route_decision_moderate_feedback(state: WorkflowState) -> str:
            status = _router(state["moderate_simplification_feedback"])
            if status == "approved":
                return "approved"
            if state.get("moderate_attempts", 0) < 3:
                print("Moderate simplification rejected, attempting again")
                return "rejected"
            return "max_attempts"

        def route_decision_aggressive_feedback(state: WorkflowState) -> str:
            status = _router(state["aggressive_simplification_feedback"])
            if status == "approved":
                return "approved"
            if state.get("aggressive_attempts", 0) < 3:
                print("Aggressive simplification rejected, attempting again")
                return "rejected"
            return "max_attempts"

        # WORKFLOW
        parallel_builder = StateGraph(WorkflowState)

        # NODES
        parallel_builder.add_node("llm_analisador", llm_analisador_node)

        parallel_builder.add_node("llm_simple_simplificator", llm_simple_simplificator)
        parallel_builder.add_node("llm_moderate_simplificator", llm_moderate_simplificator)
        parallel_builder.add_node("llm_aggressive_simplificator", llm_aggressive_simplificator)

        parallel_builder.add_node("llm_simple_avaliator", llm_simple_avaliator)
        parallel_builder.add_node("llm_moderate_avaliator", llm_moderate_avaliator)
        parallel_builder.add_node("llm_aggressive_avaliator", llm_aggressive_avaliator)

        parallel_builder.add_node("simple_finished", branch_finished)
        parallel_builder.add_node("moderate_finished", branch_finished)
        parallel_builder.add_node("aggressive_finished", branch_finished)

        parallel_builder.add_node("aggregator", aggregator)

        # EDGES
        parallel_builder.add_edge(START, "llm_analisador")

        parallel_builder.add_edge("llm_analisador", "llm_simple_simplificator")
        parallel_builder.add_edge("llm_analisador", "llm_moderate_simplificator")
        parallel_builder.add_edge("llm_analisador", "llm_aggressive_simplificator")

        parallel_builder.add_edge("llm_simple_simplificator", "llm_simple_avaliator")
        parallel_builder.add_edge("llm_moderate_simplificator", "llm_moderate_avaliator")
        parallel_builder.add_edge("llm_aggressive_simplificator", "llm_aggressive_avaliator")

        parallel_builder.add_conditional_edges(
            "llm_simple_avaliator",
            route_decision_simple_feedback,
            {
                "approved": "simple_finished",
                "rejected": "llm_simple_simplificator",
                "max_attempts": "simple_finished",
            },
        )

        parallel_builder.add_conditional_edges(
            "llm_moderate_avaliator",
            route_decision_moderate_feedback,
            {
                "approved": "moderate_finished",
                "rejected": "llm_moderate_simplificator",
                "max_attempts": "moderate_finished",
            },
        )

        parallel_builder.add_conditional_edges(
            "llm_aggressive_avaliator",
            route_decision_aggressive_feedback,
            {
                "approved": "aggressive_finished",
                "rejected": "llm_aggressive_simplificator",
                "max_attempts": "aggressive_finished",
            },
        )

        parallel_builder.add_edge(
            ["simple_finished", "moderate_finished", "aggressive_finished"],
            "aggregator",
        )

        parallel_builder.add_edge("aggregator", END)

        # COMPILE WORKFLOW
        parallel_workflow = parallel_builder.compile()

        # DISPLAY GRAPH if file does not exist
        if not os.path.exists("graph.png"):
            print("Gerando gráfico do workflow...")
            graph_image = parallel_workflow.get_graph().draw_mermaid_png()
            with open("graph.png", "wb") as f:
                f.write(graph_image)

        # RUN WORKFLOW
        state = parallel_workflow.invoke(cast(WorkflowState, {"text": input_text, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0}))
        return state
