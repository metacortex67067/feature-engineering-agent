from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    data_profiler_node,
    feature_judge_node,
    ft_generator_node,
    iterative_coder_node,
    planner_node,
    sandbox_executor_node,
    writer_node,
)
from .state import AgentState


def route_after_execution(state: AgentState) -> str:
    """Route after executor: success -> judge, retry -> coder, fallback -> judge."""
    if state.get("status") == "timeout":
        return "fallback"
    if not state.get("execution_error"):
        return "success"
    if state.get("retry_count", 0) < state.get("max_retries", 2):
        return "retry"
    return "fallback"


def route_after_judge(state: AgentState) -> str:
    """Route after judge: continue iterating or finish."""
    if state.get("iteration", 0) >= state.get("max_iterations", 15):
        return "done"
    if state.get("status") == "timeout":
        return "done"
    return "continue"


def build_agent_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("profiler", data_profiler_node)
    workflow.add_node("ft_generator", ft_generator_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("coder", iterative_coder_node)
    workflow.add_node("executor", sandbox_executor_node)
    workflow.add_node("judge", feature_judge_node)
    workflow.add_node("writer", writer_node)

    # Entry: profiler -> ft_generator -> planner -> coder
    workflow.set_entry_point("profiler")
    workflow.add_edge("profiler", "ft_generator")
    workflow.add_edge("ft_generator", "planner")
    workflow.add_edge("planner", "coder")

    # Coder -> executor
    workflow.add_edge("coder", "executor")

    # After executor: success/fallback -> judge, retry -> coder
    workflow.add_conditional_edges(
        "executor",
        route_after_execution,
        {
            "success": "judge",
            "retry": "coder",
            "fallback": "judge",
        },
    )

    # After judge: continue -> planner (new plan each iteration), done -> writer
    workflow.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "continue": "planner",
            "done": "writer",
        },
    )

    workflow.add_edge("writer", END)
    return workflow.compile()
