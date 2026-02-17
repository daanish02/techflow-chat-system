"""LangGraph orchestration for multi-agent system."""

from typing import Literal

from langgraph.graph import StateGraph, END

from src.agents.greeter_agent import greeter_node_sync
from src.agents.retention_agent import retention_node_sync
from src.agents.processor_agent import processor_node_sync
from src.agents.state import ConversationState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def route_from_greeter(
    state: ConversationState,
) -> Literal["retention", "tech_support", "billing", "__end__"]:
    """
    Route from greeter based on intent classification.

    Routes to:
    - retention: If cancellation intent
    - tech_support: If technical issue (terminal node)
    - billing: If billing question (terminal node)
    - __end__: End turn, wait for next user input
    """
    routing = state.get("routing_decision")

    if routing == "retention":
        logger.info("Routing: greeter -> retention")
        return "retention"
    elif routing == "tech_support":
        logger.info("Routing: greeter -> tech_support (end)")
        return "tech_support"
    elif routing == "billing":
        logger.info("Routing: greeter -> billing (end)")
        return "billing"
    else:
        logger.info("Routing: greeter -> END (end turn, wait for next input)")
        return "__end__"


def route_from_retention(state: ConversationState) -> Literal["processor", "__end__"]:
    """
    Route from retention based on customer decision.

    Routes to:
    - processor: If customer made a decision (accept or decline)
    - __end__: End turn, wait for next user input
    """
    routing = state.get("routing_decision")

    if routing == "processor":
        logger.info("Routing: retention -> processor")
        return "processor"
    else:
        logger.info("Routing: retention -> END (end turn, wait for next input)")
        return "__end__"


def create_agent_graph() -> StateGraph:
    """
    Create and compile the multi-agent conversation graph.

    Graph structure:
        START -> greeter -> [retention | tech_support | billing | END]
                retention -> [processor | END]
                processor -> END
                tech_support -> END
                billing -> END

    Returns:
        Compiled StateGraph ready for execution
    """
    logger.info("Building agent graph")

    # initialise graph with state schema
    graph = StateGraph(ConversationState)

    # add nodes
    graph.add_node("greeter", greeter_node_sync)
    graph.add_node("retention", retention_node_sync)
    graph.add_node("processor", processor_node_sync)

    # terminal nodes
    graph.add_node(
        "tech_support",
        lambda state: {
            "routing_decision": "end",
            "messages": [
                type(state["messages"][-1])(
                    content="I'm transferring you to our technical support team who can help with your device issue. They'll be with you shortly."
                )
            ],
        },
    )

    graph.add_node(
        "billing",
        lambda state: {
            "routing_decision": "end",
            "messages": [
                type(state["messages"][-1])(
                    content="I'm transferring you to our billing department who can help explain your charges. They'll assist you right away."
                )
            ],
        },
    )

    # entry point
    graph.set_entry_point("greeter")

    # conditional edges from greeter
    graph.add_conditional_edges(
        "greeter",
        route_from_greeter,
        {
            "retention": "retention",
            "tech_support": "tech_support",
            "billing": "billing",
            "__end__": END,
        },
    )

    # conditional edges from retention
    graph.add_conditional_edges(
        "retention",
        route_from_retention,
        {
            "processor": "processor",
            "__end__": END,
        },
    )

    # terminal edges
    graph.add_edge("processor", END)
    graph.add_edge("tech_support", END)
    graph.add_edge("billing", END)

    logger.info("Agent graph built successfully")

    return graph


def get_agent_graph():
    """
    Get the compiled agent graph.

    This is the main function to use for executing conversations.

    Returns:
        Compiled graph ready for invocation

    Example:
        >>> graph = get_agent_graph()
        >>> result = graph.invoke(initial_state)
    """
    graph = create_agent_graph()
    compiled = graph.compile()
    logger.info("Agent graph compiled")
    return compiled
