"""Conversation state schema for multi-agent system."""

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ConversationState(TypedDict, total=False):
    """
    State schema for conversation flowing through the agent graph.

    This state is passed between agents and updated as the conversation progresses.
    LangGraph manages state immutability and provides each agent with current state.

    Attributes:
        messages: Full conversation history (HumanMessage, AIMessage, etc.)
        customer_email: Email address provided by customer
        customer_id: Customer ID from database
        customer_data: Full customer profile data
        intent: Classified intent (cancellation, technical, billing, general)
        reason: Detailed reason for cancellation (if applicable)
        retention_offers: List of calculated retention offers
        selected_offer: Offer customer accepted (if any)
        final_action: Final action taken (cancelled, kept, accepted_discount, etc.)
        rag_context: Retrieved policy context for current query
        current_agent: Name of currently active agent
        routing_decision: Where to route next (retention, tech_support, billing, processor, end)
        error: Any error message that occurred
        metadata: Additional metadata for tracking
    """

    # conversation history
    messages: List[BaseMessage]

    # customer information
    customer_email: Optional[str]
    customer_id: Optional[str]
    customer_data: Optional[Dict[str, Any]]

    # intent classification
    intent: Optional[str]  # cancellation, technical, billing, general
    reason: Optional[str]  # financial_hardship, not_using, product_defect, etc.

    # retention data
    retention_offers: Optional[List[Dict[str, Any]]]
    selected_offer: Optional[Dict[str, Any]]

    # final outcome
    final_action: Optional[
        str
    ]  # cancelled_insurance, accepted_discount, kept_coverage, etc.

    # rag context
    rag_context: Optional[str]

    # routing
    current_agent: str  # greeter, retention, processor
    routing_decision: Optional[str]  # retention, tech_support, billing, processor, end

    # error handling
    error: Optional[str]

    # additional metadata
    metadata: Optional[Dict[str, Any]]


def create_initial_state(user_message: str) -> ConversationState:
    """
    Create initial conversation state from user's first message.

    Args:
        user_message: The customer's opening message

    Returns:
        Initial ConversationState with user message

    Example:
        >>> state = create_initial_state("I want to cancel my insurance")
        >>> state["current_agent"]
        'greeter'
        >>> len(state["messages"])
        1
    """
    from langchain_core.messages import HumanMessage

    return ConversationState(
        messages=[HumanMessage(content=user_message)],
        customer_email=None,
        customer_id=None,
        customer_data=None,
        intent=None,
        reason=None,
        retention_offers=None,
        selected_offer=None,
        final_action=None,
        rag_context=None,
        current_agent="greeter",
        routing_decision=None,
        error=None,
        metadata={},
    )


def add_message_to_state(
    state: ConversationState, message: BaseMessage
) -> ConversationState:
    """
    Add a message to conversation state.

    Helper function to append messages while preserving immutability pattern.

    Args:
        state: Current conversation state
        message: Message to add

    Returns:
        Updated state with new message
    """
    updated_state = state.copy()
    updated_state["messages"] = state["messages"] + [message]
    return updated_state


def get_last_user_message(state: ConversationState) -> Optional[str]:
    """
    Get the most recent user message content.

    Args:
        state: Current conversation state

    Returns:
        Content of last user message, or None if no user messages
    """
    from langchain_core.messages import HumanMessage

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content  # ty:ignore[invalid-return-type]
    return None


def get_last_ai_message(state: ConversationState) -> Optional[str]:
    """
    Get the most recent AI message content.

    Args:
        state: Current conversation state

    Returns:
        Content of last AI message, or None if no AI messages
    """
    from langchain_core.messages import AIMessage

    for message in reversed(state["messages"]):
        if isinstance(message, AIMessage):
            return message.content  # ty:ignore[invalid-return-type]
    return None


def is_authenticated(state: ConversationState) -> bool:
    """
    Check if customer is authenticated.

    Args:
        state: Current conversation state

    Returns:
        True if customer data loaded, False otherwise
    """
    return state.get("customer_data") is not None


def has_intent(state: ConversationState) -> bool:
    """
    Check if intent has been classified.

    Args:
        state: Current conversation state

    Returns:
        True if intent is set, False otherwise
    """
    return state.get("intent") is not None
