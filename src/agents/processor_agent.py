"""Processor Agent for finalizing customer decisions and logging actions."""

from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.agents.state import ConversationState
from src.llm import create_agent_llm
from src.tools import update_customer_status
from src.utils.logger import get_logger
from src.utils.prompts import get_processor_prompt

logger = get_logger(__name__)


def get_processor_runnable():
    """Create the Processor Agent runnable."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_processor_prompt()),
            ("placeholder", "{messages}"),
        ]
    )

    llm = create_agent_llm("processor", tools=None, temperature=0.3)

    return prompt | llm


def determine_final_action(state: ConversationState) -> tuple[str, str]:
    """
    Determine what action to log based on conversation outcome.

    Returns:
        Tuple of (action, details) for update_customer_status tool
    """
    # get recent conversation
    recent_messages = [
        str(msg.content) if isinstance(msg.content, str) else str(msg.content)
        for msg in state["messages"][-5:]
        if isinstance(msg, HumanMessage)
    ]
    recent_text = " ".join(recent_messages).lower()

    # check if customer accepted an offer
    accept_keywords = ["yes", "accept", "ok", "sure", "sounds good", "take it", "deal"]
    decline_keywords = ["no", "cancel", "proceed", "still want to", "decline"]

    has_offers = state.get("retention_offers") is not None

    if has_offers and any(keyword in recent_text for keyword in accept_keywords):
        # determine which type of offer was accepted
        offers = state["retention_offers"]
        if not offers:
            return ("kept_coverage", "customer decided to keep current coverage")

        # check conversation for offer type mentions
        if any(offer["type"] == "discount" for offer in offers):
            if "discount" in recent_text or "%" in recent_text:
                discount_offer = next(o for o in offers if o["type"] == "discount")
                return ("accepted_discount", discount_offer["description"])

        if any(offer["type"] == "pause" for offer in offers):
            if "pause" in recent_text:
                pause_offer = next(o for o in offers if o["type"] == "pause")
                return ("accepted_pause", pause_offer["description"])

        if any(offer["type"] == "upgrade" for offer in offers):
            if "upgrade" in recent_text:
                upgrade_offer = next(o for o in offers if o["type"] == "upgrade")
                return ("accepted_upgrade", upgrade_offer["description"])

        # default to first offer if unclear
        first_offer = offers[0]
        return (f"accepted_{first_offer['type']}", first_offer["description"])

    elif any(keyword in recent_text for keyword in decline_keywords):
        # customer wants to cancel
        reason = state.get("reason", "customer_request")
        return ("cancelled_insurance", f"reason: {reason}")

    else:
        # customer kept coverage without changes
        return ("kept_coverage", "customer decided to keep current coverage")


async def processor_node(
    state: ConversationState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Processor Agent node execution.

    1. Determine the final action taken
    2. Log the action using update_customer_status tool
    3. Generate confirmation message
    4. Mark conversation as complete
    """
    logger.info("Executing Processor Agent node")

    updates = {}

    # determine final action
    if not state.get("final_action"):
        action, details = determine_final_action(state)
        updates["final_action"] = action
        logger.info(f"Determined final action: {action}")

        # log the action
        if state.get("customer_id"):
            result = update_customer_status.invoke(
                {
                    "customer_id": state["customer_id"],
                    "action": action,
                    "details": details,
                }
            )

            if result.get("success"):
                logger.info(f"Successfully logged action for {state['customer_id']}")

                # Add confirmation to context
                log_context = SystemMessage(
                    content=f"Action logged: {action} - {details}"
                )
            else:
                logger.error(f"Failed to log action: {result.get('error')}")
                log_context = SystemMessage(
                    content="Note: Action logging encountered an error"
                )
        else:
            logger.warning("No customer_id available for logging")
            log_context = SystemMessage(content="Processing without customer_id")
    else:
        action = state["final_action"]
        log_context = SystemMessage(content=f"Action already logged: {action}")

    # generate confirmation response
    chain = get_processor_runnable()

    # add context
    messages_with_context = state["messages"] + [log_context]
    response_state = state.copy()
    response_state["messages"] = messages_with_context

    response = await chain.ainvoke(response_state)

    # update state with all messages
    updates["messages"] = state["messages"] + [log_context, response]

    # mark as complete
    updates["routing_decision"] = "end"
    logger.info("Conversation complete")

    return updates


async def processor_node_sync(
    state: ConversationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Async wrapper for processor_node."""
    return await processor_node(state, config)
