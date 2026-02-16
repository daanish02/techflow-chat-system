"""Retention Agent for customer retention and offer management."""

from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.agents.state import ConversationState
from src.llm import create_agent_llm
from src.rag import query_policies
from src.tools import calculate_retention_offer
from src.utils.logger import get_logger
from src.utils.prompts import get_retention_prompt

logger = get_logger(__name__)


def get_retention_runnable():
    """Create the Retention Agent runnable."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_retention_prompt()),
            ("placeholder", "{messages}"),
        ]
    )

    # agent needs offer calculation tool
    tools = [calculate_retention_offer]
    llm = create_agent_llm("retention", tools=tools, temperature=0.7)

    return prompt | llm


def determine_cancellation_reason(state: ConversationState) -> str:
    """
    Determine the detailed reason for cancellation from conversation.

    Analyzes messages to identify specific reason category.
    """
    # get all user messages
    user_messages = [
        str(msg.content) if isinstance(msg.content, str) else str(msg.content)
        for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    ]
    all_text = " ".join(user_messages).lower()

    # map keywords
    reason_mapping = {
        "financial_hardship": [
            "can't afford",
            "too expensive",
            "cost",
            "money",
            "budget",
            "financial",
            "save money",
            "cheaper",
        ],
        "not_using": [
            "never use",
            "haven't used",
            "don't use",
            "unused",
            "not worth it",
            "no claims",
        ],
        "product_defect": [
            "broken",
            "defect",
            "not working",
            "problem with phone",
            "overheating",
            "screen issue",
            "battery problem",
        ],
        "too_expensive": ["expensive", "high price", "costly", "price"],
        "switching_carrier": [
            "switching",
            "new carrier",
            "moving to",
            "changing provider",
        ],
    }

    # check if reason keywords
    for reason, keywords in reason_mapping.items():
        if any(keyword in all_text for keyword in keywords):
            logger.info(f"Detected cancellation reason: {reason}")
            return reason

    # default
    logger.info("Using default cancellation reason: other")
    return "other"


def should_query_rag(state: ConversationState) -> bool:
    """
    Determine if RAG query is needed based on conversation context.

    Query RAG if:
    - Customer questioning value/benefits
    - Product defect mentioned (need return policy)
    - Need to explain coverage
    """
    last_messages = " ".join(
        [
            str(msg.content) if isinstance(msg.content, str) else str(msg.content)
            for msg in state["messages"][-3:]
            if isinstance(msg, HumanMessage)
        ]
    ).lower()

    rag_triggers = [
        "what does",
        "coverage",
        "benefit",
        "worth",
        "value",
        "what's included",
        "return",
        "replacement",
        "defect",
        "never used",
        "don't use",
    ]

    return any(trigger in last_messages for trigger in rag_triggers)


async def retention_node(
    state: ConversationState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Retention Agent node execution.

    1. Determine cancellation reason if not already set
    2. Query RAG if needed for policy context
    3. Calculate retention offers if not already done
    4. Generate empathetic response with solutions
    5. Determine if customer accepts or declines
    """
    logger.info("Executing Retention Agent node")

    updates = {}

    # determine cancellation reason
    reason = state.get("reason")
    if not reason:
        reason = determine_cancellation_reason(state)
        updates["reason"] = reason

    # query rag if necessary
    rag_context = None
    if should_query_rag(state):
        # build query from reason and recent messages
        last_user_msg = (
            [
                msg.content
                for msg in state["messages"][-3:]
                if isinstance(msg, HumanMessage)
            ][-1]
            if state["messages"]
            else ""
        )

        query = f"{reason.replace('_', ' ')} {last_user_msg}"
        rag_context = query_policies(query)
        updates["rag_context"] = rag_context
        logger.info(f"Retrieved RAG context ({len(rag_context)} chars)")

    # calculate offers if not done
    offers = state.get("retention_offers")
    customer_data = state.get("customer_data")
    if not offers and customer_data:
        customer_tier = customer_data["tier"]
        offer_result = calculate_retention_offer.invoke(
            {"customer_tier": customer_tier, "reason": reason}
        )

        if "error" not in offer_result:
            updates["retention_offers"] = offer_result["offers"]
            logger.info(f"Calculated {len(offer_result['offers'])} retention offers")

    # build context for llm
    # add rag context to messages
    context_messages = []
    if rag_context:
        context_messages.append(
            SystemMessage(content=f"Relevant Policy Information:\n{rag_context}")
        )

    # add offers to context
    offers_to_present = state.get("retention_offers") or updates.get("retention_offers")
    if offers_to_present:
        offers_text = "\n".join(
            [
                f"- {offer['type']}: {offer['description']}"
                for offer in offers_to_present
            ]
        )
        context_messages.append(
            SystemMessage(content=f"Available Offers:\n{offers_text}")
        )

    # generate response
    chain = get_retention_runnable()

    # build message list with context
    messages_with_context = state["messages"] + context_messages
    response_state = state.copy()
    response_state["messages"] = messages_with_context

    response = await chain.ainvoke(response_state)

    # Add response to updates
    if "messages" not in updates:
        updates["messages"] = []
    if context_messages:
        updates["messages"].extend(context_messages)
    updates["messages"].append(response)

    # determine routing
    # check if customer has made a decision
    last_user_messages = [
        msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)
    ]
    last_user_msg = last_user_messages[-1] if last_user_messages else ""
    # check if string
    last_user_msg_str = (
        str(last_user_msg) if isinstance(last_user_msg, str) else str(last_user_msg)
    )

    last_msg_lower = last_user_msg_str.lower()

    # check for acceptance keywords
    accept_keywords = [
        "yes",
        "ok",
        "sure",
        "accept",
        "sounds good",
        "i'll take",
        "that works",
        "agree",
        "deal",
    ]

    # check for decline keywords
    decline_keywords = [
        "no thanks",
        "still want to cancel",
        "not interested",
        "just cancel",
        "proceed with cancellation",
        "decline",
    ]

    if any(keyword in last_msg_lower for keyword in accept_keywords):
        # customer likely accepts
        updates["routing_decision"] = "processor"
        logger.info("Customer appears to accept offer, routing to Processor")
    elif any(keyword in last_msg_lower for keyword in decline_keywords):
        # customer declining, wants to cancel
        updates["routing_decision"] = "processor"
        logger.info("Customer declining offers, routing to Processor for cancellation")
    else:
        # continue conversation
        updates["routing_decision"] = "retention"
        logger.info("Continuing retention conversation")

    return updates


def retention_node_sync(
    state: ConversationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Synchronous wrapper for retention_node."""
    import asyncio

    return asyncio.run(retention_node(state, config))
