"""Greeter Agent for customer authentication and intent classification."""

import re
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.agents.state import ConversationState
from src.llm import create_agent_llm
from src.tools import get_customer_data
from src.utils.logger import get_logger
from src.utils.prompts import get_greeter_prompt

logger = get_logger(__name__)


def get_greeter_runnable():
    """Create the Greeter Agent runnable."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", get_greeter_prompt()),
            ("placeholder", "{messages}"),
        ]
    )

    llm = create_agent_llm("greeter", tools=None, temperature=0.0)

    return prompt | llm


def extract_email(text: str) -> str | None:
    """Extract email address from text using regex."""
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    match = re.search(email_pattern, text)
    return match.group(0) if match else None


def classify_intent(text: str) -> str:
    """
    Classify user intent based on message content.

    Simple rule-based classification as a fallback or pre-processor.
    The LLM will do the heavy lifting, but this helps with routing logic.
    """
    text = text.lower()

    # intent keywords
    cancellation_keywords = [
        "cancel",
        "terminate",
        "stop",
        "end subscription",
        "too expensive",
        "afford",
        "switching",
        "don't need",
    ]

    billing_keywords = [
        "bill",
        "charged",
        "charges",
        "cost",
        "price",
        "invoice",
        "payment",
        "refund",
        "amount",
    ]

    tech_keywords = [
        "broken",
        "not working",
        "screen",
        "battery",
        "charging",
        "won't charge",
        "wont charge",
        "won't turn on",
        "glitch",
        "repair",
        "fix",
        "overheating",
    ]

    # check keywords
    if any(k in text for k in billing_keywords):
        return "billing"
    if any(k in text for k in cancellation_keywords):
        return "cancellation"
    if any(k in text for k in tech_keywords):
        return "technical"

    return "general"


async def greeter_node(
    state: ConversationState, config: RunnableConfig
) -> Dict[str, Any]:
    """
    Greeter Agent node execution.

    1. Check if authenticated
    2. If not, try to extract email and lookup customer
    3. If authenticated, classify intent
    4. Generate response or route to next agent
    """
    logger.info("Executing Greeter Agent node")

    messages = state["messages"]
    last_msg = messages[-1] if messages else None
    last_msg_content = (
        last_msg.content if last_msg and isinstance(last_msg, HumanMessage) else ""
    )
    # check if string
    last_user_msg = (
        str(last_msg_content)
        if isinstance(last_msg_content, str)
        else str(last_msg_content)
    )

    # initialize updates
    updates = {}

    # authentication check
    if not state.get("customer_data"):
        # extract email from message
        email = extract_email(last_user_msg)

        if email:
            logger.info(f"Detected email: {email}")
            customer = get_customer_data.invoke(email)

            if "error" not in customer:
                updates["customer_data"] = customer
                updates["customer_email"] = email
                updates["customer_id"] = customer["customer_id"]
                logger.info(f"Authenticated customer: {customer['name']}")
            else:
                logger.warning(f"Customer lookup failed for {email}")

    # intent classification
    # classify only if we have enough context or explicit intent
    current_intent = state.get("intent")

    if not current_intent:
        intent = classify_intent(last_user_msg)
        if intent != "general":
            updates["intent"] = intent
            logger.info(f"Classified intent: {intent}")

    # generate response
    # if updates to state are made, run the chain with new context
    chain = get_greeter_runnable()

    # invoke chain
    response = await chain.ainvoke(state)

    # add response to updates - include all existing messages plus new response
    updates["messages"] = state["messages"] + [response]

    # update routing decision
    # if authenticated and intent is clear, route to appropriate agent
    # else, end turn and wait for next user input

    # determine next step based on state and new updates
    is_auth = state.get("customer_data") or updates.get("customer_data")
    intent = state.get("intent") or updates.get("intent")

    if is_auth and intent == "cancellation":
        updates["routing_decision"] = "retention"
        logger.info("Routing to Retention Agent")
    elif is_auth and intent == "technical":
        updates["routing_decision"] = "tech_support"
        logger.info("Routing to Tech Support (End)")
    elif is_auth and intent == "billing":
        updates["routing_decision"] = "billing"
        logger.info("Routing to Billing (End)")
    else:
        # Don't set routing_decision - this will cause route_from_greeter to return "__end__"
        logger.info("Not enough info yet - ending turn, waiting for next input")

    return updates


async def greeter_node_sync(
    state: ConversationState, config: RunnableConfig
) -> Dict[str, Any]:
    """Async wrapper for greeter_node."""
    return await greeter_node(state, config)
