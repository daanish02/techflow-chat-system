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

    # greeter needs customer lookup tool
    tools = [get_customer_data]
    llm = create_agent_llm("greeter", tools=tools, temperature=0.0)

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

    tech_keywords = [
        "broken",
        "not working",
        "screen",
        "battery",
        "charge",
        "won't turn on",
        "glitch",
        "repair",
        "fix",
        "overheating",
    ]

    billing_keywords = [
        "bill",
        "charge",
        "cost",
        "price",
        "invoice",
        "payment",
        "refund",
        "amount",
        "charged",
    ]

    # check keywords
    if any(k in text for k in cancellation_keywords):
        return "cancellation"
    if any(k in text for k in tech_keywords):
        return "technical"
    if any(k in text for k in billing_keywords):
        return "billing"

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
    last_user_msg = (
        messages[-1].content if isinstance(messages[-1], HumanMessage) else ""
    )

    # initialize updates
    updates = {}

    # authentication check
    if not state.get("customer_data"):
        # extract email from message
        email = extract_email(last_user_msg)

        if email:
            logger.info(f"Detected email: {email}")
            customer = get_customer_data.invoke({"email": email})

            if "error" not in customer:
                updates["customer_data"] = customer
                updates["customer_email"] = email
                updates["customer_id"] = customer["customer_id"]
                logger.info(f"Authenticated customer: {customer['name']}")

                # system message about authentication
                updates["messages"] = [
                    SystemMessage(
                        content=f"Authenticated customer: {customer['name']} ({customer['email']})"
                    )
                ]
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

    # add response to updates
    if "messages" not in updates:
        updates["messages"] = []
    updates["messages"].append(response)

    # update routing decision
    # if authenticated and intent is clear, route to appropriate agent
    # else, stay with greeter to get more info

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
        updates["routing_decision"] = "greeter"  # stays here
        logger.info("Staying with Greeter Agent")

    return updates
