"""Langfuse tracing utilities."""

from typing import Any, Dict, Optional

from langfuse.langchain import CallbackHandler

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_langfuse_handler() -> Optional[CallbackHandler]:
    """
    Create Langfuse callback handler if enabled.

    Note: Session ID, user ID, and metadata should be passed via the config dict
    when invoking the graph using the 'metadata' key with special fields:
    - langfuse_session_id: Session identifier for grouping traces
    - langfuse_user_id: User/customer identifier
    - Additional fields: Custom metadata to attach to traces

    Returns:
        CallbackHandler if Langfuse is enabled, None otherwise
    """
    if not settings.langfuse_enabled:
        logger.debug("Langfuse tracing disabled")
        return None

    try:
        # initialize global langfuse client
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        # create callback handler
        handler = CallbackHandler()
        logger.debug("Created Langfuse handler")
        return handler

    except Exception as e:
        logger.error(f"Failed to create Langfuse handler: {e}")
        return None


def create_trace_metadata(
    customer_id: Optional[str] = None,
    intent: Optional[str] = None,
    agent: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Create standardized metadata for traces.

    Args:
        customer_id: Customer identifier
        intent: Classified intent
        agent: Current agent name
        **kwargs: Additional metadata fields

    Returns:
        Metadata dictionary
    """
    metadata = {
        "application": "techflow_chat_system",
        "environment": settings.ENVIRONMENT,
    }

    if customer_id:
        metadata["customer_id"] = customer_id
    if intent:
        metadata["intent"] = intent
    if agent:
        metadata["current_agent"] = agent

    metadata.update(kwargs)

    return metadata


def test_langfuse_connection() -> bool:
    """
    Test Langfuse connection and configuration.

    Returns:
        True if connection successful, False otherwise
    """
    if not settings.langfuse_enabled:
        logger.info("Langfuse is disabled in settings")
        return False

    try:
        # initialize langfuse client and create handler
        from langfuse import Langfuse

        Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )

        # check handler creation
        _ = CallbackHandler()

        logger.info("Langfuse connection successful")
        return True

    except Exception as e:
        logger.error(f"Langfuse connection failed: {e}")
        return False
