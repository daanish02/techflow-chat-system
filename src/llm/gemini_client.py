"""Google Gemini LLM client wrapper."""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_llm(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Get configured Gemini LLM instance.

    Creates a ChatGoogleGenerativeAI instance with configuration from settings.
    Supports function calling and tool usage for LangChain agents.

    Args:
        temperature: Override default temperature (0.0-1.0)
        max_tokens: Override default max output tokens
        model: Override default model name
        **kwargs: Additional arguments for ChatGoogleGenerativeAI

    Returns:
        Configured ChatGoogleGenerativeAI instance

    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is Care+?")
        >>> print(response.content)
    """
    llm_config = {
        "model": model or settings.LLM_MODEL,
        "temperature": temperature
        if temperature is not None
        else settings.LLM_TEMPERATURE,
        "max_output_tokens": max_tokens or settings.LLM_MAX_TOKENS,
        "google_api_key": settings.GEMINI_API_KEY,
        "convert_system_message_to_human": True,  # gemini compatibility
    }

    # merge additional kwargs
    llm_config.update(kwargs)

    logger.debug(
        f"Initializing Gemini LLM: {llm_config['model']} "
        f"(temp={llm_config['temperature']}, max_tokens={llm_config['max_output_tokens']})"
    )

    try:
        llm = ChatGoogleGenerativeAI(**llm_config)
        return llm

    except Exception as e:
        logger.error(f"Failed to initialize Gemini LLM: {e}")
        raise


def get_llm_with_tools(tools: List[Any], **kwargs: Any) -> Runnable:
    """
    Get LLM configured with tool binding.

    Binds LangChain tools to the LLM for function calling support.
    The LLM will be able to call these tools during conversation.

    Args:
        tools: List of LangChain tool objects
        **kwargs: Additional arguments for get_llm()

    Returns:
        LLM with tools bound

    Example:
        >>> from src.tools.data_tools import get_customer_data
        >>> llm = get_llm_with_tools([get_customer_data])
        >>> # LLM can now call get_customer_data during conversation
    """
    llm = get_llm(**kwargs)

    logger.info(f"Binding {len(tools)} tools to LLM")

    try:
        llm_with_tools = llm.bind_tools(tools)
        return llm_with_tools

    except Exception as e:
        logger.error(f"Failed to bind tools to LLM: {e}")
        raise


def create_agent_llm(
    agent_name: str,
    tools: Optional[List[Any]] = None,
    temperature: Optional[float] = None,
) -> BaseChatModel | Runnable:
    """
    Create an LLM instance configured for a specific agent.

    Convenience function that creates an LLM with optional tool binding
    and logs which agent is using it.

    Args:
        agent_name: Name of the agent (for logging)
        tools: Optional list of tools to bind
        temperature: Override temperature for this agent

    Returns:
        Configured LLM instance

    Example:
        >>> from src.tools.data_tools import get_customer_data
        >>> llm = create_agent_llm("greeter", tools=[get_customer_data])
    """
    logger.info(f"Creating LLM for {agent_name} agent")

    if tools:
        return get_llm_with_tools(tools, temperature=temperature)
    else:
        return get_llm(temperature=temperature)


def test_llm_connection() -> bool:
    """
    Test that LLM connection works.

    Useful for verifying API key and configuration during setup.

    Returns:
        True if connection successful, False otherwise

    Example:
        >>> if test_llm_connection():
        ...     print("LLM ready")
    """
    try:
        logger.info("Testing LLM connection...")
        llm = get_llm()

        # test query
        response = llm.invoke("Say 'OK' if you can read this.")

        if response and response.content:
            logger.info("LLM connection successful")
            return True
        else:
            logger.error("LLM returned empty response")
            return False

    except Exception as e:
        logger.error(f"LLM connection failed: {e}")
        return False
