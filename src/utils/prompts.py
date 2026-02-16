"""Utility for loading agent prompts from files."""

from typing import Dict

from src.config import settings
from .logger import get_logger

logger = get_logger(__name__)

# cache loaded prompts
_prompt_cache: Dict[str, str] = {}


def load_prompt(agent_name: str) -> str:
    """
    Load system prompt for an agent from file.

    Prompts are cached after first load for performance.

    Args:
        agent_name: Name of the agent (greeter, retention, processor)

    Returns:
        System prompt string

    Example:
        >>> prompt = load_prompt("greeter")
        >>> "Greeter Agent" in prompt
        True
    """
    # check cache
    if agent_name in _prompt_cache:
        return _prompt_cache[agent_name]

    prompt_file = settings.PROMPTS_DIR / f"{agent_name}_agent.txt"

    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    try:
        prompt = prompt_file.read_text(encoding="utf-8")
        _prompt_cache[agent_name] = prompt
        logger.info(f"Loaded prompt for {agent_name} agent ({len(prompt)} chars)")
        return prompt

    except Exception as e:
        logger.error(f"Error loading prompt for {agent_name}: {e}")
        raise


def get_greeter_prompt() -> str:
    """Get the Greeter Agent system prompt."""
    return load_prompt("greeter")


def get_retention_prompt() -> str:
    """Get the Retention Agent system prompt."""
    return load_prompt("retention")


def get_processor_prompt() -> str:
    """Get the Processor Agent system prompt."""
    return load_prompt("processor")


def clear_prompt_cache() -> None:
    """Clear the prompt cache (useful for testing)."""
    global _prompt_cache
    _prompt_cache.clear()
    logger.info("Prompt cache cleared")
