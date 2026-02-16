"""Session management for conversations."""

import uuid
from typing import Dict

from src.agents.state import ConversationState, create_initial_state
from src.utils.logger import get_logger

logger = get_logger(__name__)

_sessions: Dict[str, ConversationState] = {}


def create_session(initial_message: str) -> tuple[str, ConversationState]:
    """
    Create a new conversation session.

    Args:
        initial_message: First message from customer

    Returns:
        Tuple of (session_id, initial_state)
    """
    session_id = str(uuid.uuid4())
    state = create_initial_state(initial_message)

    _sessions[session_id] = state
    logger.info(f"Created session: {session_id}")

    return session_id, state


def get_session(session_id: str) -> ConversationState | None:
    """
    Retrieve existing session state.

    Args:
        session_id: Session identifier

    Returns:
        ConversationState if exists, None otherwise
    """
    return _sessions.get(session_id)


def update_session(session_id: str, state: ConversationState) -> None:
    """
    Update session with new state.

    Args:
        session_id: Session identifier
        state: Updated conversation state
    """
    _sessions[session_id] = state
    logger.debug(f"Updated session: {session_id}")


def delete_session(session_id: str) -> bool:
    """
    Delete a session.

    Args:
        session_id: Session identifier

    Returns:
        True if session existed and was deleted, False otherwise
    """
    if session_id in _sessions:
        del _sessions[session_id]
        logger.info(f"Deleted session: {session_id}")
        return True
    return False


def get_session_count() -> int:
    """Get total number of active sessions."""
    return len(_sessions)


def clear_all_sessions() -> int:
    """
    Clear all sessions (useful for testing).

    Returns:
        Number of sessions cleared
    """
    count = len(_sessions)
    _sessions.clear()
    logger.info(f"Cleared {count} sessions")
    return count
