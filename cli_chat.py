"""CLI interface for TechFlow Chat System."""

import asyncio
import sys

from langchain_core.messages import HumanMessage

from src.agents import create_initial_state, add_message_to_state, get_agent_graph
from src.rag import get_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

sys.stdout.reconfigure(line_buffering=True) if hasattr(
    sys.stdout, "reconfigure"
) else None


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("TechFlow Chat System - CLI Interface")
    print("=" * 70)
    print("Chat with our support agents about your insurance.")
    print("Type 'exit' or 'quit' to end the conversation.\n")


def format_response(message: str, agent_name: str = None) -> str:
    """Format agent response for display."""
    if agent_name:
        return f"\n[{agent_name}]: {message}\n"
    return f"\n{message}\n"


def run_cli_chat():
    """Run the CLI chat interface."""
    print_banner()

    try:
        logger.info("Initializing RAG vector store...")
        try:
            vector_store = get_vector_store()
            logger.info("Vector store initialized")
        except Exception as e:
            logger.warning(f"Could not initialize vector store: {e}")
            print("Warning: RAG vector store not available")

        logger.info("Loading agent graph...")
        agent_graph = get_agent_graph()
        logger.info("Agent graph loaded")

        print("You can now start chatting. Type 'exit' to quit.\n")

        state = None
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\n" + "=" * 70)
                print("Thank you for chatting with TechFlow. Goodbye!")
                print("=" * 70 + "\n")
                break

            if state is None:
                state = create_initial_state(user_input)
                state["current_agent"] = "greeter"
            else:
                state = add_message_to_state(state, HumanMessage(content=user_input))

            try:
                logger.info(f"Processing message from user...")
                result_state = asyncio.run(agent_graph.ainvoke(state))

                state = result_state

                if result_state.get("messages"):
                    last_message = result_state["messages"][-1]
                    agent_name = result_state.get("current_agent", "Assistant").title()

                    print(format_response(last_message.content, agent_name))
                    sys.stdout.flush()

                    routing = result_state.get("routing_decision")
                    if routing and routing != "__end__":
                        logger.debug(f"Routing decision: {routing}")

            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                print(f"\n[System Error]: {str(e)}\n")
                continue

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Conversation interrupted. Goodbye!")
        print("=" * 70 + "\n")
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\nFatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_cli_chat())
