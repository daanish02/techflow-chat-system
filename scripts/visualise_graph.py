"""Visualize the agent graph structure."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.graph import get_agent_graph  # noqa: E402


def main():
    """Save graph as PNG."""

    graph = get_agent_graph()

    output_path = project_root / "docs" / "techflow_chat_system-agentic_workflow.png"

    png_bytes = graph.get_graph().draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(png_bytes)

    print(f"Graph saved to: {output_path}")


if __name__ == "__main__":
    main()
