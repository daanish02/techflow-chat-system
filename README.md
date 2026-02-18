# TechFlow Chat System

A multi-agent customer support system for handling phone insurance cancellations. The system attempts to retain customers through personalized offers before processing cancellations.

## Architecture

Three specialized agents orchestrated through LangGraph:

- **Greeter Agent**: Authenticates customers and classifies intent
- **Retention Agent**: Offers solutions based on customer data and business rules
- **Processor Agent**: Finalizes cancellations or applies account changes

Agents use RAG to retrieve policy information and call tools to access customer data and calculate offers.

## Requirements

- Python 3.12
- UV package manager
- Google Gemini API key
- Langfuse account (optional, for observability)

## Project Setup

After cloning the project, go into projects root directory and run.

```bash
uv sync
```

followed by,

```bash
uv run scripts/test_cli_scenarios.py
```

You can also run to chat for custom queries.

```bash
uv run cli_chat.py
```
