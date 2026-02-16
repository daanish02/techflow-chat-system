"""Pydantic schemas for API requests and responses."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Customer's message",
        examples=["I want to cancel my insurance"],
    )

    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity",
        examples=["customer_C001_session_123"],
    )

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate message is not empty or whitespace only."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class AgentInfo(BaseModel):
    """Information about which agent handled the message."""

    name: str = Field(
        ...,
        description="Name of the agent",
        examples=["greeter", "retention", "processor"],
    )

    action: Optional[str] = Field(
        None,
        description="Action taken by the agent",
        examples=["authenticated_customer", "calculated_offers", "logged_cancellation"],
    )


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    message: str = Field(..., description="Agent's response message")

    session_id: str = Field(..., description="Session ID for this conversation")

    agent: AgentInfo = Field(
        ..., description="Information about the agent that responded"
    )

    conversation_status: str = Field(
        ...,
        description="Status of the conversation",
        examples=["active", "completed", "escalated"],
    )

    customer_authenticated: bool = Field(
        default=False, description="Whether customer has been authenticated"
    )

    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata about the conversation"
    )


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type or code")

    message: str = Field(..., description="Human-readable error message")

    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )


class ConversationSummary(BaseModel):
    """Summary of a conversation."""

    session_id: str
    customer_id: Optional[str]
    intent: Optional[str]
    final_action: Optional[str]
    message_count: int
    agents_involved: List[str]


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(
        ...,
        description="Overall health status",
        examples=["healthy", "degraded", "unhealthy"],
    )

    api: str = Field(..., description="API service status")

    rag: str = Field(..., description="RAG system status")

    llm: str = Field(default="not_tested", description="LLM connection status")

    rag_document_count: Optional[int] = Field(
        None, description="Number of documents in RAG system"
    )

    rag_error: Optional[str] = Field(None, description="RAG error message if unhealthy")


class ConfigResponse(BaseModel):
    """Configuration information response."""

    llm_model: str = Field(..., description="LLM model being used")
    environment: str = Field(..., description="Environment (development/production)")
    langfuse_enabled: bool = Field(
        ..., description="Whether Langfuse tracing is enabled"
    )
    chunk_size: int = Field(..., description="RAG chunk size")
    top_k_results: int = Field(..., description="Number of RAG results retrieved")
