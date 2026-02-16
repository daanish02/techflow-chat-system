"""FastAPI application for TechFlow Chat System."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from langchain_core.messages import HumanMessage

from src.agents import get_agent_graph
from src.api.schemas import (
    AgentInfo,
    ChatRequest,
    ChatResponse,
    ConfigResponse,
    ErrorResponse,
    HealthResponse,
)
from src.api.session import (
    create_session,
    get_session,
    get_session_count,
    update_session,
)
from src.config import settings
from src.rag import get_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.

    Startup: Initialize RAG vector store
    Shutdown: Cleanup resources
    """
    logger.info("Starting TechFlow Chat System API")

    try:
        # initialize vector store on startup
        vector_store = get_vector_store()
        logger.info(
            f"Vector store loaded with {vector_store._collection.count()} documents"
        )
    except Exception as e:
        logger.warning(f"Vector store initialization failed: {e}")
        logger.warning(
            "Run 'uv run python scripts/setup_rag.py' to initialize RAG system"
        )

    yield

    logger.info("Shutting down TechFlow Chat System API")


app = FastAPI(
    title="TechFlow Chat System",
    description="Multi-agent customer support system for phone insurance retention",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "TechFlow Chat System",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {"health": "/health", "chat": "/chat", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Checks:
    - API is running
    - Vector store is initialized
    - LLM connection (optional)
    """
    health_status = {
        "status": "healthy",
        "api": "running",
        "rag": "unknown",
        "llm": "not_tested",
    }

    # check rag
    try:
        vector_store = get_vector_store()
        count = vector_store._collection.count()
        health_status["rag"] = "healthy" if count > 0 else "empty"
        health_status["rag_document_count"] = count
    except Exception as e:
        health_status["rag"] = "unhealthy"
        health_status["rag_error"] = str(e)

    # check llm
    try:
        from src.llm import test_llm_connection

        if test_llm_connection():
            health_status["llm"] = "healthy"
    except Exception:
        health_status["llm"] = "unhealthy"

    return HealthResponse(**health_status)


@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint for customer conversations.

    Handles:
    - New conversations (no session_id)
    - Continuing conversations (with session_id)
    - Multi-agent orchestration
    - Session state management
    """
    try:
        # determine session
        if request.session_id:
            # continue existing conversation
            state = get_session(request.session_id)
            if not state:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Session {request.session_id} not found or expired",
                )
            session_id = request.session_id
            logger.info(f"Continuing session: {session_id}")

            # add user message to state
            state["messages"].append(HumanMessage(content=request.message))
        else:
            # create new conversation
            session_id, state = create_session(request.message)
            logger.info(f"New session created: {session_id}")

        # get agent graph
        graph = get_agent_graph()

        # add langfuse tracing if enabled
        config = {}
        if settings.langfuse_enabled:
            langfuse_handler = get_langfuse_handler()

            if langfuse_handler:
                # create trace metadata with langfuse keys
                trace_metadata = create_trace_metadata(
                    customer_id=state.get("customer_id"),
                    intent=state.get("intent"),
                    agent=state.get("current_agent"),
                    message_count=len(state["messages"]),
                )

                # add session and user tracking
                trace_metadata["langfuse_session_id"] = session_id
                if state.get("customer_id"):
                    trace_metadata["langfuse_user_id"] = state["customer_id"]

                config["callbacks"] = [langfuse_handler]
                config["metadata"] = trace_metadata
                logger.debug("Langfuse tracing enabled for this request")

        # invoke agent graph
        logger.info(f"Invoking agent graph for session {session_id}")
        result_state = graph.invoke(state, config=config)

        # update session
        update_session(session_id, result_state)

        # extract response
        last_message = result_state["messages"][-1]
        response_text = last_message.content

        # determine current agent and status
        current_agent = result_state.get("current_agent", "unknown")
        routing = result_state.get("routing_decision", "active")

        # determine conversation status
        if routing == "end":
            conversation_status = "completed"
        elif routing in ["tech_support", "billing"]:
            conversation_status = "escalated"
        else:
            conversation_status = "active"

        # agent info
        agent_info = AgentInfo(
            name=current_agent, action=result_state.get("final_action")
        )

        metadata = {
            "message_count": len(result_state["messages"]),
            "intent": result_state.get("intent"),
            "customer_id": result_state.get("customer_id"),
        }

        # create response
        response = ChatResponse(
            message=response_text,
            session_id=session_id,
            agent=agent_info,
            conversation_status=conversation_status,
            customer_authenticated=result_state.get("customer_data") is not None,
            metadata=metadata,
        )

        logger.info(
            f"Session {session_id}: {current_agent} -> {conversation_status} "
            f"({len(result_state['messages'])} messages)"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        )


@app.get("/sessions/count")
async def session_count():
    """Get number of active sessions."""
    return {"active_sessions": get_session_count()}


@app.get("/config", response_model=ConfigResponse)
async def get_config() -> ConfigResponse:
    """
    Get public configuration information.

    Returns non-sensitive configuration details.
    """
    return ConfigResponse(
        llm_model=settings.LLM_MODEL,
        environment=settings.ENVIRONMENT,
        langfuse_enabled=settings.langfuse_enabled,
        chunk_size=settings.CHUNK_SIZE,
        top_k_results=settings.TOP_K_RESULTS,
    )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on port {settings.API_PORT}")

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=settings.API_PORT,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level=settings.LOG_LEVEL.lower(),
    )
