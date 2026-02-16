"""FastAPI application for TechFlow Chat System."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import ConfigResponse, HealthResponse
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
