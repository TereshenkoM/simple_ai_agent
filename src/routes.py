from fastapi import Depends
from fastapi.routing import APIRouter

from src.config import logger
from src.dependencies import get_embedding_model, get_ollama_client, get_qdrant_client
from src.handlers.query import QueryHandler
from src.shemas import AgentRequest

agents_router = APIRouter()


@agents_router.post("/ask")
async def ask(
    request: AgentRequest,
    qdrant_client=Depends(get_qdrant_client),
    ollama_client=Depends(get_ollama_client),
    embedding_model=Depends(get_embedding_model),
):
    logger.info(f"user_id={request.user_id}, query={request.query}")
    query_handler = QueryHandler(
        embedding_model=embedding_model,
        qdrant_client=qdrant_client,
        ollama_client=ollama_client,
    )
    query_handler.process(request.query)

    return {"message": "ok"}
