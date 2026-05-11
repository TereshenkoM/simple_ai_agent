from fastapi import Depends, HTTPException
from fastapi.routing import APIRouter

from ollama import AsyncClient
from src.config import logger
from src.dependencies import get_ollama_client
from src.handlers.query import QueryHandler
from src.schemas import AgentRequest

agents_router = APIRouter()


@agents_router.post("/ask")
async def ask(request: AgentRequest, ollama_client: AsyncClient = Depends(get_ollama_client)):
    logger.info(f"user_id={request.user_id}, query={request.query}")
    query_handler = QueryHandler(ollama_client)
    try:
        response = await query_handler.process(request.query, request.user_id)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    logger.info(f"response={response}")
    return response
