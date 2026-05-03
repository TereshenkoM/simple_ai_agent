from fastapi.routing import APIRouter
from src.shemas import AgentRequest
from src.config import logger

agents_router = APIRouter()


@agents_router.post("/ask")
async def ask(request: AgentRequest):
    logger.info(f'user_id={request.user_id}, query={request.query}')

    return {"message": "ok"}