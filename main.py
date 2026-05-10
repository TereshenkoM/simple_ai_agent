from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from src.config import app_config, llm_config, qdrant_config
from src.routes import agents_router
from src.services.embedding_service import EmbeddingService


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    qdrant_client = AsyncQdrantClient(host=qdrant_config.host, port=qdrant_config.port)
    embedding_model = SentenceTransformer(llm_config.embedding_model)
    try:
        if not await qdrant_client.collection_exists(collection_name=app_config.collection_name):
            embedding_service = EmbeddingService(qdrant_client, embedding_model)
            await embedding_service.create_collection()
        yield
    finally:
        await qdrant_client.close()


def run_app(app: FastAPI) -> None:
    uvicorn.run(
        app,
        host=app_config.host,
        port=app_config.port,
    )


if __name__ == "__main__":
    app = FastAPI(
        title="Простой ИИ агент",
        version=app_config.version,
        lifespan=lifespan,
    )
    app.include_router(agents_router)
    run_app(app)
