import asyncio

from fastapi import FastAPI
import uvicorn
from src.config import app_config
from src.routes import agents_router
from src.dependencies import get_qdrant_client, get_embedding_model
from qdrant_client.models import VectorParams, Distance, PointStruct
import aiofiles
from os import listdir
from os.path import isfile, join
from src.config import logger
from contextlib import asynccontextmanager


async def create_points(file, qdrant_client, embedding_model=get_embedding_model()):
    async with aiofiles.open(app_config.docs_filepath + "/" + file, "r") as file:
        docs = file.read()
        embeddings = embedding_model.encode(docs)
        points = [
            PointStruct(id=i, vector=emb.tolist(), payload={"text": doc})
            for i, (doc, emb) in enumerate(zip(docs, embeddings.tolist()))
        ]
        await qdrant_client.upsert(points)


@asynccontextmanager
async def lifespan(
    app: FastAPI,  # noqa
    qdrant_client=get_qdrant_client(),
):
    try:
        if not await qdrant_client.collection_exists(collection_name=app_config.collection_name):
            await qdrant_client.create_collection(
                collection_name=app_config.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            files = [
                file
                for file in listdir(app_config.docs_filepath)
                if isfile(join(app_config.docs_filepath, file))
            ]
            tasks = [asyncio.create_task(create_points(file, qdrant_client)) for file in files]
            await asyncio.gather(*tasks)
            logger.info(f"Коллекция {app_config.collection_name} создана и заполнена")

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
