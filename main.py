import asyncio
from contextlib import asynccontextmanager
from os import listdir
from uuid import uuid4

import aiofiles
import uvicorn
from fastapi import FastAPI
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import app_config, logger
from src.dependencies import get_embedding_model, get_qdrant_client
from src.routes import agents_router


async def create_points(file, qdrant_client):
    embedding_model = get_embedding_model()
    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    filename = file
    async with aiofiles.open(app_config.docs_filepath + "/" + filename) as f:
        markdown_text = await f.read()
        header_chunks = markdown_splitter.split_text(markdown_text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
        )

        chunks = text_splitter.split_documents(header_chunks)
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_model.encode(texts)
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=emb,
                payload={
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "source_file": filename,
                },
            )
            for doc, emb in zip(chunks, embeddings.tolist(), strict=True)
        ]
        await qdrant_client.upsert(app_config.collection_name, points)
        logger.info(f"Данные из {file} добавлены к коллекцию {app_config.collection_name}")


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    qdrant_client = (get_qdrant_client(),)
    try:
        if await qdrant_client.collection_exists(collection_name=app_config.collection_name):
            await qdrant_client.delete_collection(app_config.collection_name)

        await qdrant_client.create_collection(
            collection_name=app_config.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        files = [file for file in listdir(app_config.docs_filepath) if file.endswith(".md")]
        tasks = [asyncio.create_task(create_points(file)) for file in files]
        await asyncio.gather(*tasks)
        logger.info(f"Коллекция {app_config.collection_name} создана и заполнена")

        yield
    finally:
        await qdrant_client.delete_collection(app_config.collection_name)
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
