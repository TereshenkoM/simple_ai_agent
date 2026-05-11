import asyncio
from os import listdir
from uuid import uuid4

import aiofiles
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.config import app_config, logger, qdrant_config


class QdrantCollectionBuilder:
    def __init__(self, qdrant_client, embedding_model) -> None:
        self._qdrant_client = qdrant_client
        self._embedding_model = embedding_model
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
            strip_headers=False,
        )

    async def _create_points(self, file):
        async with aiofiles.open(app_config.docs_filepath + "/" + file) as f:
            markdown_text = await f.read()
            header_chunks = self._markdown_splitter.split_text(markdown_text)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=qdrant_config.chunk_size,
                chunk_overlap=qdrant_config.chunk_overlap,
            )

            chunks = text_splitter.split_documents(header_chunks)
            texts = [doc.page_content for doc in chunks]
            embeddings = self._embedding_model.encode(texts)
            points = [
                PointStruct(
                    id=str(uuid4()),
                    vector=emb,
                    payload={
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                        "source_file": file,
                    },
                )
                for doc, emb in zip(chunks, embeddings.tolist(), strict=True)
            ]
            await self._qdrant_client.upsert(app_config.collection_name, points)
            logger.info(f"Данные из {file} добавлены к коллекцию {app_config.collection_name}")

    async def create_collection(self):
        await self._qdrant_client.create_collection(
            collection_name=app_config.collection_name,
            vectors_config=VectorParams(size=qdrant_config.vector_size, distance=Distance.COSINE),
        )
        files = [file for file in listdir(app_config.docs_filepath) if file.endswith(".md")]
        tasks = [asyncio.create_task(self._create_points(file)) for file in files]
        await asyncio.gather(*tasks)
        logger.info(f"Коллекция {app_config.collection_name} создана и заполнена")
