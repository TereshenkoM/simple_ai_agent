from sentence_transformers import SentenceTransformer
from qdrant_client import AsyncQdrantClient
from ollama import AsyncClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from src.config import llm_config, qdrant_config


class RAGService:
    def __init__(self, collection_name="docs"):
        self._embedding_model = SentenceTransformer(llm_config.embedding_model)
        self._qdrant = AsyncQdrantClient(qdrant_config.qdrant_host, port=qdrant_config.qdrant_port)
        self._ollama = AsyncClient(host=llm_config.ollama_host)
        self._collection_name = collection_name

    async def create_collections(self):
        await self._qdrant.create_collections(
            collection_name=self._collection_name,
            vector_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        embeddings = self._embedding_model.encode()