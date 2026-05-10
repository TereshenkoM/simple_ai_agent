from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from ollama import AsyncClient
from src.config import llm_config, qdrant_config


def get_ollama_client() -> AsyncClient:
    return AsyncClient(host=llm_config.ollama_host)


def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(host=qdrant_config.host, port=qdrant_config.port)


def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(llm_config.embedding_model)
