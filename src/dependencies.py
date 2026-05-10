from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from ollama import AsyncClient
from src.config import llm_config, qdrant_config


def get_ollama_client():
    return AsyncClient(host=llm_config.ollama_host)


def get_qdrant_client():
    return AsyncQdrantClient(host=qdrant_config.qdrant_host, port=qdrant_config.qdrant_port)


def get_embedding_model():
    return SentenceTransformer(llm_config.embedding_model)
