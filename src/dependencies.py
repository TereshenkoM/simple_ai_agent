from ollama import AsyncClient
from src.config import llm_config


# TODO перейти на dishka
def get_ollama_client() -> AsyncClient:
    return AsyncClient(host=llm_config.ollama_host)
