from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer

from ollama import AsyncClient
from src.config import app_config, llm_config, logger, qdrant_config


# TODO Убрать дубли после MVP (dishka)
class QuestionService:
    @staticmethod
    async def process(question: str):
        qdrant_client = AsyncQdrantClient(host=qdrant_config.host, port=qdrant_config.port)
        embedding_model = SentenceTransformer(llm_config.embedding_model)
        ollama = AsyncClient(host=llm_config.ollama_host)
        query_vector = embedding_model.encode(question).tolist()

        search_result = await qdrant_client.query_points(
            collection_name=app_config.collection_name, query=query_vector, limit=1
        )
        context = search_result.points[0].payload["text"]
        prompt = f"""
        Отвечай только на основе контекста. Если не знаешь - скажи "Не знаю".
        Контект: {context}
        Вопрос: {question}
        """

        response = await ollama.chat(
            model=llm_config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        logger.info(f"Ответ на запрос пользователя {response.message.content}")

        return {"response": response.message.content}
