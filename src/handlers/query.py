import json

from ollama import AsyncClient, ResponseError
from src.config import llm_config, logger
from src.services.question import QuestionService
from src.services.task import TaskService


class QueryHandler:
    def __init__(self, ollama_client: AsyncClient):
        self._ollama = ollama_client
        self._services_map = {"question": QuestionService(), "task": TaskService()}

    async def process(self, query: str, user_id=None):
        prompt = f"""
        Классифицируй запрос пользователя по типу сервиса:

        - "question": запрос про поиск/ответ по документам/контексту (RAG), справка, объяснение, "что такое", "как работает" и т.п.
        - "task": любые действия с задачами: создать задачу, добавить/изменить комментарий, обновить задачу, статус, номер, "добавь комментарий", "создай таск", "задача", "тикет" и т.п.

        Ответ:
        - Если это "question" -> {{ "type": "question" }}
        - Если это "task" -> {{ "type": "task" }}
        - Иначе -> {{ "type": "error", "error": "<коротко почему>" }}

        Примеры:
        - "Создай задачу на проверку" -> {{ "type": "task" }}
        - "Добавь комментарий протестировано" -> {{ "type": "task" }}
        - "Что такое RAG?" -> {{ "type": "question" }}

        В ответе только валидный JSON и ничего лишнего.

        Ввод пользователя: {query}
        """
        try:
            response = await self._ollama.chat(
                model=llm_config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            response_content = json.loads(response.message.content)
            query_type = response_content["type"]

            if query_type == "error":
                raise ResponseError(response_content["error"])

            service = self._services_map[query_type]

            return await service.process(query, user_id)

        except json.JSONDecodeError:
            logger.error(
                f"Ошибка при сериализации ответа в JSON. Ответ от модели на запрос {query} - {response_content}"
            )
        except ResponseError as e:
            logger.error(f"Ошибка при получении ответа от модели на запрос {query}", exc_info=e)
