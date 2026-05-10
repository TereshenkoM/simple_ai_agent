import json

from ollama import AsyncClient, ResponseError
from src.config import llm_config, logger


class QueryHandler:
    def __init__(self, ollama_client: AsyncClient):
        self._ollama = ollama_client

    async def process(self, query: str):
        prompt = f"""
        Пользователь может задать какой-либо вопрос для поиска информации на основе контекста.
        Также пользователь может попросить создать задачу.

        Если пользователь задаёт вопрос, то ответь мне JSON с ключом "type" и значением "question".
        Если пользователь хочет создать задачу, то овтеть мне JSON с ключом "type" и значением "task".
        В противном случае верни ключ "type" со значением "error".

        В ответе только валидный JSON и ничего лишнего. Без пояснений и текста. ТОЛЬКО ВАЛДИНЫЙ JSON.

        Ввод пользователя: {query}
        """
        try:
            response = await self._ollama.chat(
                model=llm_config.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            response_content = response.message.content
            answer = json.loads(response_content)

            return {"answer": answer}
        except json.JSONDecodeError:
            logger.error(
                f"Ошибка при сериализации ответа в JSON. Ответ от модели на запрос {query} - {response_content}"
            )
        except ResponseError as e:
            logger.error(f"Ошибка при получении ответа от модели на запрос {query}", exc_info=e)
