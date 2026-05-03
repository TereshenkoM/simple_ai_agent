import os
from src.config import logger
from ollama import chat


class FileService:
    def __init__(self, directory: str = "./docs"):
        self._directory = directory

    def _list_markdown_files(self) -> list[str]:
        if not os.path.exists(self._directory):
            return []
        return [file for file in os.listdir(self._directory) if file.endswith(".md")]

    def agent_query(self, question: str) -> str:
        files = self._list_markdown_files()

        if not files:
            return "Нет .md файлов"

        promt = f"""
        Вопрос: {question}
        Доступные файлы: {files} 

        Выбери ОДИН файл, который, скорее всего, описывает тему из вопроса.
        Никаких пояснений, только имя файла.
        """

        try:
            response = chat(model='mistral', messages=[{"role": "user", "content": promt}])
            filename = response.message.content.strip().strip('"').strip("'")

            if filename in files:
                return filename
            else:
                return "Не удалось найти подходящий файл"
        except Exception as e:
            logger.error(f"Ошибка: {e}")
