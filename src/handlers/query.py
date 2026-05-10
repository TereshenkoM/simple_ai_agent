class QueryHandler:
    def __init__(self, embedding_model, qdrant_client, ollama_client):
        self._embedding_model = embedding_model
        self._qdrant_client = qdrant_client
        self._ollama_client = ollama_client

    def process(self, question: str):
        pass
