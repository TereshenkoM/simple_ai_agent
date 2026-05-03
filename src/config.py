from pydantic_settings import BaseSettings, SettingsConfigDict
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("agent")


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class AppConfig(BaseConfig):
    model_config = SettingsConfigDict(env_prefix="APP_")

    host: str
    port: int
    version: str


class QdrantConfid(BaseConfig):
    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    qdrant_host: str
    qdrant_port: int


class LLMConfig(BaseConfig):
    ollama_host: str
    embedding_model: str
    llm_model: str


class DBConfig(BaseConfig):
    model_config = SettingsConfigDict(env_prefix="DB_")

    HOST: str
    PORT: int
    USER: str
    PASSWORD: str
    NAME: str

    @property
    def database_url(self):
        return f"postgresql+asyncpg://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.NAME}"


app_config = AppConfig()
qdrant_config = QdrantConfid()
llm_config = LLMConfig()
db_config = DBConfig()