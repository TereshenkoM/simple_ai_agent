from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import db_config

async_engine = create_async_engine(
    url=db_config.database_url,
    pool_size=5,
    max_overflow=10,
)

async_session = async_sessionmaker(async_engine)


class Base(DeclarativeBase):
    pass
