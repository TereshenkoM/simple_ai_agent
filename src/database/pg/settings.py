from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.config import db_config, logger

async_engine = create_async_engine(
    url=db_config.async_database_url,
    pool_size=5,
    max_overflow=10,
)

async_session_factory = async_sessionmaker(async_engine)


class Base(DeclarativeBase):
    pass


@asynccontextmanager
async def get_session():
    session = async_session_factory()
    try:
        yield session
    except Exception as e:
        logger.error(e)
        await session.rollback()
    finally:
        await session.close()
