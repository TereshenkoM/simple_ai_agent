from __future__ import annotations

import datetime
from typing import Annotated

from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.pg.settings import Base

created_at = Annotated[datetime.datetime, mapped_column(server_default=func.now())]


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tasks: Mapped[list[Task]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    created_at: Mapped[created_at]


class Task(Base):
    __tablename__ = "task"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"), index=True)
    prefix: Mapped[str] = mapped_column(nullable=False)
    number: Mapped[int] = mapped_column(nullable=False)
    comment: Mapped[str | None] = mapped_column(nullable=True)

    user: Mapped[User] = relationship(back_populates="tasks")
    created_at: Mapped[created_at]
