from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentRequest(BaseModel):
    user_id: int | None = None
    query: str


class State(TypedDict):
    user_id: int
    task_id: int
    number: int
    messages: Annotated[list, add_messages]
