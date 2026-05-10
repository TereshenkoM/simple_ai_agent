from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel


class AgentRequest(BaseModel):
    user_id: int
    query: str


class State(TypedDict):
    user_id: int
    messages: Annotated[list, add_messages]
