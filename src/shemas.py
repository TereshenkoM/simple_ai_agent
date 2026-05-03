from pydantic import BaseModel
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentRequest(BaseModel):
    user_id: int
    query: str


class State(TypedDict):
    user_id: int
    messages: Annotated[list, add_messages]
