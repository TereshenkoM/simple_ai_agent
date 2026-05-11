import ast
import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from sqlalchemy import select

from src.config import llm_config, logger
from src.database.pg.models import Task, User
from src.database.pg.settings import get_session
from src.schemas import State


class Tools:
    @staticmethod
    @tool
    async def create_task(user_id: int, prefix: str = "TASK"):
        """Создаёт задачу и возвращает task_id и номер (prefix-number)."""
        async with get_session() as session:
            user_stmt = select(User).where(User.id == user_id)
            result = await session.execute(user_stmt)
            user = result.scalar_one_or_none()

            if not user:
                logger.info("Пользователь не найден")
                raise ValueError(f"User with id={user_id} not found")

            normalized_prefix = (prefix or "TASK").strip().upper()
            if normalized_prefix != "TASK":
                logger.warning(
                    "Неподдерживаемый prefix=%s для create_task, принудительно использую TASK",
                    normalized_prefix,
                )
                normalized_prefix = "TASK"

            last_number_stmt = (
                select(Task.number)
                .where(Task.prefix == normalized_prefix)
                .order_by(Task.number.desc())
                .limit(1)
            )
            last_number_result = await session.execute(last_number_stmt)
            last_number = last_number_result.scalar_one_or_none() or 0

            task = Task(
                user_id=user_id,
                prefix=normalized_prefix,
                number=last_number + 1,
            )
            session.add(task)
            await session.commit()

            return {
                "task_id": task.id,
                "prefix": task.prefix,
                "number": task.number,
                "task_key": f"{task.prefix}-{task.number}",
            }

    @staticmethod
    @tool
    async def add_comment(task_id: int, comment: str):
        """Добавляет комментарий к задаче по ID"""
        async with get_session() as session:
            stmt = select(Task).where(Task.id == task_id)
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()

            if not task:
                raise ValueError(f"Task with id={task_id} not found")

            task.comment = comment
            await session.commit()

        return {"task_id": task_id, "comment": comment}

    @staticmethod
    def parse_tool_content(content) -> dict:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return {}
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(content)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            return {}


class TaskService:
    def __init__(self):
        self._tools = [Tools.create_task, Tools.add_comment]
        self._tool_node = ToolNode(self._tools)
        self._llm = ChatOllama(model=llm_config.llm_model, temperature=0)
        self._llm_with_tools = self._llm.bind_tools(self._tools)

    async def _call_model(self, state: State):
        system = SystemMessage(
            content=(
                "Ты — агент управления задачами. Для любых действий с БД ты ОБЯЗАН вызывать tools.\n\n"
                f"Контекст:\n- user_id: {state['user_id']}\n- текущий task_id (если уже известен): {state.get('task_id') or 'не задан'}\n\n"
                "Доступные tools:\n"
                "- create_task(user_id:int, prefix:str='TASK') -> возвращает {task_id, task_key, number}\n"
                "- add_comment(task_id:int, comment:str) -> возвращает подтверждение\n\n"
                "Инструкции:\n"
                "1) Если пользователь просит создать задачу — сразу вызови `create_task(user_id=<user_id>, prefix='TASK')`.\n"
                "   После tool-вызова финальный ответ строго: 'Задача создана, номер задачи: TASK-<number>'.\n"
                "2) Если пользователь просит добавить комментарий:\n"
                "   - Если в запросе явно указан task_id (целое число) — используй его.\n"
                "   - Иначе, если в контексте уже есть текущий task_id — используй его.\n"
                "   - Иначе задай один уточняющий вопрос: 'К какой задаче (task_id)?'.\n"
                "   Когда task_id известен — сразу вызови `add_comment(task_id=<task_id>, comment=<comment>)`.\n"
                "   После tool-вызова финальный ответ строго: 'Комментарий добавлен к задаче task_id=<task_id>'.\n"
                "3) Prefix всегда 'TASK'. Не используй текст пользователя как prefix.\n"
                "4) Не выдумывай task_id/number и не описывай процесс. Либо tool-вызов, либо одна финальная фраза по шаблону."
            )
        )

        messages = [system] + state["messages"]

        return {"messages": [await self._llm_with_tools.ainvoke(messages)]}

    @staticmethod
    async def _update_task_id(state: State):
        for message in reversed(state["messages"]):
            if isinstance(message, ToolMessage) and message.name == "create_task":
                content = Tools.parse_tool_content(message.content)
                if isinstance(content.get("task_id"), int):
                    return {"task_id": content["task_id"]}
                try:
                    return {"task_id": int(str(message.content).strip())}
                except ValueError:
                    return {"task_id": state["task_id"]}

        return {"task_id": state["task_id"]}

    async def _compile_workflow(self):
        workflow = StateGraph(State)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._tool_node)
        workflow.add_node("update", self._update_task_id)

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent", lambda state: "tools" if state["messages"][-1].tool_calls else "__end__"
        )
        workflow.add_edge("tools", "update")
        workflow.add_edge("update", "agent")

        return workflow.compile()

    async def process(self, question: str, user_id: int):
        async with get_session() as session:
            task_stmt = select(Task.id).order_by(Task.id.desc()).limit(1)
            result = await session.execute(task_stmt)
            row = result.first()
            task_id = row[0] if row else 0

            if not task_id:
                task_id = 0

        state = State(
            messages=[],
            task_id=task_id,
            user_id=user_id,
        )
        workflow = await self._compile_workflow()

        state = await workflow.ainvoke({**state, "messages": [HumanMessage(content=question)]})

        # Финальный ответ для create_task берём из БД (а не из текста модели),
        # чтобы исключить галлюцинации и проблемы сериализации tool-ответа.
        if any(
            isinstance(message, ToolMessage) and message.name == "create_task"
            for message in state["messages"]
        ):
            async with get_session() as session:
                row = (
                    await session.execute(
                        select(Task.prefix, Task.number)
                        .where(Task.user_id == user_id, Task.prefix == "TASK")
                        .order_by(Task.id.desc())
                        .limit(1)
                    )
                ).first()
            prefix, number = row if row else ("TASK", 1)
            prefix_str = str(prefix or "TASK").strip().upper()
            try:
                number_int = int(number) if number is not None else 1
            except (TypeError, ValueError):
                number_int = 1
            if number_int <= 0:
                number_int = 1
            return f"Задача создана, номер задачи: {prefix_str}-{number_int}."

        last_message = state["messages"][-1]
        return getattr(last_message, "content", str(last_message))
