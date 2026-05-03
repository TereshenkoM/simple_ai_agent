from fastapi import FastAPI
import uvicorn
from src.config import  app_config
from src.routes import agents_router


def run_app(app: FastAPI) -> None:
    uvicorn.run(
        app,
        host=app_config.host,
        port=app_config.port
    )


if __name__ == "__main__":
    app = FastAPI(
        title="Простой ИИ агент",
        version=app_config.version,
    )
    app.include_router(agents_router)
    run_app(app)