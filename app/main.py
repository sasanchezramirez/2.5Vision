from fastapi import FastAPI
from app.infrastructure.entry_point.routes import router as item_router

def create_app() -> FastAPI:
    app = FastAPI(title="2.5Vision - BackEnd Repository", version="1.0.0")

    # Incluir routers
    app.include_router(item_router)

    # Agregar middleware, CORS, etc. si es necesario

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
