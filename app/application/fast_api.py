from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.application.container import Container
from app.application.handler import Handlers
from typing import Final


def create_app() -> FastAPI:
    """
    Crea y configura la aplicación FastAPI con sus dependencias y rutas.
    
    Returns:
        FastAPI: Instancia configurada de la aplicación FastAPI
    """
    container: Final[Container] = Container()
    app: Final[FastAPI] = FastAPI(
        title="Hexagonal Architecture FastAPI Backend",
        description="API REST implementada con FastAPI y arquitectura hexagonal",
        version="1.0.0"
    )
    
    # Configuración de CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000", "http://localhost:4200", "https://vision.gaugelife.co"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.container = container
    
    for handler in Handlers.iterator():
        app.include_router(handler.router)
        
    return app