import logging
from functools import wraps
from typing import Callable, TypeVar, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from app.application.settings import settings 
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')

def retry_on_db_error(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorador para reintentar operaciones de base de datos que fallan con errores transitorios.
    
    Args:
        max_retries: Número máximo de reintentos
        initial_delay: Tiempo de espera inicial en segundos
        max_delay: Tiempo máximo de espera entre reintentos
        backoff_factor: Factor de multiplicación para el tiempo de espera
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except OperationalError as e:
                    last_exception = e
                    if attempt == max_retries:
                        logger.error(f"Error de base de datos después de {max_retries} reintentos: {str(e)}")
                        raise
                    
                    logger.warning(
                        f"Error de base de datos (intento {attempt + 1}/{max_retries + 1}). "
                        f"Reintentando en {delay} segundos..."
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                except SQLAlchemyError as e:
                    logger.error(f"Error de SQLAlchemy no recuperable: {str(e)}")
                    raise
                
            raise last_exception
        return wrapper
    return decorator

SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,  # Verifica conexiones antes de usarlas
    pool_recycle=3600,   # Recicla conexiones después de una hora
    pool_size=10,        # Tamaño del pool de conexiones
    max_overflow=20      # Conexiones adicionales permitidas
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_session():
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
