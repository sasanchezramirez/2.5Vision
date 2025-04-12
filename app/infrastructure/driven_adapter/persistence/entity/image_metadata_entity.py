from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.sql import func
from typing import Optional
from datetime import datetime

from app.infrastructure.driven_adapter.persistence.config.database import Base
from app.domain.model.image_metadata import ImageMetadata

class ImageMetadataEntity(Base):
    """
    Entidad de base de datos para metadatos de imágenes.
    
    Esta clase representa la tabla de metadatos de imágenes en la base de datos
    utilizando SQLAlchemy como ORM.
    """
    __tablename__ = "image_metadata"
    __table_args__ = {"schema": "vision_2_5"} 

    id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    latitude: float = Column(Float, nullable=False)
    longitude: float = Column(Float, nullable=False)
    datetime_taken: datetime = Column(DateTime, nullable=False)
    visibility_score: int = Column(Integer, nullable=False)
    weather_tags: Optional[str] = Column(String, nullable=True)
    uploader_username: Optional[str] = Column(String, nullable=True)
    image_url: str = Column(String, nullable=False)
    image_name: str = Column(String, nullable=False)

    def __init__(self, latitude: float, longitude: float, datetime_taken: datetime, visibility_score: int, weather_tags: Optional[str] = None, uploader_username: Optional[str] = None, image_url: str = "", image_name: str = "") -> None:
        self.latitude = latitude
        self.longitude = longitude
        self.datetime_taken = datetime_taken
        self.visibility_score = visibility_score
        self.weather_tags = weather_tags
        self.uploader_username = uploader_username
        self.image_url = image_url
        self.image_name = image_name    

    def __repr__(self) -> str:
        return f"<ImageMetadataEntity(id={self.id}, latitude={self.latitude}, longitude={self.longitude}, datetime_taken={self.datetime_taken}, visibility_score={self.visibility_score}, weather_tags={self.weather_tags}, uploader_username={self.uploader_username}, image_url={self.image_url}, image_name={self.image_name})>"
