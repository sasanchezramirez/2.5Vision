import logging
from typing import Final

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from dependency_injector.wiring import inject, Provide
from datetime import datetime
from typing import Optional

from app.application.container import Container
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.infrastructure.entry_point.dto.response_dto import ResponseDTO
from app.infrastructure.entry_point.dto.image_dto import ImageUploadResponse
from app.infrastructure.entry_point.utils.api_response import ApiResponse
from app.domain.usecase.image_usecase import ImageUseCase
from app.infrastructure.entry_point.mapper.image_mapper import ImageMapper

logger: Final[logging.Logger] = logging.getLogger("Image Handler")

router: Final[APIRouter] = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={
        400: {"description": "Validación incorrecta", "model": ResponseDTO},
        422: {"description": "Error de validación de entidad", "model": ResponseDTO},
        500: {"description": "Error interno del servidor", "model": ResponseDTO},
    }
)

@router.post('/estimation', response_model=ResponseDTO)
@inject
async def upload_image_for_estimation(
    file: UploadFile = File(...),
    image_usecase: ImageUseCase = Depends(Provide[Container.image_usecase])
) -> JSONResponse:
    """
    Sube una imagen al sistema y realiza una estimación de material particulado.
    
    Args:
        file: Archivo de imagen a subir
        image_usecase: Caso de uso para operaciones con imágenes

    Returns:
        JSONResponse: Respuesta con el resultado de la operación
    """
    logger.info("Iniciando estimación de material particulado")
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen"
            )
        
        image_info = await image_usecase.data_pipeline(file)
        
        response_data = ImageUploadResponse(**image_info).model_dump()
        
        return JSONResponse(
            status_code=200,
            content=ApiResponse.create_response(ResponseCodeEnum.KO000, response_data)
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content=ApiResponse.create_response(ResponseCodeEnum.KOD01, str(e))
        )
    except CustomException as e:
        return JSONResponse(
            status_code=e.http_status,
            content=ApiResponse.create_error_response(e)
        )
    except Exception as e:
        logger.error(f"Excepción no manejada: {e}")
        return JSONResponse(
            status_code=500,
            content=ApiResponse.create_response(ResponseCodeEnum.KOG01)
        ) 
    
@router.post('/upload', response_model=ResponseDTO)
@inject
async def upload_image(
    file: UploadFile = File(...),
    location: str = Form(...),
    datetime_taken: datetime = Form(...),
    visibility_score: int = Form(...),
    weather_tags: Optional[str] = Form(None),
    uploader_username: Optional[str] = Form(None),
    image_usecase: ImageUseCase = Depends(Provide[Container.image_usecase])
) -> JSONResponse:
    """
    Sube una imagen al sistema y la almacena en S3.
    
    Args:
        file: Archivo de imagen a subir
        image_usecase: Caso de uso para operaciones con imágenes

    Returns:
        JSONResponse: Respuesta con el resultado de la operación
    """ 
    logger.info("Iniciando subida de imagen")
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="El archivo debe ser una imagen" 
            )
        
        image_metadata = ImageMapper.map_upload_image_request_to_image_metadata(location, datetime_taken, visibility_score, weather_tags, uploader_username)

        image_info = await image_usecase.upload_image(file, image_metadata)
        
        response_data = ImageUploadResponse(**image_info).model_dump()
        
        return JSONResponse(
            status_code=200,
            content=ApiResponse.create_response(ResponseCodeEnum.KO000, response_data)
        )
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content=ApiResponse.create_response(ResponseCodeEnum.KOD01, str(e))
        )
    except CustomException as e:
        return JSONResponse(
            status_code=e.http_status,
            content=ApiResponse.create_error_response(e)
        )
    except Exception as e:
        logger.error(f"Excepción no manejada: {e}")
        return JSONResponse(
            status_code=500,
            content=ApiResponse.create_response(ResponseCodeEnum.KOG01)
        )