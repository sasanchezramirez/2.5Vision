import logging
from typing import Final

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dependency_injector.wiring import inject, Provide

from app.application.container import Container
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.infrastructure.entry_point.dto.response_dto import ResponseDTO
from app.infrastructure.entry_point.dto.image_dto import ImageUploadResponse
from app.infrastructure.entry_point.utils.api_response import ApiResponse
from app.domain.usecase.image_usecase import ImageUseCase

logger: Final[logging.Logger] = logging.getLogger("Image Handler")

router: Final[APIRouter] = APIRouter(
    prefix='/image',
    tags=['image'],
    responses={
        400: {"description": "Validation Error", "model": ResponseDTO},
        500: {"description": "Internal Server Error", "model": ResponseDTO},
    }
)

@router.post('/upload', response_model=ResponseDTO)
@inject
async def upload_image(
    file: UploadFile = File(...),
    image_usecase: ImageUseCase = Depends(Provide[Container.image_usecase])
) -> JSONResponse:
    """
    Sube una imagen al sistema.
    
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
        
        image_info = await image_usecase.execute(file)
        
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
            content=e.to_dict()
        )
    except Exception as e:
        logger.error(f"Excepción no manejada: {e}")
        return JSONResponse(
            status_code=500,
            content=ApiResponse.create_response(ResponseCodeEnum.KOG01)
        ) 