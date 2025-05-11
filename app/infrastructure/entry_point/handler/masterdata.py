import logging
from typing import Final

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from dependency_injector.wiring import inject, Provide

from app.application.container import Container
from app.domain.model.util.custom_exceptions import CustomException
from app.domain.model.util.response_codes import ResponseCodeEnum
from app.infrastructure.entry_point.dto.response_dto import ResponseDTO
from app.infrastructure.entry_point.utils.api_response import ApiResponse
from app.domain.usecase.masterdata_usecase import MasterdataUseCase

logger: Final[logging.Logger] = logging.getLogger("Masterdata Handler")

router: Final[APIRouter] = APIRouter(
    prefix='/masterdata',
    tags=['masterdata'],
    responses={
        
    }
)

@router.get('/total-images-uploaded', response_model=ResponseDTO)
@inject
async def get_total_images_uploaded(
    masterdata_usecase: MasterdataUseCase = Depends(Provide[Container.masterdata_usecase])
) -> JSONResponse:
    """
    Obtiene el total de imágenes subidas por los usuarios.

    Returns:
        JSONResponse: Respuesta con el total de imágenes subidas
    """
    logger.info("Iniciando obtención del total de imágenes subidas")
    try:
        total_images = await masterdata_usecase.get_total_images_uploaded()
        return JSONResponse(
            status_code=200,
            content=ApiResponse.create_response(ResponseCodeEnum.KO000, total_images)
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


@router.get('/top-contributors', response_model=ResponseDTO)
@inject
async def get_top_contributors(
    masterdata_usecase: MasterdataUseCase = Depends(Provide[Container.masterdata_usecase])
) -> JSONResponse:
    """
    Obtiene los top 3 contribuyentes con más imágenes subidas.

    Returns:    
        JSONResponse: Respuesta con los top 3 contribuyentes con más imágenes subidas
    """
    logger.info("Iniciando obtención de los top 3 contribuyentes con más imágenes subidas")
    try:
        top_users = await masterdata_usecase.get_top_users_by_images_uploaded()
        return JSONResponse(
            status_code=200,
            content=ApiResponse.create_response(ResponseCodeEnum.KO000, top_users)
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
    
@router.get('/total-contributions-by-user', response_model=ResponseDTO)
@inject
async def get_total_contributions_by_user(
    username: str,
    masterdata_usecase: MasterdataUseCase = Depends(Provide[Container.masterdata_usecase])
) -> JSONResponse:
    """
    Obtiene el total de contribuciones por un usuario.

    Returns:
        JSONResponse: Respuesta con el total de contribuciones por un usuario
    """
    logger.info("Iniciando obtención del total de contribuciones por un usuario")
    try:
        total_contributions = await masterdata_usecase.get_total_images_uploaded_by_user(username)
        return JSONResponse(
            status_code=200,
            content=ApiResponse.create_response(ResponseCodeEnum.KO000, total_contributions)
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
