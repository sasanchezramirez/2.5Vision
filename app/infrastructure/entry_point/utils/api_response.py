from app.domain.model.util.response_codes import ResponseCodeEnum, ResponseCode
from app.domain.model.util.custom_exceptions import CustomException
from datetime import datetime
from typing import Any, Dict, List, Union

class ApiResponse:
    @staticmethod
    def create_response(response_enum: ResponseCodeEnum, data=None):
        response_code = ResponseCode(response_enum)
        if data is not None:
            data = ApiResponse._convert_datetime_to_string(data)
        return {
            "apiCode": response_code.code,
            "data": data,
            "message": response_code.message,
            "status": response_code.http_status == 200
        }
    
    @staticmethod
    def create_error_response(exception: CustomException):
        return {
            "apiCode": exception.code,
            "data": None,
            "message": exception.message,
            "status": exception.http_status == 200
        }
    
    @staticmethod
    def _convert_datetime_to_string(obj: Any) -> Any:
        """
        Convierte recursivamente cualquier objeto datetime a string ISO y objetos Pydantic a diccionarios.
        
        Args:
            obj: Objeto que puede contener datetimes o modelos Pydantic
            
        Returns:
            El mismo objeto con los datetimes convertidos a strings y modelos Pydantic a diccionarios
        """
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'model_dump') and callable(getattr(obj, 'model_dump')):
            # Es un modelo Pydantic v2
            return ApiResponse._convert_datetime_to_string(obj.model_dump())
        elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
            # Es un modelo Pydantic v1
            return ApiResponse._convert_datetime_to_string(obj.dict())
        elif isinstance(obj, dict):
            return {k: ApiResponse._convert_datetime_to_string(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ApiResponse._convert_datetime_to_string(item) for item in obj]
        else:
            return obj
