from datetime import datetime
from app.domain.model.user import User
from app.infrastructure.entry_point.dto.user_dto import (
    NewUserInput,
    GetUser,
    UserOutput,
    LoginInput,
    UpdateUserInput,
    ChangePasswordInput
)


def map_user_dto_to_user(user_dto: NewUserInput) -> User:
    """
    Mapea un DTO de nuevo usuario a un modelo de dominio.

    Args:
        user_dto: DTO con los datos del nuevo usuario

    Returns:
        User: Modelo de dominio de usuario
    """
    return User(
        username=user_dto.username.lower(),
        password=user_dto.password,
        profile_id=user_dto.profile_id,
        status_id=user_dto.status_id,
        creation_date=datetime.now().isoformat(),
        contact_info=user_dto.contact_info
    )


def map_user_to_user_output_dto(user: User) -> UserOutput:
    """
    Mapea un modelo de dominio a un DTO de salida.

    Args:
        user: Modelo de dominio de usuario

    Returns:
        UserOutput: DTO con los datos del usuario
    """
    return UserOutput(
        id=user.id,
        username=user.username,
        creation_date=user.creation_date,
        profile_id=user.profile_id,
        status_id=user.status_id
    )


def map_get_user_dto_to_user(user_dto: GetUser) -> User:
    """
    Mapea un DTO de búsqueda a un modelo de dominio.

    Args:
        user_dto: DTO con los criterios de búsqueda

    Returns:
        User: Modelo de dominio de usuario
    """
    # Si no hay username, usar un username por defecto para la validación
    # El username real se manejará en el caso de uso
    username = user_dto.username if user_dto.username else "default"
    
    return User(
        id=user_dto.id,
        username=username.lower(),
        creation_date=datetime.now().isoformat()  # Fecha por defecto para búsqueda
    )


def map_login_dto_to_user(user_dto: LoginInput) -> User:
    """
    Mapea un DTO de login a un modelo de dominio.

    Args:
        user_dto: DTO con las credenciales de login

    Returns:
        User: Modelo de dominio de usuario
    """
    return User(
        username=user_dto.username.lower(),
        password=user_dto.password,
        creation_date=datetime.now().isoformat()  # Fecha por defecto para login
    )


def map_update_user_dto_to_user(user_dto: UpdateUserInput) -> User:
    """
    Mapea un DTO de actualización a un modelo de dominio.

    Args:
        user_dto: DTO con los datos a actualizar

    Returns:
        User: Modelo de dominio de usuario
    """
    # Crear diccionario con los campos base
    user_data = {
        "id": user_dto.id,
        "username": user_dto.username.lower(),
        "profile_id": user_dto.profile_id,
        "status_id": user_dto.status_id,
        "creation_date": datetime.now().isoformat()  # Fecha por defecto para actualización
    }
    
    # Solo agregar password si no es None
    if user_dto.password is not None:
        user_data["password"] = user_dto.password
    
    return User(**user_data)

def map_change_password_dto_to_user(user_dto: ChangePasswordInput) -> User:
    """
    Mapea un DTO de cambio de contraseña a un modelo de dominio.

    Args:
        user_dto: DTO con los datos para cambiar la contraseña

    Returns:
        User: Modelo de dominio de usuario
    """
    return User(
        username=user_dto.username.lower(),
        password=user_dto.old_password
    )