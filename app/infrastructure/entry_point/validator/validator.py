
from app.infrastructure.entry_point.dto.user_dto import (
    NewUserInput,
    GetUser,
    LoginInput,
    UpdateUserInput
)


def is_valid_username(username: str) -> bool:
    """
    Valida el formato de un nombre de usuario de manera más flexible.
    
    Args:
        username: Nombre de usuario a validar
        
    Returns:
        bool: True si el formato es válido
    """
    length = len(username)
    if length < 3 or length > 20:
        return False
    return True

def validate_new_user(user: NewUserInput) -> bool:
    """
    Valida los datos de un nuevo usuario.

    Args:
        user: DTO con los datos del nuevo usuario

    Returns:
        bool: True si la validación es exitosa

    Raises:
        ValueError: Si los datos no son válidos
    """
    if not user.username:
        raise ValueError("El nombre de usuario es obligatorio")
    
    if not is_valid_username(user.username):
        raise ValueError("El formato del nombre de usuario no es válido")
    
    if not user.password or len(user.password) < 8:
        raise ValueError("La contraseña debe tener al menos 8 caracteres")
    
    if not user.profile_id or user.profile_id <= 0:
        raise ValueError("El ID del perfil debe ser mayor que 0")
    
    if not user.status_id or user.status_id <= 0:
        raise ValueError("El ID del estado debe ser mayor que 0")
    
    return True


def validate_get_user(user: GetUser) -> bool:
    """
    Valida los criterios de búsqueda de usuario.

    Args:
        user: DTO con los criterios de búsqueda

    Returns:
        bool: True si la validación es exitosa

    Raises:
        ValueError: Si los datos no son válidos
    """
    if not user.id and not user.username:
        raise ValueError("Debe proporcionar un ID o nombre de usuario")
    
    if user.username and not is_valid_username(user.username):
        raise ValueError("El formato del nombre de usuario no es válido")
    
    return True


def validate_login(user: LoginInput) -> bool:
    """
    Valida las credenciales de inicio de sesión.

    Args:
        user: DTO con las credenciales de login

    Returns:
        bool: True si la validación es exitosa

    Raises:
        ValueError: Si los datos no son válidos
    """
    if not user.username:
        raise ValueError("El nombre de usuario es obligatorio")
    
    if not is_valid_username(user.username):
        raise ValueError("El formato del nombre de usuario no es válido")
    
    if not user.password or len(user.password) < 8:
        raise ValueError("La contraseña debe tener al menos 8 caracteres")
    
    return True


def validate_update_user(user: UpdateUserInput) -> bool:
    """
    Valida los datos de actualización de usuario.

    Args:
        user: DTO con los datos a actualizar

    Returns:
        bool: True si la validación es exitosa

    Raises:
        ValueError: Si los datos no son válidos
    """
    if not user.id or user.id <= 0:
        raise ValueError("El ID del usuario es obligatorio y debe ser mayor que 0")
    
    if user.username and not is_valid_username(user.username):
        raise ValueError("El formato del nombre de usuario no es válido")
    
    if user.password and len(user.password) < 8:
        raise ValueError("La contraseña debe tener al menos 8 caracteres")
    
    if user.profile_id is not None and user.profile_id <= 0:
        raise ValueError("El ID del perfil debe ser mayor que 0")
    
    if user.status_id is not None and user.status_id <= 0:
        raise ValueError("El ID del estado debe ser mayor que 0")
    
    return True
