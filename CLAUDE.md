# CLAUDE.md - Repositorio 2.5Vision

  ## Información General del Repositorio

  **Nombre**: 2.5Vision  
  **Arquitectura**: Hexagonal (Clean Architecture)  
  **Framework Principal**: FastAPI  
  **Lenguaje**: Python  
  **Propósito**: Aplicación para estimación de material particulado (PM2.5) usando análisis de imágenes con machine learning

  ## Estructura del Proyecto

  ```
  C:\Users\sasan\Projects\2.5Vision\
  ├── .cursorrules          # Reglas y mejores prácticas de desarrollo
  ├── .gitignore           # Archivos ignorados por Git
  ├── Dockerfile           # Configuración de contenedor Docker
  ├── README.md            # Documentación del proyecto (vacío actualmente)
  ├── requirements.txt     # Dependencias de Python
  ├── venv/               # Entorno virtual de Python
  └── app/                # Código fuente de la aplicación
      ├── main.py         # Punto de entrada de la aplicación
      ├── application/    # Capa de aplicación
      ├── domain/         # Capa de dominio (lógica de negocio)
      └── infrastructure/ # Capa de infraestructura (adaptadores externos)
  ```

  ## Arquitectura Hexagonal

  El proyecto sigue los principios de arquitectura hexagonal con tres capas principales:

  ### 1. Application Layer (`app/application/`)
  - **`container.py`**: Configuración de inyección de dependencias usando `dependency-injector`
  - **`fast_api.py`**: Configuración y creación de la aplicación FastAPI
  - **`handler.py`**: Cargador dinámico de handlers
  - **`settings.py`**: Gestión de configuración y variables de entorno

  ### 2. Domain Layer (`app/domain/`)

  #### Modelos de Dominio (`app/domain/model/`)
  - **`data_sensors.py`**: Modelo para datos de sensores
  - **`gps.py`**: Modelo para coordenadas GPS
  - **`image_metadata.py`**: Metadatos de imágenes
  - **`image_config_metadata.py`**: Configuración de metadatos de imagen
  - **`pm_estimation.py`**: Modelo para estimación de material particulado
  - **`user.py`**: Modelo de usuario
  - **`zones.py`**: Definición de zonas geográficas
  - **`util/custom_exceptions.py`**: Excepciones personalizadas
  - **`util/response_codes.py`**: Códigos de respuesta

  #### Casos de Uso (`app/domain/usecase/`)
  - **`auth_usecase.py`**: Lógica de autenticación
  - **`image_usecase.py`**: Procesamiento y análisis de imágenes
  - **`masterdata_usecase.py`**: Gestión de datos maestros
  - **`user_usecase.py`**: Gestión de usuarios
  - **`util/`**: Utilidades para procesamiento de imágenes, geolocalización, JWT, etc.

  #### Gateways (`app/domain/gateway/`)
  Interfaces para servicios externos:
  - **`estimation_ml_model_gateway.py`**: Interfaz para modelo ML
  - **`persistence_gateway.py`**: Interfaz para persistencia
  - **`purpleair_gateway.py`**: Interfaz para API PurpleAir
  - **`s3_gateway.py`**: Interfaz para AWS S3
  - **`siata_gateway.py`**: Interfaz para servicio SIATA

  ### 3. Infrastructure Layer (`app/infrastructure/`)

  #### Entry Points (`app/infrastructure/entry_point/`)
  Controladores REST:
  - **`handler/auth.py`**: Endpoints de autenticación
  - **`handler/image.py`**: Endpoints para subida y procesamiento de imágenes
  - **`handler/masterdata.py`**: Endpoints de datos maestros
  - **`dto/`**: DTOs para entrada y salida
  - **`mapper/`**: Mappers para transformación de datos
  - **`validator/`**: Validadores de datos
  - **`utils/`**: Utilidades para manejo de excepciones y respuestas API

  #### Driven Adapters (`app/infrastructure/driven_adapter/`)

  **Persistence** (`persistence/`):
  - **`config/database.py`**: Configuración de SQLAlchemy con PostgreSQL
  - **`entity/`**: Entidades de base de datos
  - **`repository/`**: Repositorios de datos
  - **`service/persistence.py`**: Servicio de persistencia

  **ML Model** (`estimation_ml_model/`):
  - **`service/estimation_model_service.py`**: Servicio de estimación ML
  - **`model/`**: Modelos de machine learning (joblib)
  - **`dto/`** y **`util/`**: DTOs y utilidades

  **External APIs**:
  - **`purpleair/`**: Adaptador para API PurpleAir (sensores de calidad del aire)
  - **`siata/`**: Adaptador para servicio SIATA
  - **`s3/`**: Adaptador para AWS S3

  ## Tecnologías y Dependencias Principales

  ### Framework y Web
  - **FastAPI**: Framework web asíncrono
  - **Uvicorn**: Servidor ASGI
  - **Starlette**: Framework base de FastAPI

  ### Base de Datos
  - **SQLAlchemy**: ORM
  - **psycopg2-binary**: Driver PostgreSQL
  - **Alembic**: (configurado según .cursorrules)

  ### Autenticación y Seguridad
  - **PyJWT**: Tokens JWT
  - **bcrypt**: Hash de contraseñas
  - **passlib**: Manejo de contraseñas

  ### Machine Learning y Procesamiento de Imágenes
  - **Pillow**: Procesamiento de imágenes
  - **piexif**: Manejo de metadatos EXIF
  - **numpy**: Operaciones numéricas
  - **scikit-learn**: Machine learning
  - **joblib**: Serialización de modelos ML

  ### Cloud y Almacenamiento
  - **boto3/aioboto3**: SDK AWS
  - **aiofiles**: Manejo asíncrono de archivos

  ### Validación y Configuración
  - **pydantic**: Validación de datos y configuración
  - **pydantic-settings**: Gestión de configuración
  - **python-dotenv**: Variables de entorno

  ### Inyección de Dependencias
  - **dependency-injector**: Contenedor DI

  ## Configuración de Entorno

  ### Variables de Entorno (settings.py)
  - **ENV**: Entorno de ejecución (local/development/production)
  - **DATABASE_URL**: URL de conexión a PostgreSQL
  - **DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME**: Configuración de BD
  - **SECRET_KEY**: Clave para JWT
  - **PURPLEAIR_API_KEY, PURPLEAIR_BASE_URL**: API PurpleAir
  - **AWS_***: Configuración AWS S3

  ## Docker

  **Dockerfile**:
  - Base: Python 3.11-slim
  - Instala dependencias del sistema para PostgreSQL
  - Puerto expuesto: 8000
  - Comando: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`

  ## Funcionalidades Principales

  ### 1. Procesamiento de Imágenes
  - **Endpoint**: `POST /image/estimation`
  - **Funcionalidad**: Pipeline completo de análisis de imagen para estimar PM2.5
  - **Proceso**:
    1. Validación de archivo de imagen
    2. Extracción de metadatos EXIF (GPS, fecha)
    3. Procesamiento y normalización de imagen
    4. Obtención de datos de sensores de la zona
    5. Estimación ML de material particulado
    6. Respuesta con estimación cuantitativa y cualitativa

  ### 2. Subida de Imágenes
  - **Endpoint**: `POST /image/upload`
  - **Funcionalidad**: Subida y almacenamiento en S3 con metadatos

  ### 3. Autenticación
  - Sistema JWT para autenticación de usuarios

  ### 4. Datos Maestros
  - Gestión de zonas geográficas y configuraciones

## Integraciónes Externas

1. **PurpleAir API**: Obtención de datos de sensores de calidad del aire
2. **SIATA**: Servicio de información ambiental
3. **AWS S3**: Almacenamiento de imágenes
4. **PostgreSQL**: Base de datos principal

## Estado del Desarrollo

- **Total de archivos Python**: 95 archivos
- **Infraestructura**: 40 archivos en adaptadores
- **Modelo ML**: En desarrollo (usando valores simulados temporalmente)
- **Base de datos**: Configurada con reintentos y pool de conexiones
- **CORS**: Configurado para múltiples orígenes de desarrollo y producción

## Reglas de Desarrollo (.cursorrules)

- Python 3.12 (aunque Dockerfile usa 3.11)
- Gestión de dependencias con Poetry (aunque usa requirements.txt)
- Arquitectura hexagonal estricta
- Documentación con docstrings
- Manejo de excepciones robusto
- Type hints obligatorios
- Tests unitarios requeridos
- Seguimiento de PEP 8

## Comandos para Desarrollo

**Ejecutar aplicación**:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Docker**:
```bash
docker build -t 2.5vision .
docker run -p 8000:8000 2.5vision
```

## Git

- **Rama principal**: main
- **Estado actual**: Limpio (sin cambios pendientes)
- **Commits recientes**: Refactorizaciones y mejoras en health check

## Notas Importantes

1. **Discrepancia en versión Python**: .cursorrules especifica Python 3.12 pero Dockerfile usa 3.11
2. **Gestión de dependencias**: .cursorrules menciona Poetry pero se usa requirements.txt
3. **Modelo ML**: Actualmente en desarrollo, usando valores simulados
4. **README.md**: Está vacío y necesita documentación
5. **Configuración robusta**: Sistema de reintentos para BD y manejo de errores avanzado