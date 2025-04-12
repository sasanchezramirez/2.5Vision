import logging
from typing import Final
from fastapi import UploadFile
from PIL import Image
import piexif
import io
from PIL import ImageEnhance
from PIL import ImageStat

from app.domain.usecase.util.image_processing_utils import ImageProcessingUtils
from app.domain.usecase.util.roi_utils import ROIUtils
from app.domain.usecase.util.statistical_utils import StatisticalUtils
from app.domain.usecase.util.geolocalization_utils import GeolocalizationUtils
from app.domain.gateway.siata_gateway import SiataGateway
from app.domain.gateway.purpleair_gateway import PurpleAirGateway
from app.domain.gateway.estimation_ml_model_gateway import EstimationMLModelGateway
from app.domain.gateway.s3_gateway import S3Gateway
from app.domain.gateway.persistence_gateway import PersistenceGateway
from app.domain.model.gps import GPS
from app.domain.model.pm_estimation import PMEstimation
from app.domain.model.zones import ZoneDictionary
from app.domain.model.data_sensors import DataSensor
from app.domain.model.image_metadata import ImageMetadata
from app.domain.model.image_config_metadata import ImageConfigMetadata

logger: Final[logging.Logger] = logging.getLogger("Image UseCase")

class ImageUseCase:
    def __init__(self, siata_gateway: SiataGateway, purpleair_gateway: PurpleAirGateway, estimation_ml_model_gateway: EstimationMLModelGateway, s3_gateway: S3Gateway, persistence_gateway: PersistenceGateway):
        self.siata_gateway = siata_gateway
        self.zone_dictionary = ZoneDictionary()
        self.purpleair_gateway = purpleair_gateway
        self.estimation_ml_model_gateway = estimation_ml_model_gateway
        self.s3_gateway = s3_gateway
        self.persistence_gateway = persistence_gateway
    async def data_pipeline(self, file: UploadFile) -> PMEstimation:
        """
        Pipeline de procesamiento de imagen para detectar la cantidad de material particulado presente.
        - Normaliza la imagen
        - Obtiene los datos de la imagen
        - Obtiene la zona geográfica de la imagen~
        - Obtiene los datos del material particulado de la zona
        - Obtiene la estimación de la cantidad de material particulado presente
        - Obtiene la estimación cualitativa de la cantidad de material particulado presente

        Args:
            file: Archivo de imagen a subir 
            
        Returns:
            PMEstimation: Estimación de la cantidad de material particulado presente
        """
        normalized_image = self._image_normalization(file)
        gps_data = self._get_gps_data(file)
        roi_image = ROIUtils.get_roi(normalized_image)
        feature_vector = self._image_processing(roi_image)
        if gps_data.zone and gps_data.zone != 0:
            pm_data = self._get_pm_data(gps_data)
        pm_quantitative_estimation = self._get_pm_quantitative_estimation(pm_data, feature_vector)
        pm_estimation = self._get_pm_qualitative_estimation(pm_quantitative_estimation)
        return pm_estimation
    
    async def upload_image(self, file: UploadFile, image_metadata: ImageMetadata) -> ImageMetadata:
        """
        Sube una imagen a un bucket de AWS S3.  

        Args:
            file: Archivo de imagen a subir
            image_metadata: Metadatos de la imagen
        Returns:
            dict: Información sobre la imagen subida (URL, nombre, tipo, tamaño)
        """
        logger.info("Inicia el flujo de carga de imágenes")
        image_pilow = await ImageProcessingUtils.format_upload_file_to_image(file)
        gps_data = self._get_gps_data(image_pilow)
        if gps_data.latitude or gps_data.longitude: 
            has_metadata = True
            image_metadata.latitude = gps_data.latitude
            image_metadata.longitude = gps_data.longitude
        else:
            has_metadata = False
            image_metadata.latitude = None
            image_metadata.longitude = None
        image_config_metadata = self._get_image_configuration_metadata(image_pilow)
        normalized_image = self._image_normalization(file, image_config_metadata)
        file_details = self.s3_gateway.upload_image(file, has_metadata)
        image_metadata.image_url = file_details.image_url
        image_metadata.image_name = file_details.image_name
        self.persistence_gateway.create_image_metadata(image_metadata)
        logger.info(f"Imagen subida a S3: {image_metadata.image_url}")
        return image_metadata
    
    def _image_normalization(self, file_or_image, image_config_metadata: ImageConfigMetadata) -> Image:
        """
        Normaliza la imagen como preprocesamiento para el modelo de detección de material particulado.
        - Ajusta el contraste
        - Ajusta el balance de blancos
        - Reduce el ruido
    
        Args:
            file_or_image: Archivo o imagen a normalizar
            image_config_metadata: Metadatos de configuración de la imagen
            
        Returns:
            Image: Imagen normalizada
        """
        # Si es un UploadFile, convertirlo a Image
        if isinstance(file_or_image, UploadFile):
            # Convertir UploadFile a PIL.Image
            try:
                # Leer el contenido del archivo
                file_contents = file_or_image.file.read()
                # Volver a la posición inicial para futuras lecturas
                file_or_image.file.seek(0)
                # Crear objeto Image
                image = Image.open(io.BytesIO(file_contents))
            except Exception as e:
                logger.error(f"Error al convertir UploadFile a Image: {e}")
                # En caso de error, usar un enfoque alternativo
                stat = ImageStat.Stat(file_or_image)
                mean_lum = sum(stat.mean) / len(stat.mean)
                desired_mean = 128
                if mean_lum < 1: mean_lum = 1
                scale = desired_mean / mean_lum
                enhancer = ImageEnhance.Brightness(file_or_image)
                return enhancer.enhance(scale)
        else:
            # Ya es un objeto Image
            image = file_or_image
            
        # Aplicar correcciones
        try:
            exposure_corrected_image = ImageProcessingUtils.exposure_correction(image, image_config_metadata)
            #white_balanced_image = ImageProcessingUtils.white_balance_correction(exposure_corrected_image)
            #noise_reduced_image = ImageProcessingUtils.noise_reduction(white_balanced_image)
            #return noise_reduced_image
            return exposure_corrected_image
        except Exception as e:
            logger.error(f"Error en la normalización de imagen: {e}")
            # En caso de error, devolver la imagen original
            return image
    def _image_processing(self, image: Image) -> list[float]:
        """
        Procesa la imagen para detectar la cantidad de material particulado presente.
        - Mapea los colores de la imagen
        - Obtiene los píxeles de la imagen
        - Obtiene la mediana y la media de los píxeles
        - Devuelve un vector de características

        Args:
            image: Imagen a procesar
            
        Returns:
            list[float]: Vector de características
        """
        color_mapping_image = ImageProcessingUtils.color_mapping(image)
        pixel_values = ROIUtils.get_pixels(color_mapping_image)
        median = StatisticalUtils.get_median(pixel_values)
        mean = StatisticalUtils.get_mean(pixel_values)
        feature_vector = [median, mean]
        return feature_vector
    
    def _get_image_configuration_metadata(self, image: Image) -> ImageConfigMetadata:
        """
        Obtiene los metadatos de la imagen.
        - Obtiene los metadatos de configuracion de la cámara con la que se tomó la imagen.
        Args:
            image: Imagen a procesar

        Returns:
            ImageConfigMetadata: Metadatos de la imagen
        """
        image_metadata = ImageProcessingUtils.get_image_metadata(image)
        datetime_original = image_metadata.get("DateTimeOriginal")
        if datetime_original and isinstance(datetime_original, str):
            parts = datetime_original.split(" ")
            if len(parts) == 2:
                date_part = parts[0].replace(":", "-")
                time_part = parts[1]
                datetime_original = f"{date_part} {time_part}"     
        image_config_metadata = ImageConfigMetadata(
            camera_make= image_metadata.get("Make") if image_metadata.get("Make") else None,
            camera_model=image_metadata.get("Model") if image_metadata.get("Model") else None,
            iso=image_metadata.get("ISOSpeedRatings") if image_metadata.get("ISOSpeedRatings") else None,
            shutter_speed=image_metadata.get("ExposureTime") if image_metadata.get("ExposureTime") else None,
            aperture=image_metadata.get("FNumber") if image_metadata.get("FNumber") else None,
            exposure_compensation=image_metadata.get("ExposureBiasValue") if image_metadata.get("ExposureBiasValue") else None,
            focal_length=image_metadata.get("FocalLength") if image_metadata.get("FocalLength") else None,
            datetime_original=datetime_original
        )
        return image_config_metadata
        

    def _get_gps_data(self, image: Image) -> GPS:
        """
        Obtiene los datos de la imagen.
        - Obtiene los datos de la imagen
        - Obtiene la latitud y la longitud de la imagen
        - Obtiene la zona geográfica de la imagen

        Args:
            image: Imagen a verificar

        Returns:
            GPS: Datos de la imagen
        """
        logger.info("Inicia el flujo de obtención de datos de la imagen")
        image_metadata = ImageProcessingUtils.get_image_metadata(image)
        logger.info(f"Metadatos de la imagen obtenidos: {image_metadata}")
        try:
            gps = image_metadata["GPSInfo"]
        except KeyError:
                return GPS(latitude=None, longitude=None, zone=0)
        gps_latitude = gps["GPSLatitude"]
        gps_latitude_ref = gps["GPSLatitudeRef"]
        gps_longitude = gps["GPSLongitude"]
        gps_longitude_ref = gps["GPSLongitudeRef"]
        latitude = GeolocalizationUtils.dms_to_decimal(gps_latitude, gps_latitude_ref)
        longitude = GeolocalizationUtils.dms_to_decimal(gps_longitude, gps_longitude_ref)        
        gps_obj = GPS(latitude=latitude, longitude=longitude, zone=0)
        zone = self.zone_dictionary.get_zone(gps_obj)
        logger.info(f"Zona geográfica obtenida: {zone}, para la latitud: {latitude} y la longitud: {longitude}")
        return GPS(latitude=latitude, longitude=longitude, zone=zone)
    
    def _get_pm_data(self, gps_data: GPS) -> list[DataSensor]:
        """
        Obtiene los datos de material particulado de la zona estipulada usando la información de SIATA.   

        Args:
            gps_data: Datos de la zona geográfica de la imagen

        Returns:
            list[float]: Datos del material particulado de la zona
        """
        siata_pm_data = self.siata_gateway.get_data_by_zone(gps_data.zone)
        purpleair_pm_data = self.purpleair_gateway.get_data_by_zone(gps_data.zone)
        list_pm_data = [siata_pm_data, purpleair_pm_data]
        return list_pm_data


    def _get_pm_quantitative_estimation(self, pm_data: list[DataSensor], feature_vector: list[float]) -> PMEstimation:
        """
        Obtiene la estimación de la cantidad de material particulado presente.

        Args:
            pm_data: Datos del material particulado de la zona
            feature_vector: Vector de características

        Returns:
            PMEstimation: Estimación de la cantidad de material particulado presente y confianza de la estimación.
        """
        pm_estimation = self.estimation_ml_model_gateway.estimate_pm(pm_data, feature_vector)
        return pm_estimation

    def _get_pm_qualitative_estimation(self, pm_estimation: PMEstimation) -> PMEstimation:
        """
        Obtiene la estimación cualitativa de la cantidad de material particulado presente.

        Args:
            pm_estimation: Estimación de la cantidad de material particulado presente

        Returns:
            PMEstimation: Estimación cualitativa de la cantidad de material particulado presente
        """
        pass