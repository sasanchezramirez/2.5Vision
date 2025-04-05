from app.domain.model.purple_air_data_sensors import NetworkDataSensor
from app.infrastructure.driven_adapter.purpleair.dto.sensor_data_response_dto import SensorDataResponseDto

class PurpleAirMapper:
    @staticmethod
    def map_sensor_data_response_dto_to_data_sensor(sensor_data_response_dto: SensorDataResponseDto) -> NetworkDataSensor:
        return NetworkDataSensor(
            sensor_index=sensor_data_response_dto.sensor.sensor_index,
            name=sensor_data_response_dto.sensor.name,
            latitude=sensor_data_response_dto.sensor.latitude,
            longitude=sensor_data_response_dto.sensor.longitude,
            pm1_0=sensor_data_response_dto.sensor.pm1_0,
            pm2_5=sensor_data_response_dto.sensor.stats.pm2_5_30minute, #promedio de 30 minutos
            pm10_0=sensor_data_response_dto.sensor.pm10_0,
            humidity=sensor_data_response_dto.sensor.humidity,
            temperature=sensor_data_response_dto.sensor.temperature,
            pressure=sensor_data_response_dto.sensor.pressure,
            confidence=sensor_data_response_dto.sensor.confidence,
            network="PURPLEAIR"
        )
