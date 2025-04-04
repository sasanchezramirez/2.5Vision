from pydantic import BaseModel, Field

class SensorsByZone(BaseModel):
    zone: int = Field(
        default=0,
        description="Zona geográfica preestablecida en el área metropolitana de Medellín"
    )
    sensors: list[int] = Field(
        default=[],
        description="Lista de sensores que pueden ser usados en la zona"
    )

class SensorsByZoneDictionary:
    def __init__(self):
        self.sensors_by_zone = {}
        self._initialize_default_sensors_by_zone()

    def _initialize_default_sensors_by_zone(self):
        "Zona centro de Medellín"
        self.sensors_by_zone[1] = SensorsByZone(
            zone=1,
            sensors=[11394, 52163, 27552]
        )
        "Zona norte de Medellín"
        self.sensors_by_zone[2] = SensorsByZone(
            zone=2,
            sensors=[27647, 1630]
        )
        "Zona sur de Medellín"
        self.sensors_by_zone[3] = SensorsByZone(
            zone=3,
            sensors=[86891]
        )
        "Zona oriente de Medellín"
        self.sensors_by_zone[4] = SensorsByZone(
            zone=4,
            sensors=[11394, 52163]
        )
        "Zona occidente de Medellín"
        self.sensors_by_zone[5] = SensorsByZone(
            zone=5,
            sensors=[27552, 27597, 27597, 1630]
        )

    def get_sensors_by_zone(self, zone: int) -> SensorsByZone:
        return self.sensors_by_zone[zone]
