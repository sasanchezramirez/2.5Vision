from typing import Dict, List, Tuple
from dataclasses import dataclass
from app.domain.model.gps import GPS

@dataclass
class Zone:
    id: int
    name: str
    boundaries: List[Tuple[float, float]]  # Lista de coordenadas (latitud, longitud) que definen el polígono

class ZoneDictionary:
    """
    Diccionario de zonas geográficas predefinidas para el sistema.
    Contiene 5 zonas predefinidas con sus respectivos límites geográficos.
    """
    def __init__(self):
        self.zones: Dict[int, Zone] = {}
        self._initialize_default_zones()

    def _initialize_default_zones(self):
        """Inicializa las 5 zonas geográficas predefinidas"""
        # Zona 1: Centro de Medellín
        self.zones[1] = Zone(
            id=1,
            name="Centro de Medellín",
            boundaries=[
                (6.253, -75.590),  # Esquina superior izquierda
                (6.253, -75.560),  # Esquina superior derecha
                (6.233, -75.560),  # Esquina inferior derecha
                (6.233, -75.590)   # Esquina inferior izquierda
            ]
        )
        
        # Zona 2: Zona Norte
        self.zones[2] = Zone(
            id=2,
            name="Zona Norte",
            boundaries=[
                (6.313, -75.590),
                (6.313, -75.540),
                (6.253, -75.540),
                (6.253, -75.590)
            ]
        )
        
        # Zona 3: Zona Sur
        self.zones[3] = Zone(
            id=3,
            name="Zona Sur",
            boundaries=[
                (6.233, -75.590),
                (6.233, -75.540),
                (6.153, -75.540),
                (6.153, -75.590)
            ]
        )
        
        # Zona 4: Zona Oriente
        self.zones[4] = Zone(
            id=4,
            name="Zona Oriente",
            boundaries=[
                (6.253, -75.540),
                (6.253, -75.500),
                (6.233, -75.500),
                (6.233, -75.540)
            ]
        )
        
        # Zona 5: Zona Occidente
        self.zones[5] = Zone(
            id=5,
            name="Zona Occidente",
            boundaries=[
                (6.253, -75.620),
                (6.253, -75.590),
                (6.233, -75.590),
                (6.233, -75.620)
            ]
        )
    
    def get_zone(self, gps: GPS) -> int:
        """
        Determina a qué zona pertenece un punto GPS.
        
        Args:
            gps: Objeto GPS con la latitud y longitud
            
        Returns:
            int: ID de la zona a la que pertenece el punto. 0 si no pertenece a ninguna zona.
        """
        if gps.latitude is None or gps.longitude is None:
            return 0
            
        for zone_id, zone in self.zones.items():
            if self._point_in_polygon(gps.latitude, gps.longitude, zone.boundaries):
                return zone_id
                
        return 0
    
    def _point_in_polygon(self, lat: float, lon: float, polygon: List[Tuple[float, float]]) -> bool:
        """
        Determina si un punto está dentro de un polígono usando el algoritmo de ray casting.
        
        Args:
            lat: Latitud del punto
            lon: Longitud del punto
            polygon: Lista de coordenadas (latitud, longitud) que definen el polígono
            
        Returns:
            bool: True si el punto está dentro del polígono, False en caso contrario
        """
        n = len(polygon)
        inside = False
        
        p1_lat, p1_lon = polygon[0]
        for i in range(1, n + 1):
            p2_lat, p2_lon = polygon[i % n]
            
            if lon > min(p1_lon, p2_lon):
                if lon <= max(p1_lon, p2_lon):
                    if lat <= max(p1_lat, p2_lat):
                        if p1_lon != p2_lon:
                            lat_intersect = (lon - p1_lon) * (p2_lat - p1_lat) / (p2_lon - p1_lon) + p1_lat
                            
                            if p1_lat == p2_lat or lat <= lat_intersect:
                                inside = not inside
                                
            p1_lat, p1_lon = p2_lat, p2_lon
            
        return inside

    