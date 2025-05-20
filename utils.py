
import math

# Inflate straight‑line (great‑circle) distance to approximate road mileage
ROAD_FACTOR = 1.3

def haversine(lon1, lat1, lon2, lat2):
    """Return great‑circle distance in miles between two (lon, lat) points."""
    R = 3958.8  # Earth radius in miles
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))
