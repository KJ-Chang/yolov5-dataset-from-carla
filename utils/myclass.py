class CarlaVector3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class CarlaLocation:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
class CarlaRotation:
    def __init__(self, pitch, yaw, roll):
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

class CarlaTransform:
    def __init__(self, location: CarlaLocation, rotation: CarlaRotation):
        self.location = location
        self.rotation = rotation

class CarlaBoundingBox:
    def __init__(self, extent: CarlaVector3D, location: CarlaLocation, rotation: CarlaRotation):
        self.extent = extent
        self.location = location
        self.rotation = rotation

class CarlaImage:
    def __init__(self, fov, width, height, raw_data, inverse_matrix = None):
        self.fov = fov
        self.width = width
        self.height = height
        self.raw_data = raw_data
        self.inverse_matrix = inverse_matrix

class CarlaRadarDetection:
    def __init__(self, depth, azimuth, altitude, velocity):
        self.depth = depth
        self.azimuth = azimuth
        self.altitude = altitude
        self.velocity = velocity

class CarlaRadar:
    def __init__(self, matrix, transform: CarlaTransform):
        self.matrix = matrix
        self.transform = transform
        self.radar_detections = []

    def insert(self, radar_detection: CarlaRadarDetection):
        self.radar_detections.append(radar_detection)

class CarlaActor:
    def __init__(self, id, type_id, bounding_box: CarlaBoundingBox, transform: CarlaTransform):
        self.id = id
        self.type_id = type_id
        self.bounding_box = bounding_box
        self.transform = transform