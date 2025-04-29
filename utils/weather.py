import math
import random
import carla

class Sun(object):
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._t = 0.0

    def tick(self, delta_seconds):
        self._t += 0.008 * delta_seconds
        self._t %= 2.0 * math.pi
        self.azimuth += 0.25 * delta_seconds
        self.azimuth %= 360.0
        self.altitude = (70 * math.sin(self._t)) - 20

    def __str__(self):
        return 'Sun(alt: %.2f, azm: %.2f)' % (self.altitude, self.azimuth)


class Storm(object):
    def __init__(self, precipitation):
        self._t = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0

    @staticmethod
    def _clamp(value, minimum=0.0, maximum=100.0):
        return max(minimum, min(value, maximum))

    def tick(self, delta_seconds):
        delta = (3.3 if self._increasing else -3.3) * delta_seconds
        self._t = self._clamp(delta + self._t, -250.0, 100.0)
        self.clouds = self._clamp(self._t + 40.0, 0.0, 90.0)
        self.rain = self._clamp(self._t, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = self._clamp(self._t + delay, 0.0, 85.0)
        self.wetness = self._clamp(self._t * 5, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20 else 90 if self.clouds >= 70 else 40
        if self._t == -250.0:
            self._increasing = True
        if self._t == 100.0:
            self._increasing = False

    def __str__(self):
        return 'Storm(clouds=%d%%, rain=%d%%, wind=%d%%)' % (self.clouds, self.rain, self.wind)
    
class Fog(object):
    def __init__(self, density, distance):
        self._increasing4den = True if density<50 else False
        self._increasing4dis = True if random.randint(0, 1)==0 else False
        self.density = density
        self.distance = distance
        self._t4den = 0
        self._t4dis = 0

    @staticmethod
    def _clamp(value, minimum=0.0, maximum=100.0):
        return max(minimum, min(value, maximum))

    def tick(self, delta_seconds):
        delta_den = (3.3 if self._increasing4den else -3.3) * delta_seconds
        delta_dis = (3.3 if self._increasing4dis else -3.3) * delta_seconds
        self._t4den = self._clamp(delta_den + self._t4den, -150.0, 200.0)
        self._t4dis = self._clamp(delta_dis + self._t4dis, -200.0, 170.0)

        self.density = self._clamp(self._t4den, 0.0, 100.0)
        self.distance = self._(self._t4dis, 0.0, 70.0)

        if self._increasing4den:
            if self._t4den>=200.0:
                self._increasing4den = False
        else:
            if self._t4den<=-150.0:
                self._increasing4den = True
        
        if self._increasing4dis:
            if self._t4dis>=170.0:
                self._increasing4dis = False
        else:
            if self._t4dis<=-200.0:
                self._increasing4dis = True


class Weather(object):
    def __init__(self, weather):
        self.weather = weather
        self._sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self._storm = Storm(weather.precipitation)
        self._fog = Fog(weather.fog_density, weather.fog_distance)

    def tick(self, delta_seconds):
        self._sun.tick(delta_seconds)
        self._storm.tick(delta_seconds)
        self._fog.tick(delta_seconds)
        self.weather.cloudiness = self._storm.clouds
        self.weather.precipitation = self._storm.rain
        self.weather.precipitation_deposits = self._storm.puddles
        self.weather.wind_intensity = self._storm.wind
        self.weather.fog_density = self._fog.density
        self.weather.fog_distance = self._fog.distance
        self.weather.wetness = self._storm.wetness
        self.weather.sun_azimuth_angle = self._sun.azimuth
        self.weather.sun_altitude_angle = self._sun.altitude

    def __str__(self):
        return '%s %s' % (self._sun, self._storm)

weathers = {
    "NoonGoodWeather": carla.WeatherParameters(
        cloudiness = 15, # 雲量
        precipitation  = 0,
        precipitation_deposits  = 0,
        wind_intensity = 50, 
        sun_azimuth_angle = 45, 
        sun_altitude_angle = 45,
        fog_density = 0,
        fog_distance = 0,
        wetness = 0,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NightHeavyRain": carla.WeatherParameters(
        cloudiness = 0, # 雲量
        precipitation  = 100,
        precipitation_deposits  = 100,
        wind_intensity = 100, 
        sun_azimuth_angle = 45, 
        sun_altitude_angle = -40,
        fog_density = 0,
        fog_distance = 0,
        wetness = 100,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NightHeavyRainHeavyFog": carla.WeatherParameters(
        cloudiness = 0, # 雲量
        precipitation  = 100,
        precipitation_deposits  = 100,
        wind_intensity = 100, 
        sun_azimuth_angle = 45, 
        sun_altitude_angle = -40,
        fog_density = 100,
        fog_distance = 0,
        wetness = 100,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NoonHeavyRainHeavyFog": carla.WeatherParameters(
        cloudiness = 50, # 雲量
        precipitation  = 100,
        precipitation_deposits  = 100,
        wind_intensity = 50, # 風 (影響天空雲朵和樹的擺動)
        sun_azimuth_angle = 45, # 角度
        sun_altitude_angle = 90, #日夜  -夜 +日
        fog_density = 100,
        fog_distance = 0,
        wetness = 100,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NoonHeavyFog": carla.WeatherParameters(
        cloudiness = 50, # 雲量
        precipitation  = 0,
        precipitation_deposits = 0,
        wind_intensity = 50, # 風 (影響天空雲朵和樹的擺動)
        sun_azimuth_angle = 45, # 角度
        sun_altitude_angle = 90, #日夜  -夜 +日
        fog_density = 100,
        fog_distance = 0,
        wetness = 0,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NoonModerateFog": carla.WeatherParameters(
        cloudiness = 50, # 雲量
        precipitation  = 0,
        precipitation_deposits  = 0,
        wind_intensity = 50, # 風 (影響天空雲朵和樹的擺動)
        sun_azimuth_angle = 45, # 角度
        sun_altitude_angle = 90, #日夜  -夜 +日
        fog_density = 50,
        fog_distance = 0,
        wetness = 0,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "EarlyMorningGoodWeather": carla.WeatherParameters(
        cloudiness = 15, # 雲量
        precipitation  = 0,
        precipitation_deposits  = 0,
        wind_intensity = 50, # 風 (影響天空雲朵和樹的擺動)
        sun_azimuth_angle = 45, # 角度
        sun_altitude_angle = 0, #日夜  -夜 +日
        fog_density = 0,
        fog_distance = 0,
        wetness = 0,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NoonModerateRain": carla.WeatherParameters(
        cloudiness = 32, # 雲量
        precipitation  = 65,
        precipitation_deposits  = 80,
        wind_intensity = 50, # 風 (影響天空雲朵和樹的擺動)
        sun_azimuth_angle = 45, # 角度
        sun_altitude_angle = 30, #日夜  -夜 +日
        fog_density = 0,
        fog_distance = 0,
        wetness = 50,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
    "NightGoodWeather": carla.WeatherParameters(
        cloudiness = 0, # 雲量
        precipitation  = 0,
        precipitation_deposits  = 0,
        wind_intensity = 50, # 風 (影響天空雲朵和樹的擺動)
        sun_azimuth_angle = 45, # 角度
        sun_altitude_angle = -45, #日夜  -夜 +日
        fog_density = 0,
        fog_distance = 0,
        wetness = 0,
        fog_falloff = 0,
        scattering_intensity = 0,
        mie_scattering_scale = 0,
        rayleigh_scattering_scale = 0.0331, 
    ),
}
    
