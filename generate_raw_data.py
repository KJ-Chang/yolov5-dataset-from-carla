"""
Generate and save raw sensor data for training!
"""

import glob
import os
import sys

import carla.libcarla

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_b
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_f
    from pygame.locals import K_g
    from pygame.locals import K_h
    from pygame.locals import K_i
    from pygame.locals import K_l
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_o
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_t
    from pygame.locals import K_v
    from pygame.locals import K_w
    from pygame.locals import K_x
    from pygame.locals import K_z
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

import time

import cv2
import pickle

import utils.file as file
import utils.fusion as fusion
import utils.weather as weather
import utils.myclass as myclass

import config.path as path_config

from queue import Queue
rgb_cam_queue = Queue()
rdr_queue = Queue()
depth_cam_queue = Queue()
# semanticsegmentation_cam_queue = Queue()
instancesegmentation_cam_queue = Queue()

g_radar_sensor = None
g_camera_sensor = None

K = None


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2, 3]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def change_vehicle_walker_behavior(speed, physics, sim_world, all_actors, all_id):
    # 雙是ai control 單是walker
    for i in range(1, len(all_id), 2):
        if (not all_actors[i].is_alive) or (not all_actors[i-1].is_alive):
            continue
        all_bones = []
        for bone_out in all_actors[i].get_bones().bone_transforms:
            all_bones.append((bone_out.name, bone_out.relative))
        if speed == 0.0:
            all_actors[i-1].set_max_speed(speed)
            all_actors[i].set_bones(carla.WalkerBoneControlIn(all_bones))
            all_actors[i].show_pose()
        else:
            all_actors[i-1].set_max_speed(random.uniform(1.4, 5.0))
            all_actors[i].hide_pose()


    vehicles = sim_world.get_actors().filter('vehicle.*')
    for actor in vehicles:
        if not actor.is_alive:
            continue
        try:
            actor.set_simulate_physics(physics)
        except RuntimeError as e:
            print(f'Error: {e}')
            pass
    
    freeze = not physics
    traffic_lights = sim_world.get_actors().filter('traffic.traffic_light')
    for tl in traffic_lights:
        tl.freeze(freeze)


class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        self.sync = args.sync
        self.args = args
        self.actor_role_name = args.rolename
        self.cam_done = False
        self.rdr_done = False
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.radar_sensor = None
        self.rgb_camera = None
        self.depth_camera = None
        self.semanticsegmentation_camera = None
        self.instancesegmentation_camera = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._actor_generation = args.generation
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False
        self.doors_are_open = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        rgb_cam_index = self.rgb_camera.index if self.rgb_camera is not None else 0
        depth_cam_index = self.depth_camera.index if self.depth_camera is not None else 0
        rgb_cam_pos_index = self.rgb_camera.transform_index if self.rgb_camera is not None else 0
        # semanticsegmentation_cam_index = self.semanticsegmentation_camera.index if self.semanticsegmentation_camera is not None else 0
        instancesegmentation_cam_index = self.instancesegmentation_camera.index if self.instancesegmentation_camera is not None else 0
        # Get a random blueprint.
        # 哪些車輛型號可生成
        blueprint_list = get_actor_blueprints(self.world, self._actor_filter, self._actor_generation)
        if not blueprint_list:
            raise ValueError("Couldn't find any blueprints with the specified filters")
        # blueprint = random.choice(blueprint_list)
        blueprint = blueprint_list[2]
        print(blueprint)
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('terramechanics'):
            blueprint.set_attribute('terramechanics', 'true')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            # 隨機生成
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()   

            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)

        # Set up the sensors.
        self.rgb_camera = RGBCamera(self.player, self.hud, self._gamma, self.args)
        self.rgb_camera.transform_index = rgb_cam_pos_index
        self.rgb_camera.set_sensor(rgb_cam_index, notify=False)

        self.depth_camera = DepthCamera(self.player, self.hud, self.args)
        self.depth_camera.set_sensor(depth_cam_index)

        # self.semanticsegmentation_camera = SemanticSegmentationCamera(self.player, self.hud, self.args)
        # self.semanticsegmentation_camera.set_sensor(semanticsegmentation_cam_index)

        self.instancesegmentation_camera = InstanceSegmentationCamera(self.player, self.hud, self.args)
        self.instancesegmentation_camera.set_sensor(instancesegmentation_cam_index)

        self.radar_sensor = RadarSensor(self.player, self.args)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        if self.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.rgb_camera.render(display)
        self.hud.render(display)

    def destroy(self):
        my_actors = [
            self.player,
            self.rgb_camera.sensor,
            self.depth_camera.sensor,
            self.radar_sensor.sensor,
            # self.semanticsegmentation_camera.sensor,
            self.instancesegmentation_camera.sensor,
            ]
        
        all_actors = self.world.get_actors()
        for actor in my_actors:
            if actor is not None:
                exists = any(world_actor.id == actor.id for world_actor in all_actors)
                if exists:
                    if actor.type_id.split('.')[0] == 'sensor':
                        actor.stop()
                    actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._ackermann_enabled = False
        self._ackermann_reverse = 1
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._ackermann_control = carla.VehicleAckermannControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled, world.args.traport)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock, sync_mode):
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False, world.args.traport)
                        world.restart()
                        world.player.set_autopilot(True, world.args.traport)
                    else:
                        world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_v and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_map_layer(reverse=True)
                elif event.key == K_v:
                    world.next_map_layer()
                elif event.key == K_b and pygame.key.get_mods() & KMOD_SHIFT:
                    world.load_map_layer(unload=True)
                elif event.key == K_b:
                    world.load_map_layer()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.rgb_camera.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                # elif event.key == K_g:
                #     world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.rgb_camera.next_sensor()
                elif event.key == K_n:
                    world.rgb_camera.next_sensor()
                elif event.key == K_w and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.constant_velocity_enabled:
                        world.player.disable_constant_velocity()
                        world.constant_velocity_enabled = False
                        world.hud.notification("Disabled Constant Velocity Mode")
                    else:
                        world.player.enable_constant_velocity(carla.Vector3D(17, 0, 0))
                        world.constant_velocity_enabled = True
                        world.hud.notification("Enabled Constant Velocity Mode at 60 km/h")
                elif event.key == K_o:
                    try:
                        if world.doors_are_open:
                            world.hud.notification("Closing Doors")
                            world.doors_are_open = False
                            world.player.close_door(carla.VehicleDoor.All)
                        else:
                            world.hud.notification("Opening doors")
                            world.doors_are_open = True
                            world.player.open_door(carla.VehicleDoor.All)
                    except Exception:
                        pass
                elif event.key == K_t:
                    if world.show_vehicle_telemetry:
                        world.player.show_debug_telemetry(False)
                        world.show_vehicle_telemetry = False
                        world.hud.notification("Disabled Vehicle Telemetry")
                    else:
                        try:
                            world.player.show_debug_telemetry(True)
                            world.show_vehicle_telemetry = True
                            world.hud.notification("Enabled Vehicle Telemetry")
                        except Exception:
                            pass
                elif event.key > K_0 and event.key <= K_9:
                    index_ctrl = 0
                    if pygame.key.get_mods() & KMOD_CTRL:
                        index_ctrl = 9
                    world.rgb_camera.set_sensor(event.key - 1 - K_0 + index_ctrl)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.rgb_camera.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        print("recoder is ON!!!!!!!!!")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.rgb_camera.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled, world.args.traport)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.rgb_camera.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_f:
                        # Toggle ackermann controller
                        self._ackermann_enabled = not self._ackermann_enabled
                        world.hud.show_ackermann_info(self._ackermann_enabled)
                        world.hud.notification("Ackermann Controller %s" %
                                               ("Enabled" if self._ackermann_enabled else "Disabled"))
                    if event.key == K_q:
                        if not self._ackermann_enabled:
                            self._control.gear = 1 if self._control.reverse else -1
                        else:
                            self._ackermann_reverse *= -1
                            # Reset ackermann control
                            self._ackermann_control = carla.VehicleAckermannControl()
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        if not self._autopilot_enabled and not sync_mode:
                            print("WARNING: You are currently in asynchronous mode and could "
                                  "experience some issues with the traffic simulation")
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled, world.args.traport)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
                # Apply control
                if not self._ackermann_enabled:
                    world.player.apply_control(self._control)
                else:
                    world.player.apply_ackermann_control(self._ackermann_control)
                    # Update control to the last one applied by the ackermann controller.
                    self._control = world.player.get_control()
                    # Update hud with the newest ackermann control
                    world.hud.update_ackermann_control(self._ackermann_control)

            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)
                world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            if not self._ackermann_enabled:
                self._control.throttle = min(self._control.throttle + 0.1, 1.00)
            else:
                self._ackermann_control.speed += round(milliseconds * 0.005, 2) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            if not self._ackermann_enabled:
                self._control.brake = min(self._control.brake + 0.2, 1)
            else:
                self._ackermann_control.speed -= min(abs(self._ackermann_control.speed), round(milliseconds * 0.005, 2)) * self._ackermann_reverse
                self._ackermann_control.speed = max(0, abs(self._ackermann_control.speed)) * self._ackermann_reverse
        else:
            if not self._ackermann_enabled:
                self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        if not self._ackermann_enabled:
            self._control.steer = round(self._steer_cache, 1)
            self._control.hand_brake = keys[K_SPACE]
        else:
            self._ackermann_control.steer = round(self._steer_cache, 1)

    def _parse_walker_keys(self, keys, milliseconds, world):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
    
# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

        self._show_ackermann_info = False
        self._ackermann_control = carla.VehicleAckermannControl()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        # compass = world.imu_sensor.compass
        # heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        # heading += 'S' if 90.5 < compass < 269.5 else ''
        # heading += 'E' if 0.5 < compass < 179.5 else ''
        # heading += 'W' if 180.5 < compass < 359.5 else ''
        # colhist = world.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            'World Frame: % d ' % world.world.get_snapshot().frame,
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            # u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            # 'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            # 'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            # 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
            if self._show_ackermann_info:
                self._info_text += [
                    '',
                    'Ackermann Controller:',
                    '  Target speed: % 8.0f km/h' % (3.6*self._ackermann_control.speed),
                ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            # 'Collision:',
            # collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def show_ackermann_info(self, enabled):
        self._show_ackermann_info = enabled

    def update_ackermann_control(self, ackermann_control):
        self._ackermann_control = ackermann_control

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """Helper class to handle text output using pygame"""
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- RadarSensor ---------------------------------------------------------------
# ==============================================================================

import math
import json

class RadarSensor(object):
    def __init__(self, parent_actor, args):
        self.recode_done = False
        self.args = args
        self.sensor = None
        self._parent = parent_actor # 車子本體
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        # 拿藍圖
        bp = world.get_blueprint_library().find('sensor.other.radar')

        # 設定藍圖參數
        attributes = ['horizontal_fov', 'points_per_second', 'range', 'sensor_tick', 'vertical_fov']
        radar_attribute_str = 'Radar params: '
        for idx, attribute in enumerate(attributes):
            bp.set_attribute(attribute, str(self.args.radar[idx]))
            radar_attribute_str += f'\n\t{attribute}: {self.args.radar[idx]}'
        # Show radar params info
        print(radar_attribute_str)

        # 創建藍圖
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                #carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=0.6*bound_z-0.02),
                carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=0.6*bound_z),
                carla.Rotation(pitch=0)),
            attach_to=self._parent)
        
        global g_radar_sensor
        g_radar_sensor = self.sensor

        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        
        global rdr_queue
        rdr_queue.put(radar_data)

# ==============================================================================
# -- RGBCamera -------------------------------------------------------------
# ==============================================================================


class RGBCamera(object):
    def __init__(self, parent_actor, hud, gamma_correction, args):
        self.sensor = None
        self.surface = None
        self.args = args
        self._parent = parent_actor # 車子本體
        self.hud = hud
        self.recording = False
        self.cur_blueprint = None
        
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType
    
        
        self._camera_transforms = [
            # Location x:前後 y:左右 z:上下
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=0.6*bound_z), carla.Rotation(pitch=0)), Attachment.Rigid), # 駕駛視角
            # (carla.Transform(carla.Location(x=-2.0*bound_x, y=+0.0*bound_y, z=2.0*bound_z), carla.Rotation(pitch=8.0)), Attachment.SpringArmGhost), # 第三人稱
            # (carla.Transform(carla.Location(x=+1.9*bound_x, y=+1.0*bound_y, z=1.2*bound_z)), Attachment.SpringArmGhost), # 右前方往車子拍
            # (carla.Transform(carla.Location(x=-2.8*bound_x, y=+0.0*bound_y, z=4.6*bound_z), carla.Rotation(pitch=6.0)), Attachment.SpringArmGhost), # 俯拍
            # (carla.Transform(carla.Location(x=-1.0, y=-1.0*bound_y, z=0.4*bound_z)), Attachment.Rigid), # 左側邊
            ] 
        
            

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)

        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])
            self.cur_blueprint = self.sensors[index][-1]

            global g_camera_sensor
            g_camera_sensor = self.sensor
            # Get K
            global K
            image_w = self.sensors[index][-1].get_attribute('image_size_x').as_int()
            image_h = self.sensors[index][-1].get_attribute('image_size_y').as_int()
            fov_x = self.sensors[index][-1].get_attribute("fov").as_float()
            fov_y = fov_x * image_h / image_w
            K = fusion.build_projection_matrix(image_w, image_h, fov_x, fov_y)
            
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda rgb_image: RGBCamera._parse_image(weak_self, rgb_image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        print("recording:", self.recording)
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, rgb_image):
        self = weak_self()
        if not self:
            return
        
        global rgb_cam_queue
        rgb_cam_queue.put(rgb_image)

        rgb_image.convert(self.sensors[self.index][1])
        array = np.frombuffer(rgb_image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (rgb_image.height, rgb_image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

# class SemanticSegmentationCamera(object):
#     def __init__(self, parent_actor, hud, args):
#         self.args = args
#         self.hud = hud
#         self.sensor = None
#         self._parent = parent_actor
#         self.bp = None

#         bound_x = 0.5 + self._parent.bounding_box.extent.x
#         bound_y = 0.5 + self._parent.bounding_box.extent.y
#         bound_z = 0.5 + self._parent.bounding_box.extent.z
#         Attachment = carla.AttachmentType

#         self._camera_transforms = [
#             # Location x:前後 y:左右 z:上下
#             (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=0.6*bound_z), carla.Rotation(pitch=0)), Attachment.Rigid), # 駕駛視角
#             ] 

#         world = self._parent.get_world()
#         bp_library = world.get_blueprint_library()
#         bp = bp_library.find('sensor.camera.semantic_segmentation')
#         bp.set_attribute('image_size_x', str(self.hud.dim[0]))
#         bp.set_attribute('image_size_y', str(self.hud.dim[1]))
#         self.bp = bp

#         self.index = None
    
#     def set_sensor(self, index, force_respawn=False):
#         index = index % 1

#         needs_respawn = True if self.index is None else force_respawn

#         if needs_respawn:
#             if self.sensor is not None:
#                 self.sensor.destroy()
#             self.sensor = self._parent.get_world().spawn_actor(
#                 self.bp,
#                 self._camera_transforms[0][0],
#                 attach_to=self._parent,
#                 attachment_type=self._camera_transforms[0][1]
#             )
        
#             weak_self = weakref.ref(self)
#             self.sensor.listen(lambda semanticsegmentation_image: SemanticSegmentationCamera._parse_semanticsegmentation_image(weak_self, semanticsegmentation_image))

#         self.index = index
    
#     @staticmethod
#     def _parse_semanticsegmentation_image(weak_self, semanticsegmentation_image):
#         self = weak_self()
#         if not self:
#             return
        
        
#         global semanticsegmentation_cam_queue
#         semanticsegmentation_cam_queue.put(semanticsegmentation_image)

class InstanceSegmentationCamera(object):
    def __init__(self, parent_actor, hud, args):
        self.args = args
        self.hud = hud
        self.sensor = None
        self._parent = parent_actor
        self.bp = None

        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            # Location x:前後 y:左右 z:上下
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=0.6*bound_z), carla.Rotation(pitch=0)), Attachment.Rigid), # 駕駛視角
            ] 

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.instance_segmentation')
        bp.set_attribute('image_size_x', str(self.hud.dim[0]))
        bp.set_attribute('image_size_y', str(self.hud.dim[1]))
        self.bp = bp

        self.index = None
    
    def set_sensor(self, index, force_respawn=False):
        index = index % 1

        needs_respawn = True if self.index is None else force_respawn

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
            self.sensor = self._parent.get_world().spawn_actor(
                self.bp,
                self._camera_transforms[0][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[0][1]
            )
        
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda instancesegmentation_image: InstanceSegmentationCamera._parse_instancesegmentation_image(weak_self, instancesegmentation_image))

        self.index = index
    
    @staticmethod
    def _parse_instancesegmentation_image(weak_self, instancesegmentation_image):
        self = weak_self()
        if not self:
            return
        
        global instancesegmentation_cam_queue
        instancesegmentation_cam_queue.put(instancesegmentation_image)

class DepthCamera(object):
    def __init__(self, parent_actor, hud, args):
        self.args = args
        self.hud = hud
        self.sensor = None
        self._parent = parent_actor
        self.bp = None

        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        self._camera_transforms = [
            # Location x:前後 y:左右 z:上下
            (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=0.6*bound_z), carla.Rotation(pitch=0)), Attachment.Rigid), # 駕駛視角
            ] 

        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.depth')
        bp.set_attribute('image_size_x', str(self.hud.dim[0]))
        bp.set_attribute('image_size_y', str(self.hud.dim[1]))
        self.bp = bp

        self.index = None
    
    def set_sensor(self, index, force_respawn=False):
        index = index % 1

        needs_respawn = True if self.index is None else force_respawn

        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
            self.sensor = self._parent.get_world().spawn_actor(
                self.bp,
                self._camera_transforms[0][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[0][1]
            )
        
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda depth_image: DepthCamera._parse_depth_image(weak_self, depth_image))

        self.index = index
    
    @staticmethod
    def _parse_depth_image(weak_self, depth_image):
        self = weak_self()
        if not self:
            return
        
        global depth_cam_queue
        depth_cam_queue.put(depth_image)

# 第一張天氣的雷達資料 (傳遞給後續天氣使用)
first_weather_radardata = None

def show_projection_and_save_data(world: World, eligible4save, dataset_count, last_instancesegmentation_img, weather_i, all_actors, all_id, wait_frame_count):
    save_to_dataset= False

    radar_data = rdr_queue.get()
    rgb_image_data = rgb_cam_queue.get()
    depth_image_data = depth_cam_queue.get()
    instancesegmentation_image_data = instancesegmentation_cam_queue.get()
    # seg_img_data = semanticsegmentation_cam_queue.get()

    # 確保相同frame
    if len(set([rgb_image_data.frame, depth_image_data.frame, instancesegmentation_image_data.frame, radar_data.frame])) != 1:
        raise ValueError(f"rgb_frame:{rgb_image_data.frame}, depth_frame: {depth_image_data.frame}, instance_frame:{instancesegmentation_image_data.frame}, radar_frame:{radar_data.frame} not exactly the same!!!")
    
    rdr = np.copy(np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')))
    rdr = np.reshape(rdr, (len(radar_data), 4))

    rgb_img = np.copy(np.frombuffer(rgb_image_data.raw_data, dtype=np.dtype('uint8')))
    rgb_img = np.reshape(rgb_img, (rgb_image_data.height, rgb_image_data.width, 4))
    
    depth_image_data.convert(carla.ColorConverter.LogarithmicDepth)
    depth_img = np.copy(np.frombuffer(depth_image_data.raw_data, dtype=np.dtype('uint8')))
    depth_img = np.reshape(depth_img, (depth_image_data.height, depth_image_data.width, 4))
    
    instancesegmentation_img = np.copy(np.frombuffer(instancesegmentation_image_data.raw_data, dtype=np.dtype('uint8')))
    instancesegmentation_img = np.reshape(instancesegmentation_img, (instancesegmentation_image_data.height, instancesegmentation_image_data.width, 4))

    # seg_img_data.convert(carla.ColorConverter.CityScapesPalette)
    # seg_img = np.copy(np.frombuffer(seg_img_data.raw_data, dtype=np.dtype('uint8')))
    # seg_img = np.reshape(seg_img, (seg_img_data.height, seg_img_data.width, 4))

    if eligible4save and (not np.array_equal(last_instancesegmentation_img, instancesegmentation_img) or weather_i != 0):
        save_to_dataset = True
        # 讓人車瞬間靜止
        change_vehicle_walker_behavior(0.0, False, world.world, all_actors, all_id)

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # 不管semanticsegmentation或instancesegmentation，R通道都是物件type ID
    #  semanticsegmentation (0, 0, typeID=14) 若要知道真實顏色可以用 semanticsegmentation_image_data.convert(carla.ColorConverter.CityScapesPalette) 轉換
    #  instancesegmentation (157, 200, typeID=14) 是某台車輛, 而(157, 168, typeID=14) 是另一台車輛, (157, 208, typeID=14) 也是其他車輛, 用unqiue id區分 157-200, 157-168, 157-208
    # ------------------------------------------------------------------------------------------------------------------------------------------

    WAITING_TIMES = 15
    cur_weather_name = list(weather.weathers.keys())[weather_i]
    if save_to_dataset:        
        if wait_frame_count % WAITING_TIMES == 0: 
            # 第一張天氣的雷達資料 (傳遞給後續天氣使用)
            global first_weather_radardata
            if weather_i == 0 and first_weather_radardata == None:
                first_weather_radardata = radar_data
            else:
                radar_data = first_weather_radardata

            save_name = str(client_connect_id) + "_" + f"{(rgb_image_data.frame-weather_i*WAITING_TIMES):07d}" + "_" + cur_weather_name

            # 儲存最原始數據
            with open(f'{world.args.rawdata_path}/rgb/{save_name}.pkl', 'wb') as f:
                rgbData =  myclass.CarlaImage(
                                rgb_image_data.fov,
                                rgb_image_data.width,
                                rgb_image_data.height,
                                bytes(rgb_image_data.raw_data),
                                g_camera_sensor.get_transform().get_inverse_matrix()
                            )
                pickle.dump(rgbData, f)
            with open(f'{world.args.rawdata_path}/radar/{save_name}.pkl', 'wb') as f:
                transform = g_radar_sensor.get_transform()
                radarData = myclass.CarlaRadar(
                    transform.get_matrix(),
                    myclass.CarlaTransform(
                        myclass.CarlaLocation(transform.location.x, transform.location.y, transform.location.z),
                        myclass.CarlaRotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
                    )
                )
                for radar_detection in radar_data:
                    radarData.insert(
                        myclass.CarlaRadarDetection(
                            radar_detection.depth,
                            radar_detection.azimuth,
                            radar_detection.altitude,
                            radar_detection.velocity
                        )
                    )
                pickle.dump(radarData, f)
            with open(f'{world.args.rawdata_path}/depth/{save_name}.pkl', 'wb') as f:
                depthData = myclass.CarlaImage(
                                depth_image_data.fov,
                                depth_image_data.width,
                                depth_image_data.height,
                                bytes(depth_image_data.raw_data)
                            )
                pickle.dump(depthData, f)
            with open(f'{world.args.rawdata_path}/instance_segmentation/{save_name}.pkl', 'wb') as f:
                instanceSegmentationData = myclass.CarlaImage(
                                            instancesegmentation_image_data.fov,
                                            instancesegmentation_image_data.width,
                                            instancesegmentation_image_data.height,
                                            bytes(instancesegmentation_image_data.raw_data)
                                        )
                pickle.dump(instanceSegmentationData, f)
            with open(f'{world.args.rawdata_path}/actors/{save_name}.pkl', 'wb') as f:
                actorData = []
                all_actors = world.world.get_actors()
                for actor in all_actors:
                    if actor.id == world.player.id:
                        continue
                    
                    bbox = actor.bounding_box
                    transform = actor.get_transform()

                    actorData.append(myclass.CarlaActor(
                        actor.id,
                        actor.type_id,
                        myclass.CarlaBoundingBox(
                            myclass.CarlaVector3D(bbox.extent.x, bbox.extent.y, bbox.extent.z),
                            myclass.CarlaLocation(bbox.location.x, bbox.location.y, bbox.location.z),
                            myclass.CarlaRotation(bbox.rotation.pitch, bbox.rotation.yaw, bbox.rotation.roll),
                        ),
                        myclass.CarlaTransform(
                            myclass.CarlaLocation(transform.location.x, transform.location.y, transform.location.z),
                            myclass.CarlaRotation(transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll),
                        ),
                    ))
                if len(actorData) == 0:
                    print(f'{world.args.rawdata_path}/actors/{save_name}.pkl')
                pickle.dump(actorData, f)
                    
            dataset_count+=1

            # Change weather
            weather_i+=1
            new_weather_name = list(weather.weathers.keys())[weather_i % len(weather.weathers)]
            new_weather = weather.weathers[new_weather_name]
            world.world.set_weather(new_weather)

            vehicles = world.world.get_actors().filter('vehicle.*')
            for vehicle in vehicles:
                if not vehicle.is_alive:
                    continue
                # 判斷是否要開燈
                if any(word in new_weather_name for word in ['Noon']):
                    vehicle.set_light_state(carla.VehicleLightState.NONE)
                if any(word in new_weather_name for word in ['Night', 'Rain', 'Early']):
                    vehicle.set_light_state(carla.VehicleLightState.LowBeam)
                    vehicle.set_light_state(carla.VehicleLightState.Position)
                    vehicle.set_light_state(carla.VehicleLightState.Brake)
                if any(word in new_weather_name for word in ['Fog']):
                    vehicle.set_light_state(carla.VehicleLightState.Fog)
            # save_bboxNradar = False
        wait_frame_count+=1
                    
    if world.args.projectionflag:
        rdr = np.copy(np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4')))
        rdr = np.reshape(rdr, (len(radar_data), 4))
        # (vel, azimuth, altitude, depth)    但是官方範例是寫成->(vel, altitude, azimuth, depth) 官方給的畫出來的點會是錯的
        for (vel, azimuth, altitude, depth) in rdr:
            # print(f'仰角:{math.degrees(altitude)}, 方位角:{math.degrees(azimuth)}, 深度:{depth}')
            radar_local = fusion.get_radar_local(altitude, azimuth, depth)
            radar_matrix = g_radar_sensor.get_transform().get_matrix()
            radar_local_to_world = fusion.get_radar_local_to_world(radar_local, radar_matrix)
            camera_inverse_matrix = g_camera_sensor.get_transform().get_inverse_matrix()
            camera_world_to_local = fusion.get_camera_world_to_local(radar_local_to_world, camera_inverse_matrix)
            image_point = fusion.get_image_point(camera_world_to_local, K)
            if (image_point[0]<0) or (image_point[0]>=rgb_image_data.width) or (image_point[1]<0) or (image_point[1]>=rgb_image_data.height):
                continue
            cv2.circle(rgb_img, center=(round(image_point[0]), round(image_point[1])), radius=1, color=(0, 0, 255), thickness=-1)
            # cv2.putText(img=rgb_img, text=str(round(depth, 1)), org=(int(image_point[0])-1, int(image_point[1])), color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.2, lineType=cv2.LINE_AA)
            # cv2.putText(img=rgb_img, text=str(round(vel*3.6, 2)), org=(int(image_point[0])-1, int(image_point[1])), color=(0, 255, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, lineType=cv2.LINE_AA)

        
        cv2.imshow("Windows2Check", rgb_img)
        # cv2.imshow("depthCamera", depth_img)
        # cv2.imshow("segWindows", seg_img)
        # cv2.imshow("insWindows", instancesegmentation_img)
        cv2.waitKey(1)

    return dataset_count, instancesegmentation_img, weather_i, wait_frame_count


# Server重啟 frame會重新跑，怕撞到一樣的檔案名因此加上client connect id區分
client_connect_id = ''

# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    original_settings = None
    synchronous_master = False
    
    file.check_make_dir(os.path.join(path_config.MY_HDD8T, args.rawdata_path), True)
    file.check_make_dir(os.path.join(path_config.MY_HDD8T, args.rawdata_path, 'rgb'))
    file.check_make_dir(os.path.join(path_config.MY_HDD8T, args.rawdata_path, 'radar'))
    file.check_make_dir(os.path.join(path_config.MY_HDD8T, args.rawdata_path, 'depth'))
    file.check_make_dir(os.path.join(path_config.MY_HDD8T, args.rawdata_path, 'actors'))
    file.check_make_dir(os.path.join(path_config.MY_HDD8T, args.rawdata_path, 'instance_segmentation'))
     
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        
        # 重新載入地圖，將物件清空
        client.reload_world()
        time.sleep(0.5)

        # 蒐集可用地圖(排除掉Opt(優化過)) 目前遇到Town03_Opt會有問題  [01, 04, 06, 07]instancesegmentation會穿透山 [02, 05] instancesegmentation還是會看到移除掉的圍欄!
        maps = []
        for map in client.get_available_maps(): 
            if map.split('/')[-1].endswith(('_Opt', '01', '04', '06', '07', '02', '05')): 
                continue
            maps.append(map)
    
        # Town05算大地圖 Town02 比預設地圖大一些而已
        map_idx = random.randint(0, len(maps)-1)
        new_world = maps[map_idx]
        print(f"load world: {new_world}")
        client.load_world(new_world)
        time.sleep(1)

        def preprocess(map):
            if not rdr_queue.empty():
                rdr_queue.queue.clear()
            if not rgb_cam_queue.empty():
                rgb_cam_queue.queue.clear()
            if not depth_cam_queue.empty():
                depth_cam_queue.queue.clear()
            if not instancesegmentation_cam_queue.empty():
                instancesegmentation_cam_queue.queue.clear()

            global client_connect_id
            client_connect_id = str(int(time.time())) + '_' + f"{random.randint(0, 100):03d}"
            print(f"============= client_connect_id: {client_connect_id} =============")

            sim_world = client.get_world()
            traffic_manager = client.get_trafficmanager(args.traport)

            if args.sync:
                original_settings = sim_world.get_settings()
                settings = sim_world.get_settings()

                if not settings.synchronous_mode:
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.03
                sim_world.apply_settings(settings)
                traffic_manager.set_global_distance_to_leading_vehicle(50.0)
                traffic_manager.set_synchronous_mode(True)

            if args.autopilot and not sim_world.get_settings().synchronous_mode:
                print("WARNING: You are currently in asynchronous mode and could "
                    "experience some issues with the traffic simulation")
            
            
            object_labels = [carla.CityObjectLabel.Car, 
                            carla.CityObjectLabel.Truck, 
                            carla.CityObjectLabel.Bus, 
                            carla.CityObjectLabel.Pedestrians,
                            carla.CityObjectLabel.Bicycle,
                            carla.CityObjectLabel.Rider,
                            carla.CityObjectLabel.Train,
                            carla.CityObjectLabel.Motorcycle,
                            carla.CityObjectLabel.Fences, # 圍欄
                            ]
            obj_ids = []
            for label in object_labels:
                for obj in sim_world.get_environment_objects(label):
                    obj_ids.append(obj.id)
            # 事先移除地圖原先上的物件
            sim_world.enable_environment_objects(obj_ids, False)

            vehicles_list = []
            walkers_list = []
            all_id = []

            small_car_bp_ids = ['vehicle.dodge.charger_2020', 'vehicle.dodge.charger_police_2020', 'vehicle.ford.crown', 'vehicle.lincoln.mkz_2020', 'vehicle.mercedes.coupe_2020', 
                                'vehicle.mini.cooper_s_2021', 'vehicle.nissan.patrol_2021', 'vehicle.audi.a2', 'vehicle.audi.etron', 'vehicle.audi.tt', 'vehicle.bmw.grandtourer',
                                'vehicle.chevrolet.impala', 'vehicle.citroen.c3', 'vehicle.dodge.charger_police', 'vehicle.ford.mustang', 'vehicle.jeep.wrangler_rubicon', 
                                'vehicle.lincoln.mkz_2017', 'vehicle.mercedes.coupe', 'vehicle.micro.microlino', 'vehicle.mini.cooper_s', 'vehicle.nissan.micra', 'vehicle.nissan.patrol',
                                'vehicle.seat.leon', 'vehicle.tesla.model3', 'vehicle.toyota.prius']
            medium_car_bp_ids = ['vehicle.volkswagen.t2', 'vehicle.ford.ambulance', 'vehicle.mercedes.sprinter', 'vehicle.volkswagen.t2_2021']
            large_car_bp_ids = ['vehicle.carlamotors.european_hgv', 'vehicle.carlamotors.firetruck', 'vehicle.tesla.cybertruck', 'vehicle.mitsubishi.fusorosa', 'vehicle.carlamotors.carlacola']
            cycle_bp_ids = ['vehicle.harley-davidson.low_rider', 'vehicle.kawasaki.ninja', 'vehicle.vespa.zx125', 'vehicle.yamaha.yzf', 'vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']

            # blueprints = get_actor_blueprints(sim_world, args.filterv, args.generationv)
            # if not blueprints:
            #     raise ValueError("Couldn't find any vehicles with the specified filters")
            blueprintsWalkers = get_actor_blueprints(sim_world, args.filterw, args.generationw)
            if not blueprintsWalkers:
                raise ValueError("Couldn't find any walkers with the specified filters")

            # blueprints = sorted(blueprints, key=lambda bp: bp.id)

            spawn_points = sim_world.get_map().get_spawn_points()
            number_of_spawn_points = len(spawn_points)

            adjust_number_of_vehicles = args.number_of_vehicles
            adjust_number_of_walker = args.number_of_walkers
            if map.endswith('02'):
                adjust_number_of_vehicles = int(args.number_of_vehicles * 1.2) # 增加 20% 的車子數量
                adjust_number_of_walker = int(args.number_of_walkers * 1.2)
            elif map.endswith('03'):
                adjust_number_of_vehicles = int(args.number_of_vehicles * 4.0) # 增加400%
                adjust_number_of_walker = int(args.number_of_walkers * 4.0)
            elif map.endswith('05'):
                adjust_number_of_vehicles = int(args.number_of_vehicles * 2.0) # 增加100%
                adjust_number_of_walker = int(args.number_of_walkers * 2.0)

            if adjust_number_of_vehicles < number_of_spawn_points:
                random.shuffle(spawn_points)
            elif adjust_number_of_vehicles > number_of_spawn_points:
                msg = 'requested %d vehicles, but could only find %d spawn points'
                logging.warning(msg, adjust_number_of_vehicles, number_of_spawn_points)
                adjust_number_of_vehicles = number_of_spawn_points

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            SetAutopilot = carla.command.SetAutopilot
            FutureActor = carla.command.FutureActor

            # # # 測試機車腳踏車 2D Bbox
            # ls = ['vehicle.gazelle.omafiets', 'vehicle.bh.crossbike', 'vehicle.diamondback.century']
            # # ls = ["vehicle.harley-davidson.low_rider", "vehicle.yamaha.yzf", "vehicle.bh.crossbike", "vehicle.diamondback.century", 'vehicle.gazelle.omafiets']
            # blueprints = []
            # for l in ls:
            #     blueprints.append(sim_world.get_blueprint_library().find(l))


            small_car_ratio = 0.4
            medium_car_ratio = 0.2
            large_car_ratio = 0.1
            cycle_ratio = 0.3

            num_of_small_car = int(adjust_number_of_vehicles * small_car_ratio)
            num_of_medium_car = int(adjust_number_of_vehicles * medium_car_ratio)
            num_of_large_car = int(adjust_number_of_vehicles * large_car_ratio)
            num_of_cycle_car = adjust_number_of_vehicles - (num_of_small_car + num_of_medium_car + num_of_large_car)

            print(f"Try to spawn {adjust_number_of_vehicles} vehicles, (small_car: {num_of_small_car}) (medium_car: {num_of_medium_car}) (large_car: {num_of_large_car})  (cycle_car: {num_of_cycle_car})")

            blueprint_ids = random.choices(small_car_bp_ids, k=num_of_small_car) + random.choices(medium_car_bp_ids, k=num_of_medium_car) + random.choices(large_car_bp_ids, k=num_of_large_car) + random.choices(cycle_bp_ids, k=num_of_cycle_car)
            blueprints = []
            for b_id in blueprint_ids:
                blueprints.append(sim_world.get_blueprint_library().find(b_id))

            # --------------
            # Spawn vehicles
            # --------------
            batch = []
            hero = args.hero
            # bp_idx = 0
            for n, transform in enumerate(spawn_points):
                if n >= adjust_number_of_vehicles:
                    break
                # blueprint = random.choice(blueprints)
                blueprint = blueprints[n]
                if blueprint.has_attribute('color'):
                    color = random.choice(blueprint.get_attribute('color').recommended_values)
                    blueprint.set_attribute('color', color)
                if blueprint.has_attribute('driver_id'):
                    driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                    blueprint.set_attribute('driver_id', driver_id)
                if hero:
                    blueprint.set_attribute('role_name', 'hero')
                    hero = False
                else:
                    blueprint.set_attribute('role_name', 'autopilot')

                # spawn the cars and set their autopilot and light state all together
                batch.append(SpawnActor(blueprint, transform)
                    .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))
                
                # bp_idx+=1
                # bp_idx = bp_idx % len(blueprints)
                

            for response in client.apply_batch_sync(batch, synchronous_master):
                if response.error:
                    logging.error(response.error)
                else:
                    vehicles_list.append(response.actor_id)

            # Set automatic vehicle lights update if specified
            if args.car_lights_on:
                all_vehicle_actors = sim_world.get_actors(vehicles_list)
                for actor in all_vehicle_actors:
                    traffic_manager.update_vehicle_lights(actor, True)

            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            percentagePedestriansRunning = 0.0      # how many pedestrians will run
            percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
            if args.seedw:
                sim_world.set_pedestrians_seed(args.seedw)
                random.seed(args.seedw)
            # 1. take all the random locations to spawn
            spawn_points = []
            for i in range(adjust_number_of_walker):
                spawn_point = carla.Transform()
                loc = sim_world.get_random_location_from_navigation()
                if (loc != None):
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_point))
            results = client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = sim_world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put together the walkers and controllers id to get the objects from their id
            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = sim_world.get_actors(all_id)

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            sim_world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(sim_world.get_random_location_from_navigation())
                # max speed
                all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

            print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

            # Example of how to use Traffic Manager parameters
            traffic_manager.global_percentage_speed_difference(30.0)
            all_vehicle_actors = sim_world.get_actors(vehicles_list)
            # 調整各個自動駕駛的車速
            for actor in all_vehicle_actors:
                traffic_manager.set_desired_speed(actor, random.randint(30, 80))
            
            # 0.9.15雷達偵測不到行人 因此調整
            for actor in all_actors:
                if "walker.pedestrian." in actor.type_id:
                    actor.set_collisions(True)
                    actor.set_simulate_physics(True)
            return sim_world, original_settings, all_actors, vehicles_list, walkers_list, all_id, obj_ids 

        sim_world, original_settings, all_actors, vehicles_list, walkers_list, all_id, obj_ids  = preprocess(map=new_world)

        # 載入首張天氣
        weather_name = list(weather.weathers.keys())[0]
        init_weather = weather.weathers[weather_name]
        sim_world.set_weather(init_weather)
        print(f"Init Weather! {init_weather}")
        
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()

        hud = HUD(args.width, args.height)
        world = World(sim_world, hud, args)
        controller = KeyboardControl(world, args.autopilot)

        clock = pygame.time.Clock()

        # speed_factor = args.speed
        # update_freq = 0.1 / speed_factor 
        # weather = Weather(world.world.get_weather())
        # elapsed_time = 0.0  

        world_count = 0
        dataset_count = 1
        last_instancesegmentation_img = None

        weather_i = 0
        
        map_idx = random.randint(0, len(maps)-1)
        wait_frame_count = 0

        global_curtime = time.time()
        global_lastime = time.time()
        
        while True:
            # Server 上帝視角跟蹤車輛
            spectator_transform = carla.Transform(
                world.player.get_transform().transform(carla.Location(x=-4, z=50)), 
                carla.Rotation(yaw=-180, pitch=-90)
            )
            world.world.get_spectator().set_transform(spectator_transform)

            global_curtime = time.time()
            eligible4save = False

            if (global_curtime - global_lastime >= args.interval):
                eligible4save = True

                if weather_i!=0 and weather_i % len(weather.weathers) == 0:
                    change_vehicle_walker_behavior(1.4, True, world.world, all_actors, all_id)
                    # 隨機更改自動駕駛的車速
                    if controller._autopilot_enabled:
                        client.get_trafficmanager(args.traport).set_desired_speed(world.player, random.randint(35, 70))
                    global_lastime = time.time()
                    eligible4save = False
                    weather_i = weather_i % len(weather.weathers)
                    wait_frame_count = 0
                    world_count+=1
                    global first_weather_radardata
                    first_weather_radardata = None
            

            dataset_count, last_instancesegmentation_img, weather_i, wait_frame_count = show_projection_and_save_data(world, eligible4save, dataset_count, last_instancesegmentation_img, weather_i, all_actors, all_id, wait_frame_count)

            # 紀錄多少次就換一張地圖
            if world_count == 50:
                if original_settings:
                    sim_world.apply_settings(original_settings)
                    # 恢復原本地圖該有的物件
                    sim_world.enable_environment_objects(obj_ids, True)

                print('\ndestroying %d vehicles' % len(vehicles_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

                # stop walker controllers (list is [controller, actor, controller, actor ...])
                for i in range(0, len(all_id), 2):
                    all_actors[i].stop()

                print('\ndestroying %d walkers' % len(walkers_list))
                client.apply_batch([carla.command.DestroyActor(x) for x in all_id]) 

                time.sleep(0.5) 
                if world is not None:
                    world.destroy()
                map_idx+=1
                map_idx = map_idx % len(maps)
                new_world = maps[map_idx]
                print(f"ready to Change world: {new_world}")

                client.load_world(new_world)

                time.sleep(1)
                
                sim_world, original_settings, all_actors, vehicles_list, walkers_list, all_id, obj_ids  = preprocess(map=new_world)
                
                hud = HUD(args.width, args.height)
                world = World(sim_world, hud, args)
                controller = KeyboardControl(world, args.autopilot)
                world_count = 0
                continue
 
            if args.sync:
                sim_world.tick()
            clock.tick_busy_loop(60)
            # clock.tick()
            if controller.parse_events(client, world, clock, args.sync):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
        
            # weather control
            # timestamp = sim_world.get_snapshot().timestamp
            # elapsed_time += timestamp.delta_seconds
            # if elapsed_time > update_freq:
            #     weather.tick(speed_factor * elapsed_time)
            #     sim_world.set_weather(weather.weather)
            #     elapsed_time = 0.0         

    finally:

        if original_settings:
            sim_world.apply_settings(original_settings)
            # 恢復原本地圖該有的物件
            sim_world.enable_environment_objects(obj_ids, True)

        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])

        time.sleep(0.5)        

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()
        if args.projectionflag:
            breakpoint
            cv2.destroyAllWindows()
        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================



def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--traport',
        metavar='TP',
        default=8000,
        type=int,
        help='Traffic manager port  (default: 8000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        '--projectionflag',
        action='store_true',
        help='projection flag')
    # 收集資料頻率(幾秒一次)
    argparser.add_argument(
        '--interval',
        metavar='INTERVAL',
        default=60,
        type=int,
        help='recorder interval for camera and radar')
    # Radar參數 
    argparser.add_argument(
        "--radar",
        nargs=5,
        default=[90.0, 200, 20, 0.0, 24.0],
        metavar="Radar params",
        help="Radar params [horizontal_fov, points_per_second, range, sensor_tick, vertical_fov]"
    )
    # 產生車輛數量
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    # 產生行人數量
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '-s', '--speed',
        metavar='FACTOR',
        default=1.0,
        type=float,
        help='rate at which the weather changes (default: 1.0)')
    argparser.add_argument(
        '--rawdata-path',
        metavar='PATH',
        default="raw_data",
        type=str,
        help='save raw_data path (default: raw_data)')
   
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        # vncserver :9 顯示在:9   carlaserver顯示在:9
        os.environ['DISPLAY'] = ':9'

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
