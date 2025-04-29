import math
import numpy as np

# K
def build_projection_matrix(w, h, fov_x, fov_y, is_behind_camera=False):
    K = np.identity(3)
    fx = w / (2.0 * math.tan(math.radians(fov_x) / 2))
    fy = h / (2.0 * math.tan(math.radians(fov_y) / 2))
    if is_behind_camera: 
        K[0, 0] = -fx
        K[1, 1] = -fy
    else:
        K[0, 0] = fx
        K[1, 1] = fy
        
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

# 找交點
def find_intersection(p1, p2, u_boundary):
    u1, v1 = p1
    u2, v2 = p2
    if u1 == u2:  
        return None
    m = (v2 - v1) / (u2 - u1)  
    v_intersect = m * (u_boundary - u1) + v1  
    return [u_boundary, v_intersect]
        
# (u, v)
def get_image_point(cam_loc, K):

    point_camera = np.array([cam_loc[1], -cam_loc[2], cam_loc[0]])
    point_image = np.dot(K, point_camera)
    point_image[0] = np.round((point_image[0] / point_image[2]))
    point_image[1] = np.round((point_image[1] / point_image[2]))
    
    return np.array(point_image[:2], dtype=np.int32)
        
def get_radar_local(alt, azi, depth):
    x = depth * math.cos(alt) * math.cos(azi)
    y = depth * math.cos(alt) * math.sin(azi)
    z = depth * math.sin(alt)

    return [x, y, z]
            

def get_radar_local_to_world(xyz_local, r2w):
    x, y, z = xyz_local[0], xyz_local[1], xyz_local[2]
    
    radar_local = np.array([x, y, z, 1])
    radar_world = np.dot(r2w, radar_local)
    
    return radar_world

def get_camera_world_to_local(xyz_world, w2c):
    return np.dot(w2c, xyz_world)