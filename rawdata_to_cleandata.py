"""
raw data ---> claen data

"""

import os
import cv2
import sys
import json
import carla
import pickle
import argparse
import numpy as np

from tqdm import tqdm

import utils.file as file
import utils.fusion as fusion
import utils.geometry as geometry
import utils.data_processing as data_processing

import config.file as file_config
import config.path as path_config
import config.dataset as dataset_config
import config.processing as processing_config
import config.visualization as visualization_config

def update_message(msg):
    sys.stdout.write('\033[F')
    sys.stdout.write('\033[K')
    print(msg)

def processing(read_parent_dir = 'raw_data' , write_parent_dir = 'clean_data', check_parent_dir = 'check_data_bboxNradar'):
    generated_files_path = os.path.join(os.getcwd(), write_parent_dir, 'generated_files.json')
    generated_files = file.get_generated_files(generated_files_path)

    override = False

    files_filterted_count = 0
    dirEntrys = sorted(os.scandir(f'{read_parent_dir}/rgb'), key=lambda x: x.name)
    for i, dirEntry in enumerate(tqdm(dirEntrys, desc = "Process raw data")):
        file_pkl = dirEntry.name
        # 之前產生過的就直接跳過
        check_file = file_pkl.split('.')[0]
        if check_file not in generated_files:
            generated_files[check_file] = True
            override = True
        else:
            continue

        with open(f'{read_parent_dir}/rgb/{file_pkl}', 'rb') as f:
            rgb_data = pickle.load(f)
            rgb_img = np.copy(np.frombuffer(rgb_data.raw_data, dtype=np.dtype('uint8')))
            rgb_img = np.reshape(rgb_img, (rgb_data.height, rgb_data.width, 4)) # BGRA
            rgb_img_checkbboxNradar = np.copy(rgb_img)

            fov_x = rgb_data.fov
            fov_y = fov_x * rgb_data.height / rgb_data.width
            K = fusion.build_projection_matrix(rgb_data.width, rgb_data.height, fov_x, fov_y)
                
        with open(f'{read_parent_dir}/radar/{file_pkl}', 'rb') as f:
            radar_data = pickle.load(f)

        with open(f'{read_parent_dir}/actors/{file_pkl}', 'rb') as f:
            actors_data = pickle.load(f)
            vehicle_pedestrian_actors_data = []
            for actor in actors_data:
                type_id = actor.type_id
                if type_id.startswith('vehicle.') or type_id.startswith('walker.pedestrian.'):
                    vehicle_pedestrian_actors_data.append(actor)
            
        with open(f'{read_parent_dir}/instance_segmentation/{file_pkl}', 'rb') as f:
            instance_segmentation_data = pickle.load(f)
            instance_segmentation_img = np.copy(np.frombuffer(instance_segmentation_data.raw_data, dtype=np.dtype('uint8')))
            instance_segmentation_img = np.reshape(instance_segmentation_img, (instance_segmentation_data.height, instance_segmentation_data.width, 4))
            instance_segmentation_img_checkbboxNradar = np.copy(instance_segmentation_img)
            
        # 使用 Instance Segmentation Camera 的資料去畫Bounding Box
        new_instance_map = np.zeros((instance_segmentation_img.shape[0], instance_segmentation_img.shape[1], 1), dtype=np.int32)

        car_tag = [14]
        bus_tag = [16]
        truck_tag = [15]
        rider_tag = [13] 
        bicycle_tag = [19]  
        motorcycle_tag = [18]
        pedestrian_tag = [12]
        
        car_mask = np.isin(instance_segmentation_img[:, :, 2], car_tag)
        bus_mask = np.isin(instance_segmentation_img[:, :, 2], bus_tag)
        truck_mask = np.isin(instance_segmentation_img[:, :, 2], truck_tag)
        rider_mask = np.isin(instance_segmentation_img[:, :, 2], rider_tag)
        bicycle_mask = np.isin(instance_segmentation_img[:, :, 2], bicycle_tag)
        motorcycle_mask = np.isin(instance_segmentation_img[:, :, 2], motorcycle_tag)
        pedestrian_mask = np.isin(instance_segmentation_img[:, :, 2], pedestrian_tag)
        
        unique = instance_segmentation_img[:, :, 0]*256 + instance_segmentation_img[:, :, 1]
        
        # (height, width, 1)
        new_instance_map[car_mask] = (dataset_config.CLASS_IDENTIFIERS['car'] + unique[car_mask])[:, np.newaxis]
        new_instance_map[bus_mask] = (dataset_config.CLASS_IDENTIFIERS['bus'] + unique[bus_mask])[:, np.newaxis]
        new_instance_map[truck_mask] = (dataset_config.CLASS_IDENTIFIERS['truck'] + unique[truck_mask])[:, np.newaxis]
        new_instance_map[rider_mask] = (dataset_config.CLASS_IDENTIFIERS['rider'] + unique[rider_mask])[:, np.newaxis]
        new_instance_map[bicycle_mask] = (dataset_config.CLASS_IDENTIFIERS['bicycle'] + unique[bicycle_mask])[:, np.newaxis]
        new_instance_map[motorcycle_mask] = (dataset_config.CLASS_IDENTIFIERS['motorcycle'] + unique[motorcycle_mask])[:, np.newaxis]
        new_instance_map[pedestrian_mask] = (dataset_config.CLASS_IDENTIFIERS['pedestrian'] + unique[pedestrian_mask])[:, np.newaxis]

        instance_map_4_radar_depth = np.full((instance_segmentation_img.shape[0], instance_segmentation_img.shape[1], 1), -999, dtype=np.float64)
        instance_map_4_radar_velocity = np.full((instance_segmentation_img.shape[0], instance_segmentation_img.shape[1], 1), 999, dtype=np.float64)

        radar_detections = []
        # 有被雷達偵測到的instance_unique_ids
        unique_ids_detectbyradar = set()

        should_continue = False
        msg = ''
        for radar_detection in radar_data.radar_detections:
            depth = radar_detection.depth
            azimuth = radar_detection.azimuth
            altitude = radar_detection.altitude
            velocity = radar_detection.velocity * 3.6
            # 有些速度值非常奇怪，直接過濾掉這個檔案
            if velocity < -200 or velocity > 200:
                should_continue = True
                msg += 'Because velocity.'
                break

            radar_local = fusion.get_radar_local(altitude, azimuth, depth)

            radar_matrix = radar_data.matrix
            radar_local_to_world = fusion.get_radar_local_to_world(radar_local, radar_matrix)

            camera_inverse_matrix = rgb_data.inverse_matrix
            camera_world_to_local = fusion.get_camera_world_to_local(radar_local_to_world, camera_inverse_matrix)

            image_point = fusion.get_image_point(camera_world_to_local, K) # [width, height]
            # 排除超過圖片邊界框
            if (image_point[0] < 0) or (image_point[0] >= rgb_data.width) or (image_point[1] < 0) or (image_point[1] >= rgb_data.height):
                continue
            # 排除不在偵測物件上的雷達點    
            if not new_instance_map[image_point[1], image_point[0], 0]:
                continue
            
            calculate_radarNactor_distance = 0
            c = 0
            # 去計算這個雷達點所在的物件和雷達sensor的距離
            for actor in vehicle_pedestrian_actors_data:
                actor_bbox = carla.BoundingBox(
                    carla.Location(actor.bounding_box.location.x, actor.bounding_box.location.y, actor.bounding_box.location.z),
                    carla.Vector3D(actor.bounding_box.extent.x, actor.bounding_box.extent.y, actor.bounding_box.extent.z),
                )

                actor_transform = carla.Transform(
                    location=carla.Location(x=actor.transform.location.x, y=actor.transform.location.y, z=actor.transform.location.z),
                    rotation=carla.Rotation(pitch=actor.transform.rotation.pitch, yaw=actor.transform.rotation.yaw, roll=actor.transform.rotation.roll),
                )

                radar_sensor_transform = carla.Transform(
                    location=carla.Location(x=radar_data.transform.location.x, y=radar_data.transform.location.y, z=radar_data.transform.location.z),
                    rotation=carla.Rotation(pitch=radar_data.transform.rotation.pitch, yaw=radar_data.transform.rotation.yaw, roll=radar_data.transform.rotation.roll),
                )

                # 雷達點是這個物件
                if actor_bbox.contains(carla.Location(x=radar_local_to_world[0], y=radar_local_to_world[1], z=radar_local_to_world[2]), actor_transform):
                    calculate_radarNactor_distance = radar_sensor_transform.location.distance(actor_transform.location) 
                    c+=1
        
            # 跳過不屬於任何物件的雷達點
            if c == 0:
                continue

            # 跳過在物件的邊界上的雷達點
            if geometry.is_at_the_border(image_point[1], image_point[0], new_instance_map[:, :, 0], clip=1):
                cv2.circle(rgb_img_checkbboxNradar, center=(image_point[0], image_point[1]), radius=1, color=(0, 255, 255), thickness=-1)
                cv2.circle(instance_segmentation_img_checkbboxNradar, center=(image_point[0], image_point[1]), radius=1, color=(255, 0, 255), thickness=-1)
                continue

            # 蒐集有被雷達偵測到的物件unique_id
            unique_ids_detectbyradar.add(new_instance_map[image_point[1], image_point[0], 0])
            # 物件的座標給深度
            instance_map_4_radar_depth[image_point[1], image_point[0], 0] = depth
            # 物件的座標給相對速度
            instance_map_4_radar_velocity[image_point[1], image_point[0], 0] = velocity

            # 蒐集過濾後的雷達點
            radar_detections.append(
                {
                    'Depth': depth, # meters
                    'Azimuth': azimuth, # 弧度
                    'Altitude': altitude, # 弧度 
                    'Velocity': velocity, # km/h  靠近是- 遠離是+
                    'InstanceUniqueID': new_instance_map[image_point[1], image_point[0], 0],
                    'U': image_point[0],
                    'V': image_point[1],
                    'Distance': calculate_radarNactor_distance
                }
            ) 
            cv2.circle(rgb_img_checkbboxNradar, center=(image_point[0], image_point[1]), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.circle(instance_segmentation_img_checkbboxNradar, center=(image_point[0], image_point[1]), radius=1, color=(0, 0, 255), thickness=-1)

        # 前面處裡雷達時發現velocity值有問題，直接跳過這個檔案不做任何事情了
        if should_continue:
            files_filterted_count += 1
            update_message(f"已過濾{files_filterted_count}個檔案, {msg}")
            continue
        
        for idx, ch in enumerate(file_config.CHANNELS):
            # for training model
            file.check_make_dir(os.path.join(path_config.MY_SSD, write_parent_dir, f'labels-{ch}'))
            if ch >= 5:
                file.check_make_dir(os.path.join(path_config.MY_SSD, write_parent_dir, f'labels-mean-{ch}'))

            remove_unique_ids = []
            remove_depth_outlier_vus = []

            # yolo
            labels = []
            labels_mean = []
            should_break = False
            for unique_id in list(unique_ids_detectbyradar):
                y_indices, x_indices = np.where(new_instance_map[:, :, 0] == unique_id)
                
                if len(x_indices) == 0 or len(y_indices) == 0:
                    continue 

                xmin, xmax = np.min(x_indices), np.max(x_indices)
                ymin, ymax = np.min(y_indices), np.max(y_indices)
                
                # bbox太細，跳過
                if (xmax - xmin) <= 2 or (ymax - ymin) <= 2:
                    if idx == 0:
                        remove_unique_ids.append(unique_id)
                    continue

                group_tag = unique_id // dataset_config.BASIS_IDENTIFIERS

                # (cls, x, y, w, h)
                label = data_processing.convert_bbox_to_yolo_format(dataset_config.YOLO_LABEL_IDS[group_tag], xmin, ymin, xmax, ymax, rgb_data.width, rgb_data.height)
                label_mean = data_processing.convert_bbox_to_yolo_format(dataset_config.YOLO_LABEL_IDS[group_tag], xmin, ymin, xmax, ymax, rgb_data.width, rgb_data.height)
                # ch5: depth output ,ch6: depth+velocity output 
                if ch >= 5:
                    non_minus999 = np.where(instance_map_4_radar_depth[y_indices, x_indices, 0] != -999)
                    non_minus999_y_indices, non_minus999_x_indices = y_indices[non_minus999], x_indices[non_minus999] 

                    min_depth = np.min(instance_map_4_radar_depth[non_minus999_y_indices, non_minus999_x_indices, 0]) 
                    max_depth = np.max(instance_map_4_radar_depth[non_minus999_y_indices, non_minus999_x_indices, 0])
                    mean_depth = np.mean(instance_map_4_radar_depth[non_minus999_y_indices, non_minus999_x_indices, 0])

                    range_depth = np.abs(max_depth - min_depth)
                    # 再次檢查物件上有沒有最大值和最小值差很多的雷達點，若有，直接跳過這個檔案不處理了
                    if ((group_tag == 20 or group_tag == 30) and range_depth > 10) or \
                        ((group_tag == 40 or group_tag == 70) and range_depth > 2) or \
                        ((group_tag == 50 or group_tag == 60) and range_depth > 2.5) or \
                        (group_tag == 10 and range_depth > 5):
                        should_break = True
                        msg += 'Because depth'
                        break

                    label = label + (round(min(min_depth, processing_config.DEPTH_CLIP), 2), )
                    label_mean = label_mean + (round(min(mean_depth, processing_config.DEPTH_CLIP), 2), )

                    if ch == 7:
                        non_999 = np.where(instance_map_4_radar_velocity[y_indices, x_indices, 0] != 999)
                        non_999_y_indices, non_999_x_indices = y_indices[non_999], x_indices[non_999]

                        max_velocity = np.max(np.abs(instance_map_4_radar_velocity[non_999_y_indices, non_999_x_indices, 0]))
                        if np.sum(instance_map_4_radar_velocity[non_999_y_indices, non_999_x_indices, 0]<0) > np.sum(instance_map_4_radar_velocity[non_999_y_indices, non_999_x_indices, 0]>0):
                            max_velocity *= -1
                        
                        label = label + (round(min(max_velocity, processing_config.VELOCITY_CLIP) if max_velocity >= 0 else max(max_velocity, -processing_config.VELOCITY_CLIP), 2), )
                        label_mean = label_mean + (round(min(max_velocity, processing_config.VELOCITY_CLIP) if max_velocity >= 0 else max(max_velocity, -processing_config.VELOCITY_CLIP), 2), )

                labels.append(label)
                labels_mean.append(label_mean)

                if idx == 0:
                    # 檢查框框雷達點用
                    cv2.rectangle(rgb_img_checkbboxNradar, (xmin, ymin), (xmax, ymax), visualization_config.COLOR[group_tag//10], 1)
                    cv2.rectangle(instance_segmentation_img_checkbboxNradar, (xmin, ymin), (xmax, ymax), visualization_config.COLOR[group_tag//10], 1)

            if should_break:         
                should_continue = True           
                break  

            # save yolov5 labels dataset
            with open(f'{write_parent_dir}/labels-{ch}/{file_pkl.replace("pkl", "txt")}', 'w') as f:
                for label in labels:
                    f.write(" ".join(map(str, label)) + "\n")
            if ch >= 5:
                with open(f'{write_parent_dir}/labels-mean-{ch}/{file_pkl.replace("pkl", "txt")}', 'w') as f:
                    for label in labels_mean:
                        f.write(" ".join(map(str, label)) + "\n")

            # 排除很細的!
            if len(remove_unique_ids) > 0:
                radar_detections = [detection for detection in radar_detections if detection['InstanceUniqueID'] not in remove_unique_ids]

            # 排除Depth是離群值
            if len(remove_depth_outlier_vus) > 0:
                tmp = []
                for detection in radar_detections:
                    u, v = detection['U'], detection['V']
                    if (v, u) not in remove_depth_outlier_vus:
                        tmp.append(detection)
                        cv2.circle(rgb_img_checkbboxNradar, center=(u, v), radius=1, color=(0, 0, 255), thickness=-1) 
                        cv2.circle(instance_segmentation_img_checkbboxNradar, center=(u, v), radius=1, color=(0, 0, 255), thickness=-1)
                    else:
                        # 點出離群值
                        cv2.circle(rgb_img_checkbboxNradar, center=(u, v), radius=1, color=(0, 255, 255), thickness=-1) 
                        cv2.circle(instance_segmentation_img_checkbboxNradar, center=(u, v), radius=1, color=(0, 255, 255), thickness=-1) 

                radar_detections = tmp

        if should_continue:
            files_filterted_count += 1
            update_message(f"已過濾{files_filterted_count}個檔案, {msg}")
            continue

        # save radar .json
        with open(f'{write_parent_dir}/radar/{file_pkl.replace("pkl", "json")}', 'w') as f:
            json.dump({
                "Matrix": radar_matrix,
                "Data": radar_detections
            }, f, indent=4, default=int)

        # save camera .jpg
        cv2.imwrite(f'{write_parent_dir}/camera/{file_pkl.replace("pkl", "jpg")}', rgb_img)
        # save camera .json
        with open(f'{write_parent_dir}/camera/{file_pkl.replace("pkl", "json")}', 'w') as f:
            json.dump({
                "Inverse_Matrix": rgb_data.inverse_matrix,
                "Camera": {
                            "image_size_x": rgb_data.width,
                            "image_size_y": rgb_data.height,
                            "fov": rgb_data.fov,
                }
            }, f, indent=4)

        # save bboxNradar (很細的 雷達點在check圖片上沒刪掉 看到很正常!!!)
        cv2.imwrite(f'{check_parent_dir}/bboxNradar/{file_pkl.replace("pkl", "jpg")}', rgb_img_checkbboxNradar)
        cv2.imwrite(f'{check_parent_dir}/bboxNradar/{file_pkl.replace("pkl", "png")}', instance_segmentation_img_checkbboxNradar)

    if override:
        with open(generated_files_path, 'w') as f:
            json.dump(generated_files, f, indent=4)

    return len(dirEntrys), files_filterted_count

def main():
    argparser = argparse.ArgumentParser(
        description='Raw data to Clean data.'
    )
    argparser.add_argument(
        '-rpd', '--read-parent-dir',
        metavar='RPD',
        default='raw_data',
        type=str,
        help='Read paraent directory. (defalut raw_data)'
    )
    argparser.add_argument(
        '-wpd', '--write-parent-dir',
        metavar='WPD',
        default='clean_data',
        type=str,
        help='Write parent directory. (default clean_data)'
    )
    argparser.add_argument(
        '-cpd', '--check-parent-dir',
        metavar='-CPD',
        default='check_data_bboxNradar',
        type=str,
        help='Check data bboxNradar directory. (default check_data_bboxNradar)'
    )
    args = argparser.parse_args()

    file.check_make_dir(os.path.join(path_config.MY_SSD, args.write_parent_dir), True)
    file.check_make_dir(os.path.join(path_config.MY_SSD, args.write_parent_dir, 'radar'))
    file.check_make_dir(os.path.join(path_config.MY_SSD, args.write_parent_dir, 'camera'))
    # for checkbboxNradar
    file.check_make_dir(os.path.join(path_config.MY_SSD, args.check_parent_dir), True)
    file.check_make_dir(os.path.join(path_config.MY_SSD, args.check_parent_dir, 'bboxNradar'))

    files_count, files_filterted_count = processing(args.read_parent_dir, args.write_parent_dir, args.check_parent_dir)

    print(f'從 {files_count}個檔案中, 過濾掉 {files_filterted_count} 個檔案')

if __name__ == '__main__':
    main()