"""
clean_data ---> sourcedata
這個檔案是將原先照片(R, G, B)三層擴張成(R, G, B, 雷達深度)四、五、七..層並且儲存成 .npy 或 png 檔案
"""

import os
import json
import argparse
import numpy as np

import config.path as path_config
import config.file as file_config

import utils.file as file
import utils.fusion as fusion
import utils.data_loader as data_loader
import utils.data_processing as data_processing

from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def save_img(args):
    img, path = args
    img.save(path)
        
def save_npy(args):
    path, img = args
    np.save(path, img)

def processing(radar_folder_path, camera_folder_path, args):
    # all radar json file
    radar_data_list, radar_file_names = data_loader.load_radar_from_folder(radar_folder_path, args.n)
    # all camera json file
    camera_data_list, camera_json_file_names, camera_jpg_list, camera_jpg_file_names = data_loader.load_camera_from_folder(camera_folder_path, args.n)

    generated_files_path = os.path.join(os.getcwd(), args.write_parent_dir, 'generated_files.json')
    generated_files = file.get_generated_files(generated_files_path)

    override = False

    BATCH_SIZE = 20000
    batch_number = 1

    if len(radar_data_list) == len(camera_data_list) == len(camera_jpg_list):
        for i in tqdm(range(len(radar_file_names)), desc="Check file names"):
            if radar_file_names[i] == camera_json_file_names[i] == camera_jpg_file_names[i]:
                continue
            raise RuntimeError(f"File name not equal!! When i = {i},  radar_file_names = {radar_file_names[i]}, camera_json_file_names = {camera_json_file_names[i]}, camera_jpg_file_names = {camera_jpg_file_names[i]}")

        file.check_make_dir(os.path.join(path_config.MY_SSD, 'traindata', args.write_parent_dir), True)

        imgs = []
        nps = []
        for ch in file_config.CHANNELS:
            file.check_make_dir(os.path.join(args.write_parent_dir, f'{ch}-channel', 'images'))
            file_extension = file_config.FILE_EXTENSION[ch]
            for radar_data, camera_data, camera_jpg_path in tqdm(zip(radar_data_list, camera_data_list, camera_jpg_list), total=len(radar_data_list), desc=f'{ch}-channel/images'):
                check_file = camera_jpg_path.split("/")[-1].split('.')[0]
                if check_file not in generated_files:
                    generated_files[check_file] = 1
                    override = True
                else:
                    # 有新Channel
                    if generated_files[check_file] < len(file_config.CHANNELS) + 1:
                        if ch <= generated_files[check_file] + 2:
                            continue
                    elif generated_files[check_file] == len(file_config.CHANNELS) + 1:
                        continue

                    generated_files[check_file] += 1 if ch != 7  else 2
                    override = True
                    
                img = np.array(Image.open(camera_jpg_path)) # (height, width, 3)
                if ch >= 4:
                    # 創造一個相同大小的層 Default 0   
                    radar_detect_channel = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) # (0:無點 255:有點)
                    radar_depth_channel = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) # (0:最遠或背景 255:最近)
                    radar_pos_velocity_channel = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) # +遠離 (0: 慢 255:快)
                    radar_neg_velocity_channel = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) # -靠近 (0: 慢 255:快)
                    # (R, G, B) -> (R, G, B, RADAR_DETECT) 
                    img = np.dstack((img, radar_detect_channel))
                    
                    # intrinsics
                    image_w = camera_data['Camera']['image_size_x']
                    image_h = camera_data['Camera']['image_size_y']
                    fov_x = camera_data['Camera']['fov']
                    fov_y = fov_x * image_h / image_w
                    K = fusion.build_projection_matrix(image_w, image_h, fov_x, fov_y)

                    radar_matrix = radar_data['Matrix']
                    camera_inverse_matrix = camera_data['Inverse_Matrix']

                    for r_data in radar_data['Data']:
                        depth = r_data['Depth']
                        azimuth = r_data['Azimuth']
                        altitude = r_data['Altitude']
                        velocity = r_data['Velocity']
                    
                        radar_local = fusion.get_radar_local(altitude, azimuth, depth)
                        radar_local_to_world = fusion.get_radar_local_to_world(radar_local, radar_matrix)
                        camera_world_to_local = fusion.get_camera_world_to_local(radar_local_to_world, camera_inverse_matrix)
                        image_point = fusion.get_image_point(camera_world_to_local, K)

                        if (image_point[0]<0) | (image_point[0]>=image_w) | (image_point[1]<0) | (image_point[1]>=image_h):
                            continue
                        
                        img[image_point[1]][image_point[0]][3] = 255
                        radar_depth_channel[image_point[1]][image_point[0]] = data_processing.scaling(depth, 'depth')
                        
                        if velocity > 0:
                            radar_pos_velocity_channel[image_point[1]][image_point[0]] = data_processing.scaling(velocity, 'velocity')
                        elif velocity < 0:
                            radar_neg_velocity_channel[image_point[1]][image_point[0]] = data_processing.scaling(-velocity, 'velocity')
                    
                    if ch >= 5:
                        # (R, G, B) -> (R, G, B, RADAR_DETECT, RADAR_DEPTH) 
                        img = np.dstack((img, radar_depth_channel))
                    if ch >= 7:
                        # (R, G, B) -> (R, G, B, RADAR_DETECT, RADAR_DEPTH, RADAR_POS_VELOCITY, RADAR_NEG_VELOCITY)
                        img = np.dstack((img, radar_pos_velocity_channel, radar_neg_velocity_channel))
                        

                if file_extension in ['jpg', 'png']:
                    img = Image.fromarray(img, 'RGBA' if file_extension == 'png' else 'RGB')
                    imgs.append((img, os.path.join(args.write_parent_dir, f'{ch}-channel', 'images', f'{camera_jpg_path.split("/")[-1].split(".")[0]}.{file_extension}')))

                    if len(imgs) > 0 and len(imgs) % BATCH_SIZE == 0:
                        with ThreadPoolExecutor() as executor:
                            futures = []
                            for image in imgs:
                                futures.append(executor.submit(save_img, image))
                            
                            for future in futures:
                                future.result()
                            
                        print(f"Batch_num:{batch_number} 存 {len(imgs)} 張")
                        batch_number+=1
                        imgs.clear()

                elif file_extension == 'npy':
                    nps.append((os.path.join(args.write_parent_dir, f'{ch}-channel', 'images', f'{camera_jpg_path.split("/")[-1].split(".")[0]}.{file_extension}'), img))
                    
                    if len(nps) > 0 and len(nps) % BATCH_SIZE == 0:
                        with ThreadPoolExecutor() as executor:
                            futures = []
                            for np_ in nps:
                                futures.append(executor.submit(save_npy, np_))
                            
                            for future in futures:
                                future.result()
                            
                        print(f"Batch_num:{batch_number}存 {len(nps)} 個")
                        batch_number+=1
                        nps.clear()
                else:
                    raise RuntimeError(f'"{args.file_extension}" file extension currently not support')
                
            # 檢查有沒有剩下的
            if len(imgs) != 0 or len(nps) != 0:
                with ThreadPoolExecutor() as executor:
                    futures = []
                    if file_extension in ['jpg', 'png']:
                        for image in imgs:
                            futures.append(executor.submit(save_img, image))
                    elif file_extension == 'npy':
                        for np_ in nps:
                            futures.append(executor.submit(save_npy, np_))
                    for future in futures:
                        future.result()
                print(f"Batch_num:{batch_number}存 {len(futures)} 個")
            
            batch_number = 1
            imgs.clear()
            nps.clear()

            # mv labels
            import shutil
            source_labels_path = os.path.join(path_config.MY_SSD, 'clean_data', f'labels-{ch}') # clean_data/labels-{3,4,5..}
            target_labels_path = os.path.join(args.write_parent_dir, f'{ch}-channel', 'labels') # sourcedata/{3,4,5..}-channel/labels
            print(f'Move labels-{ch} ....')
            # shutil.move(source_labels_path, target_labels_path) # 移動
            shutil.copytree(source_labels_path, target_labels_path) # 複製
            source_labels_path = os.path.join(path_config.MY_SSD, 'clean_data', f'labels-mean-{ch}')

            if ch >=5:
                target_labels_path = os.path.join(args.write_parent_dir, f'{ch}-channel', 'labels-mean')
                shutil.copytree(source_labels_path, target_labels_path) # 複製
        
        if override:
            with open(generated_files_path, 'w') as f:
                json.dump(generated_files, f, indent=4)

        return f"Done!!! Dataset Size:{len(radar_data_list)}"            
    else:
        return f"len(radar_data_list)={len(radar_data_list)} not equl len(camera_data_list)={len(camera_data_list)}"
    
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    
    READ_PARENT_DIR = os.path.join(path_config.MY_SSD, 'clean_data')
    argparser.add_argument(
        '-rpd', '--read-parent-dir',
        metavar='RPD',
        default=READ_PARENT_DIR,
        type=str,
        help=f'Read parent directory. (default: {READ_PARENT_DIR})')
    
    WRITE_PARENT_DIR = 'sourcedata'
    argparser.add_argument(
        '-wpd', '--write-parent-dir',
        metavar='WPD',
        default=WRITE_PARENT_DIR,
        type=str,
        help=f'Write parent directory. (default: {WRITE_PARENT_DIR})')
    
    argparser.add_argument(
        '--n',
        metavar='N',
        default= file_config.DEFAULT_N,
        type=int,
        help=f'Number of Data  (default: {file_config.DEFAULT_N})')
    
    args = argparser.parse_args()

    if args.n == 0 or args.n < -1:
        raise ValueError('n must be equal to -1 or more than 0!')

    radar_folder_path = os.path.join(args.read_parent_dir, 'radar')
    camera_folder_path = os.path.join(args.read_parent_dir, 'camera')

    result = processing(radar_folder_path, camera_folder_path, args)
    print(result)

if __name__ == '__main__':
    main()