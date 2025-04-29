import os
import json
from tqdm import tqdm

def load_radar_from_folder(folder_path, N):
    json_data_list = []
    file_names = []

    ls = sorted(os.scandir(folder_path), key = lambda x: x.name)    
    for file in tqdm(ls, total=len(ls), desc='load radar'):
        filename = file.name
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    json_data_list.append(data)
                    file_names.append(filename.split('.')[0])
                    if N != -1 and len(json_data_list) == N:
                        break
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON Decode Error: {filename}, Error: {e}")
            except Exception as e:
                raise RuntimeError(f"讀取文件 {filename} 出錯, Error: {e}")
    
    return json_data_list, file_names

def load_camera_from_folder(folder_path, N):
    json_data_list = []
    jpg_path_list= []
    json_file_names = []
    jpg_file_names = []

    ls = sorted(os.scandir(folder_path), key = lambda x: x.name)
    for file in tqdm(ls, total=len(ls), desc='load camera'):
        filename = file.name
        if filename.endswith(".json"):
            json_file_path = os.path.join(folder_path, filename)

            try:
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    json_data_list.append(data)
                    json_file_names.append(filename.split('.')[0])
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON Decode Error: {filename}, Error: {e}")
            except Exception as e:
                raise RuntimeError(f"讀取文件 {filename} 出錯, Error: {e}")

        elif filename.endswith(".jpg"):
            jpg_path_list.append(os.path.join(folder_path, filename))
            jpg_file_names.append(filename.split('.')[0])
        
        if N != -1 and len(json_data_list) == len(jpg_path_list) == N:
            break
    
    return json_data_list, json_file_names, jpg_path_list, jpg_file_names