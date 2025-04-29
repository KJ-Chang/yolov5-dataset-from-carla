"""
(projected_data and clean_data's labels) to voc_dataset or yolo_dataset

"""

import os
import yaml
import random
import argparse
import numpy as np
import xml.etree.ElementTree as ET

import utils.file as file_utils
import config.file as file_config
import config.path as path_config
import config.dataset as dataset_config

from PIL import Image
from tqdm import tqdm
from xml.dom.minidom import parseString


def bbox_yolo2voc(yolo_label, img_size):
    x, y, w, h = yolo_label
    xmin = int((x-w / 2) * img_size[0])
    ymin = int((y-h / 2) * img_size[1])
    xmax = int((x+w / 2) * img_size[0])
    ymax = int((y+h / 2) * img_size[1])
    return xmin, ymin, xmax, ymax

def get_input_files(yolo_label_files):
    input_files = {
        3: [], # jpg
        4: [], # png
        5: [], # npy
        7: [], # npy
    }

    for file in yolo_label_files:
        if file.endswith('.txt'):
            input_files[3].append(file.replace('txt', 'jpg'))
            input_files[4].append(file.replace('txt', 'png'))
            input_files[5].append(file.replace('txt', 'npy'))
            input_files[7].append(file.replace('txt', 'npy'))

    return input_files

def convet2voc(sourcedata_dir, voc_dataset_dir, args):
    def write(path_of_set, files):
            with open(path_of_set, 'w') as f:
                for file in tqdm(files, desc=f"{path_of_set.split('.')[0].split('/')[-1]}"):
                    f.write(file + '\n')

    images_files = None
    
    for idx, ch in enumerate(file_config.CHANNELS):
        if idx == 0:
        # 只存有物件的!
            yolo_label_files = []
            files = sorted(os.scandir(os.path.join(sourcedata_dir, f'{ch}-channel', 'labels')), key=lambda x: x.name)
            target_count = min(args.n, len(files)) if args.n > 0 else len(files)
            with tqdm(total = target_count, desc = "Check label content") as pbar:
                for file in files:
                    with open(os.path.join(sourcedata_dir, f'{ch}-channel', 'labels', file.name), 'r') as f:
                            lines = f.readlines()
                            if len(lines) > 0:
                                pbar.update(1)
                                yolo_label_files.append(file.name)  
                            if len(yolo_label_files) == target_count:
                                break
                pbar.n = len(yolo_label_files)
                pbar.total = len(yolo_label_files)
                pbar.refresh()


            input_files = get_input_files(yolo_label_files)

        cur_ch_path = f"{voc_dataset_dir}-{ch}"
        file_utils.check_make_dir(cur_ch_path)
        voc_dir_xml = os.path.join(cur_ch_path, 'Annotations')
        file_utils.check_make_dir(voc_dir_xml)
        voc_dir_images = os.path.join(cur_ch_path, f'{file_config.FILE_EXTENSION[ch].upper()}Images')
        file_utils.check_make_dir(voc_dir_images)
        voc_dir_imagesets = os.path.join(cur_ch_path, 'ImageSets', 'Main')
        file_utils.check_make_dir(voc_dir_imagesets)
    
        # Check all path whether exists
        for file in tqdm(input_files[ch], desc=f'[Image-{ch}]  Check all path whether exists'):
            check = os.path.join(sourcedata_dir, f'{ch}-channel', 'images', file)
            if not os.path.exists(check):
                raise RuntimeError(f'File: "{file}" not exists. ({check})')
        
        with tqdm(total=len(input_files[ch]), desc='annotation') as pbar:
            xml_paths = []
            for idx, (img_file, label_file) in enumerate(zip(input_files[ch], yolo_label_files)):
                img_path = os.path.join(sourcedata_dir, f'{ch}-channel', 'images', img_file)
                if ch < 5:
                    img = Image.open(img_path)
                    img_size = img.size
                else:
                    img = np.load(img_path)
                    img_size = (img.shape[1], img.shape[0])
                
                xml_paths.append(os.path.join(voc_dir_xml, label_file.replace('.txt', '.xml')))
                
                with open(os.path.join(os.path.join(sourcedata_dir, f'{ch}-channel', 'labels'), label_file), 'r') as f:
                    lines = f.readlines()
                    
                annotation = ET.Element('annotation')
                
                ET.SubElement(annotation, 'folder').text = f'{file_config.FILE_EXTENSION[ch].upper()}Images'
                ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
                size = ET.SubElement(annotation, 'size')
                ET.SubElement(size, 'width').text = str(img_size[0])
                ET.SubElement(size, 'height').text = str(img_size[1])
                ET.SubElement(size, 'depth').text = str(ch)
                for line in lines:
                    parts = line.strip().split()
                    obj = ET.SubElement(annotation, 'object')
                    class_id = int(parts[0])
                    ET.SubElement(obj, 'name').text = str(dataset_config.CLASS_ID_NAMES[class_id])
                    bndbox = ET.SubElement(obj, 'bndbox')
                    bbox = list(map(float, parts[1:5]))
                    xmin, ymin, xmax, ymax = bbox_yolo2voc(bbox, img_size)
                    ET.SubElement(bndbox, 'xmin').text = str(xmin)
                    ET.SubElement(bndbox, 'ymin').text = str(ymin)
                    ET.SubElement(bndbox, 'xmax').text = str(xmax)
                    ET.SubElement(bndbox, 'ymax').text = str(ymax)
                    radar = ET.SubElement(obj, 'radar')
                    if ch >= 5:
                        depth = parts[5]
                        ET.SubElement(radar, 'depth').text = str(depth)
                    if ch >= 7:
                        velocity = parts[6]
                        ET.SubElement(radar, 'velocity').text = str(velocity)
                
                rough_string = ET.tostring(annotation, 'utf-8')
                reparsed = parseString(rough_string)

                with open(xml_paths[idx], 'w') as f:
                    f.write(reparsed.toprettyxml(indent="  "))

                #----------
                # Images
                #----------
                file_utils.create_link(img_path, os.path.join(voc_dir_images, img_file))     
                pbar.update(1) 

        #---------------
        # ImageSets/Main
        #---------------
        train_path = os.path.join(voc_dir_imagesets, 'train.txt')
        val_path = os.path.join(voc_dir_imagesets, 'val.txt')
        test_path = os.path.join(voc_dir_imagesets, 'test.txt')

        # 首次
        if images_files == None:
            images_files = [f.name.split('.')[0] for f in sorted(os.scandir(voc_dir_images), key=lambda x: x.name)]
            random.shuffle(images_files)
        
        # Calculate the number of train/val/test
        train_count = int(len(images_files) * dataset_config.TRAIN_RATIO)
        val_count = int(len(images_files) * dataset_config.VAL_RATIO)

        train_files = images_files[:train_count]
        val_files = images_files[train_count:train_count+val_count]
        test_files = images_files[train_count+val_count:]

        # Write
        write(train_path, train_files)
        write(val_path, val_files)
        write(test_path, test_files)

def convet2yolo(sourcedata_dir, yolo_dataset_dir, args):
    # Create images link to train, val, test
    def file_link(files, img_path, lbl_path, ch):
        for lbl_file in tqdm(files, desc=img_path.split('/')[-1]): 
            ori_img_path = os.path.join(sourcedata_dir, f'{ch}-channel', 'images', lbl_file.replace('txt', file_config.FILE_EXTENSION[ch])) # txt -> jpg, png, npy..
            ori_lbl_path = os.path.join(sourcedata_dir, f'{ch}-channel', 'labels', lbl_file)
            new_img_path =  os.path.join(img_path, lbl_file.replace('txt', file_config.FILE_EXTENSION[ch]))
            new_lbl_path = os.path.join(lbl_path, lbl_file)

            file_utils.create_link(ori_img_path, new_img_path)
            file_utils.create_link(ori_lbl_path, new_lbl_path)
    
    for idx, ch in enumerate(file_config.CHANNELS):
        if idx == 0:
            yolo_label_files = [f.name for f in sorted(os.scandir(os.path.join(sourcedata_dir, f'{ch}-channel', 'labels')), key=lambda x: x.name)]
            random.shuffle(yolo_label_files)
            
            if args.n != -1:
                yolo_label_files = yolo_label_files[:args.n]

            input_files = get_input_files(yolo_label_files)

            # Calculate the number of train/val/test 
            train_count = int(len(yolo_label_files) * dataset_config.TRAIN_RATIO)
            val_count = int(len(yolo_label_files) * dataset_config.VAL_RATIO)
            
            train_label_files = yolo_label_files[:train_count]
            val_label_files = yolo_label_files[train_count:train_count+val_count]
            test_label_files = yolo_label_files[train_count+val_count:]
        
        # Check all path whether exists
        for file in tqdm(input_files[ch], desc=f'[Image-{ch}]  Check all path whether exists'):
            check = os.path.join(sourcedata_dir, f'{ch}-channel', 'images', file)
            if not os.path.exists(check):
                raise RuntimeError(f'File: "{file}" not exists. ({check})')
            
        cur_ch_path = f"{yolo_dataset_dir}-{ch}"
        file_utils.check_make_dir(cur_ch_path)
        i_tr = os.path.join(cur_ch_path, 'images', 'train')
        file_utils.check_make_dir(i_tr)
        i_v = os.path.join(cur_ch_path, 'images', 'val')
        file_utils.check_make_dir(i_v)
        i_te = os.path.join(cur_ch_path, 'images', 'test')
        file_utils.check_make_dir(i_te)
        l_tr = os.path.join(cur_ch_path, 'labels', 'train')
        file_utils.check_make_dir(l_tr)
        l_v = os.path.join(cur_ch_path, 'labels', 'val')
        file_utils.check_make_dir(l_v)
        l_te = os.path.join(cur_ch_path, 'labels', 'test')
        file_utils.check_make_dir(l_te)

        # Link
        file_link(train_label_files, i_tr, l_tr, ch)
        file_link(val_label_files, i_v, l_v, ch)
        file_link(test_label_files, i_te, l_te, ch)

        # create data.yaml
        data_yaml = {
            'train': i_tr,
            'val': i_v,
            'nc': len(dataset_config.CLASS_ID_NAMES),
            'names': [name for _, name in dataset_config.CLASS_ID_NAMES.items()]
        }
        with open(f'{cur_ch_path}/data.yaml', 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

def main():
    argparser = argparse.ArgumentParser(
        description='This script is a folder converted into XXX format')
    DEFAULT_FORMAT = 'yolo'
    argparser.add_argument(
        '--format',
        metavar='FORMAT',
        default=DEFAULT_FORMAT,
        type=str,
        help=f'training dataset format  (default: {DEFAULT_FORMAT})')
    
    argparser.add_argument(
        '--n',
        metavar='N',
        default= file_config.DEFAULT_N,
        type=int,
        help=f'Number of Data  (default: {file_config.DEFAULT_N})')
    args = argparser.parse_args()

    # Check n
    if args.n <= 0 and args.n != -1:
        raise ValueError(f"The value N must be greater than 1!")

    sourcedata_dir = os.path.join(path_config.MY_SSD, 'traindata', 'sourcedata')

    if args.format == 'yolo':
        yolo_dataset_dir = os.path.join(path_config.MY_SSD, 'traindata', 'YOLOcarla')
        convet2yolo(sourcedata_dir, yolo_dataset_dir, args)
    elif args.format == 'voc':
        voc_dataset_dir = os.path.join(path_config.MY_SSD, 'traindata', 'VOCcarla')
        convet2voc(sourcedata_dir, voc_dataset_dir, args)
    else:
        raise ValueError(f"'{args.format}' format conversion is not currently supported!!")

if __name__ == "__main__":
    main()