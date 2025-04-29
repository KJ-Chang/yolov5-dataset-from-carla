import numpy as np
from config import processing as config


def convert_bbox_to_yolo_format(cls, xmin, ymin, xmax, ymax, img_w, img_h):
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin

    return (cls, x_center/img_w, y_center/img_h, w/img_w, h/img_h)

def yolo_label_xywh_to_4vertexs(x, y, w, h, img_width, img_height):
    flexible = 1.0
    cx = x*img_width
    cy = y*img_height
    bboxw = w*img_width*flexible   # 0.8是為了讓原本的眶縮小一點方便確認
    bboxh = h*img_height*flexible
    

    xmin = int(cx-bboxw/2)
    xmax = int(cx+bboxw/2)
    ymin = int(cy-bboxh/2)
    ymax = int(cy+bboxh/2)

    return xmin, ymin, xmax, ymax

def scaling(v, data_type):
    if data_type == 'depth':
        clipping_value = config.DEPTH_CLIP
        new_v = np.clip(clipping_value - v, 0, clipping_value)
    elif data_type == 'velocity':
        clipping_value = config.VELOCITY_CLIP
        new_v = np.clip(v, 0, clipping_value)
    else:
        raise ValueError(f'Unknown data type: {data_type}')

    new_v*= (255.0 / clipping_value)

    return round(new_v)