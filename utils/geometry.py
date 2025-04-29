import numpy as np

def is_at_the_border(v, u, instance_data: np.array, clip = 1):
    # instance_data[v][u]
    shape = instance_data.shape
    middle, top, bottom, left, right = instance_data[v, u], None, None, None, None 
    
    # 找出上下左右的instance id 
    if v - clip > 0:
        left = instance_data[v-clip][u]
    if v + clip < shape[0]:
        right = instance_data[v+clip][u]
    if u - clip > 0:
        top = instance_data[v][u-clip]
    if u + clip < shape[1]:
        bottom = instance_data[v][u+clip]

    return len(set([middle, top, bottom, left, right])) > 1